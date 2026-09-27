"""The clean pipeline: from Freddie Mac files to tensors, one named step at a time.

Every step returns a new frame you can inspect before moving on. The
steps are deliberately small so that a leak, when there is one, lives in
exactly one place.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from freddie_layout import (
    CREDIT_SCORE_MISSING,
    DEFAULT_ZERO_BALANCE_CODES,
    DQ_STATUS_CAP,
    DQ_STATUS_REO,
    DTI_MISSING,
    LTV_MISSING,
    ORIGINATION_COLUMNS,
    PERFORMANCE_COLUMNS,
)

LABEL_HORIZON_MONTHS = 12
SERIOUS_DELINQUENCY = 3  # 90+ days past due
LABEL = "default_next_12m"

FEATURE_COLUMNS = [
    "credit_score",
    "orig_ltv",
    "orig_dti",
    "orig_rate",
    "loan_age",
    "upb_share_remaining",
    "dq_status_now",
    "dq_status_prev",
    "months_late_last_12",
    "worst_status_last_12",
]


def month_number(periods: pd.Series) -> pd.Series:
    """Monthly periods as integers, so month arithmetic is plain subtraction."""
    return pd.Series(periods.array.asi8, index=periods.index)


def to_month(yyyymm: pd.Series) -> pd.Series:
    return pd.to_datetime(yyyymm, format="%Y%m").dt.to_period("M")


# --- Step 1: load -----------------------------------------------------------


def read_positional(files: list[Path], columns: dict[int, str]) -> pd.DataFrame:
    frames = [
        pd.read_csv(
            f,
            sep="|",
            header=None,
            usecols=list(columns),
            dtype=str,
            keep_default_na=False,
        )
        for f in files
    ]
    return pd.concat(frames, ignore_index=True).rename(columns=columns)


def load_origination(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("sample_orig_*.txt"))
    raw = read_positional(files, ORIGINATION_COLUMNS)
    assert raw["loan_id"].str.startswith("F").all(), "loan_id is not where the layout says"

    origination = pd.DataFrame({"loan_id": raw["loan_id"]})
    origination["first_payment_month"] = to_month(raw["first_payment_month"])
    origination["credit_score"] = pd.to_numeric(
        raw["credit_score"].replace(CREDIT_SCORE_MISSING, ""), errors="coerce"
    )
    origination["orig_dti"] = pd.to_numeric(raw["orig_dti"].replace(DTI_MISSING, ""), errors="coerce")
    origination["orig_ltv"] = pd.to_numeric(raw["orig_ltv"].replace(LTV_MISSING, ""), errors="coerce")
    origination["orig_upb"] = pd.to_numeric(raw["orig_upb"])
    origination["orig_rate"] = pd.to_numeric(raw["orig_rate"])
    return origination


def load_performance(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("sample_svcg_*.txt"))
    raw = read_positional(files, PERFORMANCE_COLUMNS)
    assert raw["loan_id"].str.startswith("F").all(), "loan_id is not where the layout says"

    performance = pd.DataFrame({"loan_id": raw["loan_id"]})
    performance["period"] = to_month(raw["period"])

    status_text = raw["dq_status_raw"].str.strip()
    status_text = status_text.replace(DQ_STATUS_REO, str(DQ_STATUS_CAP))
    performance["dq_status"] = pd.to_numeric(status_text, errors="coerce").clip(upper=DQ_STATUS_CAP)

    performance["current_upb"] = pd.to_numeric(raw["current_upb"], errors="coerce")
    performance["loan_age"] = pd.to_numeric(raw["loan_age"], errors="coerce")
    performance["zero_balance_code"] = raw["zero_balance_code"].str.strip()
    performance["exits_in_default"] = performance["zero_balance_code"].isin(DEFAULT_ZERO_BALANCE_CODES)
    return performance


# --- Step 2: order the panel ------------------------------------------------


def sort_panel(performance: pd.DataFrame) -> pd.DataFrame:
    return performance.sort_values(["loan_id", "period"], ignore_index=True)


def complete_monthly_calendar(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per loan per month from first to last report, gaps included.

    Months with no report get NaN values and is_reported = False, so that
    shift(1) always means "one month earlier" and never "one row earlier".
    """
    first = panel.groupby("loan_id")["period"].min()
    last = panel.groupby("loan_id")["period"].max()
    n_months = month_number(last) - month_number(first) + 1

    calendar = pd.DataFrame({"loan_id": first.index.repeat(n_months)})
    offset = calendar.groupby("loan_id").cumcount().to_numpy()
    first_month = month_number(first).loc[calendar["loan_id"]].to_numpy()
    calendar["period"] = pd.PeriodIndex.from_ordinals(first_month + offset, freq="M")

    completed = calendar.merge(panel, on=["loan_id", "period"], how="left", indicator=True)
    completed["is_reported"] = completed.pop("_merge") == "both"
    completed["exits_in_default"] = completed["exits_in_default"].fillna(False).astype(bool)
    completed["zero_balance_code"] = completed["zero_balance_code"].fillna("")
    return completed


# --- Step 3: the label ------------------------------------------------------


def build_label(panel: pd.DataFrame, data_end: pd.Period) -> pd.DataFrame:
    """Did the loan reach 90+ days past due, or exit in default, in months t+1..t+12?

    The only negative shifts in the whole project live here, because the
    label is the only thing allowed to look forward.
    """
    by_loan = panel.groupby("loan_id", sort=False)
    future_status = pd.concat(
        [by_loan["dq_status"].shift(-k) for k in range(1, LABEL_HORIZON_MONTHS + 1)], axis=1
    )
    future_default_exit = pd.concat(
        [by_loan["exits_in_default"].shift(-k) for k in range(1, LABEL_HORIZON_MONTHS + 1)], axis=1
    )
    worst_future_status = future_status.max(axis=1)
    exits_in_default_within = future_default_exit.fillna(False).astype(bool).any(axis=1)

    labelled = panel.copy()
    labelled[LABEL] = ((worst_future_status >= SERIOUS_DELINQUENCY) | exits_in_default_within).astype(int)

    # The label is only known once its window has passed, or once the loan
    # has closed inside the window.
    window_end = month_number(panel["period"]) + LABEL_HORIZON_MONTHS
    loan_last_month = month_number(by_loan["period"].transform("max"))
    loan_closed = by_loan["zero_balance_code"].transform(lambda codes: (codes != "").any())
    labelled["label_is_mature"] = (window_end <= data_end.ordinal) | (
        loan_closed & (loan_last_month <= window_end)
    )
    return labelled


def select_scoring_population(labelled: pd.DataFrame) -> pd.DataFrame:
    """The rows a PD model would actually score: active, performing, reported, mature."""
    keep = (
        labelled["is_reported"]
        & (labelled["dq_status"] < SERIOUS_DELINQUENCY)
        & (labelled["zero_balance_code"] == "")
        & labelled["label_is_mature"]
    )
    return labelled.loc[keep]


# --- Step 4: features -------------------------------------------------------


def build_features(panel: pd.DataFrame, origination: pd.DataFrame) -> pd.DataFrame:
    """Everything known at the end of month t, and nothing after it.

    Expects a panel sorted by (loan_id, period) with a complete monthly
    calendar. Returns a frame on the same index as the panel.
    """
    by_loan = panel.groupby("loan_id", sort=False)
    origination_by_loan = origination.set_index("loan_id")
    features = panel[["loan_id", "period"]].copy()

    # Fixed at origination, so known at every later month.
    for column in ["credit_score", "orig_ltv", "orig_dti", "orig_rate"]:
        features[column] = panel["loan_id"].map(origination_by_loan[column])

    # Reported for month t itself, known at the end of month t.
    features["loan_age"] = panel["loan_age"]
    orig_upb = panel["loan_id"].map(origination_by_loan["orig_upb"])
    features["upb_share_remaining"] = panel["current_upb"] / orig_upb
    features["dq_status_now"] = panel["dq_status"]

    # History: a positive shift and trailing windows, always within a loan.
    features["dq_status_prev"] = by_loan["dq_status"].shift(1)

    is_late = (panel["dq_status"] > 0).astype(float).where(panel["dq_status"].notna())
    late_by_loan = is_late.groupby(panel["loan_id"], sort=False)
    features["months_late_last_12"] = (
        late_by_loan.rolling(12, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    status_by_loan = panel["dq_status"].groupby(panel["loan_id"], sort=False)
    features["worst_status_last_12"] = (
        status_by_loan.rolling(12, min_periods=1).max().reset_index(level=0, drop=True)
    )
    return features


def assemble_rows(labelled: pd.DataFrame, features: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    population = select_scoring_population(labelled)
    return population[["loan_id", "period", LABEL]].join(features[feature_columns])


# --- Step 5: split by time --------------------------------------------------


def split_by_time(rows: pd.DataFrame, windows: dict[str, tuple[str, str]]) -> dict[str, pd.DataFrame]:
    splits = {}
    for name, (start, end) in windows.items():
        in_window = (rows["period"] >= pd.Period(start, "M")) & (rows["period"] <= pd.Period(end, "M"))
        splits[name] = rows.loc[in_window]
    return splits


# --- Step 6: preprocessing fitted on train only -----------------------------


@dataclass(frozen=True)
class Preprocessing:
    columns: list[str]
    medians: pd.Series
    means: pd.Series
    stds: pd.Series


def fit_preprocessing(train: pd.DataFrame, columns: list[str]) -> Preprocessing:
    medians = train[columns].median()
    filled = train[columns].fillna(medians)
    return Preprocessing(columns, medians, filled.mean(), filled.std().replace(0, 1.0))


def apply_preprocessing(frame: pd.DataFrame, prep: Preprocessing) -> np.ndarray:
    filled = frame[prep.columns].fillna(prep.medians)
    return ((filled - prep.means) / prep.stds).to_numpy(dtype=np.float32, copy=True)


# --- Step 7: tensors --------------------------------------------------------


def to_tensors(frame: pd.DataFrame, prep: Preprocessing) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    """Features, labels, and the keys that trace each tensor row to its source."""
    x = torch.from_numpy(apply_preprocessing(frame, prep))
    y = torch.from_numpy(frame[LABEL].to_numpy(dtype=np.float32, copy=True))
    keys = frame[["loan_id", "period"]].reset_index(drop=True)
    return x, y, keys
