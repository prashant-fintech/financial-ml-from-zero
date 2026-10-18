"""Reuse part 2 verbatim; restrict observations before building any labels."""

from pathlib import Path
import sys

import pandas as pd

PART_2 = Path(__file__).resolve().parents[1] / "part-02-leakage"
if not (PART_2 / "pipeline.py").exists():
    raise RuntimeError("Keep part-03-baseline beside part-02-leakage in the companion repository.")
sys.path.insert(0, str(PART_2))

from leakage_checks import assert_no_calendar_gaps, assert_sorted_panel, assert_split_gaps  # noqa: E402
from pipeline import (  # noqa: E402
    FEATURE_COLUMNS, LABEL, Preprocessing, apply_preprocessing, assemble_rows,
    build_features, build_label, complete_monthly_calendar, fit_preprocessing,
    load_origination, load_performance, sort_panel, split_by_time, to_tensors,
)

WINDOWS = {
    "train": ("2005-06", "2007-12"),
    "valid": ("2009-01", "2009-12"),
    "test": ("2011-01", "2011-12"),
}


def prepare_rows(data_dir: Path, as_of: str) -> pd.DataFrame:
    for pattern in ("sample_orig_*.txt", "sample_svcg_*.txt"):
        if not list(data_dir.glob(pattern)):
            raise ValueError(f"No {pattern} files in {data_dir}. Run part 2's generator first.")
    origination = load_origination(data_dir)
    performance = load_performance(data_dir)
    cutoff = pd.Period(as_of, "M")
    # The development command never constructs 2011 examples or their labels.
    performance = performance.loc[performance["period"] <= cutoff]
    if performance.empty or performance["period"].max() < cutoff:
        raise ValueError(f"Data must include reports through {as_of}.")
    origination = origination.loc[origination["first_payment_month"] <= cutoff]
    panel = complete_monthly_calendar(sort_panel(performance))
    assert_sorted_panel(panel)
    assert_no_calendar_gaps(panel)
    if not panel["is_reported"].all() or panel["dq_status"].isna().any():
        raise ValueError("This example requires complete monthly reports and known delinquency status.")
    # An unexplained end of reporting is censoring, not a known negative.
    endings = panel.groupby("loan_id", sort=False).tail(1)
    unexplained = (endings["period"] < cutoff) & endings["zero_balance_code"].eq("")
    if unexplained.any():
        raise ValueError("A loan stops reporting before the cutoff without a closure code.")
    labelled = build_label(panel, cutoff)
    features = build_features(panel, origination)
    return assemble_rows(labelled, features, FEATURE_COLUMNS)


def development_splits(data_dir: Path):
    rows = prepare_rows(data_dir, "2010-12")
    splits = split_by_time(rows, {k: WINDOWS[k] for k in ("train", "valid")})
    if any(frame.empty or frame[LABEL].nunique() != 2 for frame in splits.values()):
        raise ValueError("Train and validation must each contain both outcomes. Use 12,000 loans.")
    assert_split_gaps(splits)
    return splits


def prep_to_dict(prep: Preprocessing) -> dict:
    return {"columns": prep.columns, **{
        key: getattr(prep, key).to_dict() for key in ("medians", "means", "stds")
    }}


def prep_from_dict(value: dict) -> Preprocessing:
    return Preprocessing(value["columns"], *(pd.Series(value[k]) for k in ("medians", "means", "stds")))
