"""Write a synthetic panel in the Freddie Mac sample-file layout.

The real files need a free registration and cannot be redistributed, so
this script lets the pipeline run end to end without them. Loans are
simulated with a simple delinquency state machine whose monthly risk
depends on credit score, LTV, DTI, a borrower-level random effect, and a
macro shock centred on early 2009.

The files it writes are read by exactly the same loader as the real
ones. Numbers produced on this panel are illustrations, not estimates of
real mortgage risk.

Usage:
    python make_synthetic_freddie.py --out data/synthetic --loans 12000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from freddie_layout import (
    ORIGINATION_COLUMNS,
    ORIGINATION_WIDTH,
    PERFORMANCE_COLUMNS,
    PERFORMANCE_WIDTH,
)

FIRST_VINTAGE = pd.Period("2005-01", freq="M")
LAST_VINTAGE = pd.Period("2010-12", freq="M")
DATA_END = pd.Period("2012-12", freq="M")
CRISIS_PEAK = pd.Period("2009-01", freq="M")


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def market_rate(period: pd.Period) -> float:
    """A crude mortgage rate path: ~6% until 2008, falling to ~4.3% by 2012."""
    years_after_2008 = max(0.0, (period.year + period.month / 12) - 2008.0)
    return 6.2 - min(1.9, 0.45 * years_after_2008)


def crisis_shock(period: pd.Period) -> float:
    months_from_peak = (period - CRISIS_PEAK).n
    return 1.6 * np.exp(-(months_from_peak**2) / (2 * 14.0**2))


def simulate_loan_attributes(rng: np.random.Generator, n_loans: int) -> pd.DataFrame:
    n_vintage_months = (LAST_VINTAGE - FIRST_VINTAGE).n + 1
    first_payment = [
        FIRST_VINTAGE + int(k) for k in rng.integers(0, n_vintage_months, n_loans)
    ]
    credit_score = np.clip(rng.normal(725, 50, n_loans), 520, 840).round()
    orig_ltv = np.clip(rng.normal(74, 13, n_loans), 25, 97).round()
    orig_dti = np.clip(rng.normal(34, 9, n_loans), 8, 55).round()
    orig_upb = (np.exp(rng.normal(12.2, 0.45, n_loans)) / 1000).round() * 1000
    rate_spread = (740 - credit_score) / 100 * 0.6
    orig_rate = np.array(
        [market_rate(p) for p in first_payment]
    ) + rate_spread + rng.normal(0, 0.2, n_loans)

    loans = pd.DataFrame(
        {
            "first_payment_month": first_payment,
            "credit_score": credit_score,
            "orig_ltv": orig_ltv,
            "orig_dti": orig_dti,
            "orig_upb": orig_upb,
            "orig_rate": orig_rate.round(3),
            "frailty": rng.normal(0, 0.6, n_loans),
        }
    )
    loans["vintage_year"] = [p.year for p in loans["first_payment_month"]]
    loans["loan_id"] = [
        f"F{p.year % 100:02d}Q{p.quarter}{i:07d}"
        for i, p in enumerate(loans["first_payment_month"])
    ]
    # About 1% of scores are reported as unavailable, as in the real data.
    missing_score = rng.random(n_loans) < 0.01
    loans["credit_score_reported"] = np.where(
        missing_score, "9999", loans["credit_score"].astype(int).astype(str)
    )
    return loans


def monthly_miss_probability(loan: pd.Series, period: pd.Period) -> float:
    logit = (
        -6.3
        + 0.9 * (720 - loan["credit_score"]) / 50
        + 0.35 * (loan["orig_ltv"] - 75) / 10
        + 0.25 * (loan["orig_dti"] - 34) / 10
        + crisis_shock(period) * (0.6 + loan["orig_ltv"] / 100)
        + loan["frailty"]
    )
    return sigmoid(logit)


def simulate_performance(rng: np.random.Generator, loan: pd.Series) -> list[dict]:
    rows: list[dict] = []
    status = 0
    upb = float(loan["orig_upb"])
    period = loan["first_payment_month"]
    age = 0

    while period <= DATA_END:
        zero_balance_code = ""
        u = rng.random()

        if status == 0:
            p_miss = monthly_miss_probability(loan, period)
            refinance_incentive = loan["orig_rate"] - market_rate(period)
            p_prepay = 0.006 + (0.012 if refinance_incentive > 0.75 else 0.0)
            if u < p_prepay:
                zero_balance_code = "01"
            elif u < p_prepay + p_miss:
                status = 1
        elif status == 1:
            status = 2 if u < 0.45 else (0 if u < 0.90 else 1)
        elif status == 2:
            status = 3 if u < 0.60 else (0 if u < 0.90 else 2)
        else:
            if status >= 6 and u < 0.12:
                zero_balance_code = "09" if rng.random() < 0.7 else "03"
            elif u < 0.05:
                status = 0
            elif u < 0.85:
                status = status + 1

        if status == 0 and zero_balance_code == "":
            upb = upb * 0.9985

        is_reo = zero_balance_code == "09" and rng.random() < 0.5
        rows.append(
            {
                "loan_id": loan["loan_id"],
                "period": period.strftime("%Y%m"),
                "current_upb": 0.0 if zero_balance_code else round(upb, 2),
                "dq_status_raw": "RA" if is_reo else str(status),
                "loan_age": age,
                "zero_balance_code": zero_balance_code,
                "zero_balance_month": period.strftime("%Y%m") if zero_balance_code else "",
            }
        )
        if zero_balance_code:
            break
        period = period + 1
        age += 1
    return rows


def to_positional_frame(frame: pd.DataFrame, columns: dict[int, str], width: int) -> pd.DataFrame:
    """Lay named columns out at their Freddie Mac positions; fill the rest blank."""
    positional = pd.DataFrame("", index=frame.index, columns=range(width))
    for position, name in columns.items():
        positional[position] = frame[name].astype(str)
    return positional


def write_vintage_files(out_dir: Path, loans: pd.DataFrame, performance: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    origination_out = loans.assign(
        credit_score=loans["credit_score_reported"],
        first_payment_month=loans["first_payment_month"].map(lambda p: p.strftime("%Y%m")),
        orig_ltv=loans["orig_ltv"].astype(int),
        orig_dti=loans["orig_dti"].astype(int),
        orig_upb=loans["orig_upb"].astype(int),
    )
    vintage_of_loan = loans.set_index("loan_id")["vintage_year"]
    performance = performance.assign(vintage_year=performance["loan_id"].map(vintage_of_loan))

    for year, orig_year in origination_out.groupby("vintage_year"):
        orig_positional = to_positional_frame(orig_year, ORIGINATION_COLUMNS, ORIGINATION_WIDTH)
        orig_positional.to_csv(out_dir / f"sample_orig_{year}.txt", sep="|", header=False, index=False)

        perf_year = performance[performance["vintage_year"] == year]
        perf_positional = to_positional_frame(perf_year, PERFORMANCE_COLUMNS, PERFORMANCE_WIDTH)
        perf_positional.to_csv(out_dir / f"sample_svcg_{year}.txt", sep="|", header=False, index=False)


def write_bureau_snapshot(out_dir: Path, rng: np.random.Generator, loans: pd.DataFrame, performance: pd.DataFrame) -> None:
    """A simulated credit-bureau refresh taken at DATA_END.

    Freddie Mac has no such table. It exists here to demonstrate mistake 7:
    a score pulled today reflects every default that has already happened.
    It plays the role of Lending Club's last_fico_range_high/low.
    """
    worst_status = (
        performance.assign(dq=pd.to_numeric(performance["dq_status_raw"], errors="coerce").fillna(6))
        .groupby("loan_id")["dq"]
        .max()
    )
    ever_seriously_late = loans["loan_id"].map(worst_status).fillna(0) >= 3
    drift = rng.normal(5, 20, len(loans))
    latest = np.where(ever_seriously_late, loans["credit_score"] - 130 + drift, loans["credit_score"] + drift)
    snapshot = pd.DataFrame(
        {
            "loan_id": loans["loan_id"],
            "latest_credit_score": np.clip(latest, 300, 850).round().astype(int),
            "as_of_month": DATA_END.strftime("%Y%m"),
        }
    )
    snapshot.to_csv(out_dir / "bureau_snapshot_latest.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--loans", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    loans = simulate_loan_attributes(rng, args.loans)

    performance_rows: list[dict] = []
    for _, loan in loans.iterrows():
        performance_rows.extend(simulate_performance(rng, loan))
    performance = pd.DataFrame(performance_rows)

    write_vintage_files(args.out, loans, performance)
    write_bureau_snapshot(args.out, rng, loans, performance)
    print(f"wrote {len(loans):,} loans and {len(performance):,} loan-months to {args.out}")


if __name__ == "__main__":
    main()
