"""Break the clean pipeline one line at a time and measure what each break does.

Every experiment fits the same logistic regression, to convergence, on
the same training window and scores it on the same validation window.
Only the feature or label construction changes.

Usage:
    python break_it.py --data data/synthetic
"""

import argparse
from pathlib import Path

import pandas as pd

from leakage_checks import (
    assert_joined_table_is_point_in_time,
    assert_no_calendar_gaps,
    assert_sorted_panel,
    assert_split_gaps,
    features_that_see_the_future,
    single_feature_scores,
)
from model import fit_to_convergence, predict_proba, score
from pipeline import (
    FEATURE_COLUMNS,
    LABEL,
    assemble_rows,
    build_features,
    build_label,
    complete_monthly_calendar,
    fit_preprocessing,
    load_origination,
    load_performance,
    sort_panel,
    split_by_time,
    to_tensors,
)
from run_clean import SPLIT_WINDOWS


def fit_and_score(train: pd.DataFrame, valid: pd.DataFrame, columns: list[str], prep_source: pd.DataFrame | None = None) -> dict[str, float]:
    prep = fit_preprocessing(train if prep_source is None else prep_source, columns)
    x_train, y_train, _ = to_tensors(train, prep)
    x_valid, y_valid, _ = to_tensors(valid, prep)
    model = fit_to_convergence(x_train, y_train)
    return score(y_valid, predict_proba(model, x_valid))


def splits_for(labelled: pd.DataFrame, features: pd.DataFrame, columns: list[str]) -> dict[str, pd.DataFrame]:
    rows = assemble_rows(labelled, features, columns)
    splits = split_by_time(rows, SPLIT_WINDOWS)
    assert_split_gaps(splits)
    return splits


# --- The broken feature builders ---------------------------------------------


def features_shift_wrong_way(panel: pd.DataFrame, origination: pd.DataFrame) -> pd.DataFrame:
    """Mistake 1: shift(-1) where shift(1) was meant."""
    features = build_features(panel, origination)
    features["dq_status_prev"] = panel.groupby("loan_id", sort=False)["dq_status"].shift(-1)
    return features


def features_without_groupby(panel: pd.DataFrame, origination: pd.DataFrame) -> pd.DataFrame:
    """Mistake 2: history features computed down the whole frame, across loan boundaries."""
    features = build_features(panel, origination)
    features["dq_status_prev"] = panel["dq_status"].shift(1)
    is_late = (panel["dq_status"] > 0).astype(float).where(panel["dq_status"].notna())
    features["months_late_last_12"] = is_late.rolling(12, min_periods=1).sum()
    features["worst_status_last_12"] = panel["dq_status"].rolling(12, min_periods=1).max()
    return features


def features_on_unsorted_frame(panel: pd.DataFrame, origination: pd.DataFrame) -> pd.DataFrame:
    """Mistake 3: the right groupby calls, on rows that are not in month order."""
    shuffled = panel.sample(frac=1.0, random_state=0)
    return build_features(shuffled, origination).reindex(panel.index)


def features_with_lagged_label(panel: pd.DataFrame, origination: pd.DataFrame, data_end: pd.Period) -> pd.DataFrame:
    """Mistake 5: last month's label as a feature. The shift is positive; the column looks forward."""
    features = build_features(panel, origination)
    labelled = build_label(panel, data_end)
    features["label_last_month"] = labelled.groupby("loan_id", sort=False)[LABEL].shift(1)
    return features


def features_with_post_outcome_column(panel: pd.DataFrame, origination: pd.DataFrame) -> pd.DataFrame:
    """Mistake 6: a loan-level field that only exists because of how the loan ended."""
    features = build_features(panel, origination)
    ever_exits_in_default = panel.groupby("loan_id", sort=False)["exits_in_default"].transform("max")
    features["loan_exited_in_default"] = ever_exits_in_default.astype(float)
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/synthetic"))
    args = parser.parse_args()

    origination = load_origination(args.data)
    sorted_panel = sort_panel(load_performance(args.data))
    assert_sorted_panel(sorted_panel)
    panel = complete_monthly_calendar(sorted_panel)
    assert_no_calendar_gaps(panel)
    data_end = panel["period"].max()
    cutoff = data_end - 48
    labelled = build_label(panel, data_end)

    results = []

    def record(name: str, scores: dict[str, float], caught_by: str) -> None:
        results.append({"experiment": name, **scores, "caught_by": caught_by})
        print(f"{name:<40} pr_auc {scores['pr_auc']:.3f}  roc_auc {scores['roc_auc']:.3f}  caught by: {caught_by}")

    def future_check(builder, columns: list[str]) -> str:
        leaks = features_that_see_the_future(builder, panel, columns, cutoff)
        if not leaks:
            return "not caught by the future check"
        detail = ", ".join(f"{c} ({share:.0%} of rows)" for c, share in leaks.items())
        return f"future check: {detail}"

    # Clean baseline.
    clean_features = build_features(panel, origination)
    clean = splits_for(labelled, clean_features, FEATURE_COLUMNS)
    record("clean", fit_and_score(clean["train"], clean["valid"], FEATURE_COLUMNS),
           future_check(lambda p: build_features(p, origination), FEATURE_COLUMNS))

    # 1. Wrong direction.
    broken = splits_for(labelled, features_shift_wrong_way(panel, origination), FEATURE_COLUMNS)
    record("1 shift(-1) on a feature", fit_and_score(broken["train"], broken["valid"], FEATURE_COLUMNS),
           future_check(lambda p: features_shift_wrong_way(p, origination), FEATURE_COLUMNS))

    # 2. No groupby.
    broken = splits_for(labelled, features_without_groupby(panel, origination), FEATURE_COLUMNS)
    record("2 shift/rolling without groupby", fit_and_score(broken["train"], broken["valid"], FEATURE_COLUMNS),
           future_check(lambda p: features_without_groupby(p, origination), FEATURE_COLUMNS))

    # 3. Unsorted frame.
    broken = splits_for(labelled, features_on_unsorted_frame(panel, origination), FEATURE_COLUMNS)
    record("3 groupby shift on unsorted rows", fit_and_score(broken["train"], broken["valid"], FEATURE_COLUMNS),
           future_check(lambda p: features_on_unsorted_frame(p, origination), FEATURE_COLUMNS))

    # 4. Missing months, shifted by rows. Drop 10% of reports, skip the calendar step.
    gappy = sorted_panel.sample(frac=0.9, random_state=1).sort_values(["loan_id", "period"], ignore_index=True)
    gappy["is_reported"] = True
    gappy_labelled = build_label(gappy, data_end)
    gappy_features = build_features(gappy, origination)
    gappy_splits = splits_for(gappy_labelled, gappy_features, FEATURE_COLUMNS)

    true_labels = labelled.set_index(["loan_id", "period"])[LABEL]
    gappy_rows = pd.concat(gappy_splits.values())
    true_for_gappy = true_labels.reindex(pd.MultiIndex.from_frame(gappy_rows[["loan_id", "period"]])).to_numpy()
    wrong = gappy_rows[LABEL].to_numpy() != true_for_gappy
    wrong_positive = (gappy_rows[LABEL].to_numpy() == 1) & (true_for_gappy == 0)
    print(
        f"   gaps: {wrong.mean():.2%} of labels wrong; "
        f"{wrong_positive.sum() / max(1, (gappy_rows[LABEL] == 1).sum()):.1%} of positive labels are not real defaults "
        f"within 12 months; default rate {gappy_rows[LABEL].mean():.2%} vs {true_for_gappy.mean():.2%}"
    )
    valid_true = gappy_splits["valid"].assign(
        **{LABEL: true_labels.reindex(pd.MultiIndex.from_frame(gappy_splits["valid"][["loan_id", "period"]])).to_numpy()}
    )
    record("4 row shifts over missing months", fit_and_score(gappy_splits["train"], valid_true, FEATURE_COLUMNS),
           "assert_no_calendar_gaps")

    # 5. Lagged label.
    columns = FEATURE_COLUMNS + ["label_last_month"]
    broken_features = features_with_lagged_label(panel, origination, data_end)
    broken = splits_for(labelled, broken_features, columns)
    record("5 last month's label as a feature", fit_and_score(broken["train"], broken["valid"], columns),
           future_check(lambda p: features_with_lagged_label(p, origination, data_end), columns))
    lagged_label_splits = broken

    # 6. Post-outcome column.
    columns = FEATURE_COLUMNS + ["loan_exited_in_default"]
    broken = splits_for(labelled, features_with_post_outcome_column(panel, origination), columns)
    record("6 post-outcome loan field", fit_and_score(broken["train"], broken["valid"], columns),
           future_check(lambda p: features_with_post_outcome_column(p, origination), columns))

    # 7. Today's bureau score instead of the score at origination.
    snapshot = pd.read_csv(args.data / "bureau_snapshot_latest.csv", dtype={"as_of_month": str})
    snapshot_by_loan = snapshot.set_index("loan_id")
    snapshot_features = clean_features.copy()
    snapshot_features["credit_score"] = panel["loan_id"].map(snapshot_by_loan["latest_credit_score"])
    broken = splits_for(labelled, snapshot_features, FEATURE_COLUMNS)
    as_of = pd.to_datetime(broken["train"]["loan_id"].map(snapshot_by_loan["as_of_month"]), format="%Y%m").dt.to_period("M")
    try:
        assert_joined_table_is_point_in_time(broken["train"], as_of)
        caught_by = "not caught"
    except AssertionError as error:
        caught_by = f"assert_joined_table_is_point_in_time ({error})"
    record("7 today's credit score", fit_and_score(broken["train"], broken["valid"], FEATURE_COLUMNS), caught_by)

    # 8. Scaler statistics from every split, not just train.
    all_rows = pd.concat(clean.values())
    record("8 scaler fitted on all splits",
           fit_and_score(clean["train"], clean["valid"], FEATURE_COLUMNS, prep_source=all_rows),
           "structural: fit_preprocessing only ever receives train")

    print("\n| Experiment | Validation PR-AUC | Validation ROC-AUC | Caught by |")
    print("|---|---|---|---|")
    for r in results:
        print(f"| {r['experiment']} | {r['pr_auc']:.3f} | {r['roc_auc']:.3f} | {r['caught_by']} |")
    print(f"\nvalidation base rate: {results[0]['base_rate']:.2%}")

    print("\nSingle-feature check, clean features:")
    print(single_feature_scores(clean["train"], clean["valid"], FEATURE_COLUMNS, fit_and_score).to_string())
    print("\nSingle-feature check, with last month's label:")
    print(single_feature_scores(
        lagged_label_splits["train"], lagged_label_splits["valid"], ["label_last_month", "dq_status_now"], fit_and_score
    ).to_string())


if __name__ == "__main__":
    main()
