"""Checks that catch leakage before a reviewer does.

Later posts in the series import this module. Each check is cheap enough
to run on every pipeline change.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from pipeline import LABEL_HORIZON_MONTHS, month_number


def assert_sorted_panel(panel: pd.DataFrame) -> None:
    assert panel["loan_id"].is_monotonic_increasing, "panel is not sorted by loan_id"
    same_loan = panel["loan_id"].eq(panel["loan_id"].shift())
    months = month_number(panel["period"])
    later_month = months > months.shift()
    assert (~same_loan | later_month).all(), "periods are not strictly increasing within a loan"


def assert_no_calendar_gaps(panel: pd.DataFrame) -> None:
    same_loan = panel["loan_id"].eq(panel["loan_id"].shift())
    months = month_number(panel["period"])
    step = months - months.shift()
    n_gaps = int((same_loan & (step != 1)).sum())
    assert n_gaps == 0, f"{n_gaps} rows follow a missing month; shift(1) would skip back further than one month"


def assert_split_gaps(splits: dict[str, pd.DataFrame], horizon: int = LABEL_HORIZON_MONTHS) -> None:
    """Each split must start more than one label horizon after the previous one ends."""
    ordered = sorted(splits.items(), key=lambda item: item[1]["period"].min())
    for (earlier_name, earlier), (later_name, later) in zip(ordered, ordered[1:]):
        gap = later["period"].min().ordinal - earlier["period"].max().ordinal
        assert gap > horizon, (
            f"{earlier_name} ends {gap} months before {later_name} starts; "
            f"its labels reach {horizon} months ahead into {later_name}'s period"
        )


def assert_joined_table_is_point_in_time(rows: pd.DataFrame, as_of: pd.Series) -> None:
    """A table joined from outside the panel must be dated no later than each row's decision month."""
    too_late = month_number(as_of) > month_number(rows["period"])
    assert not too_late.any(), (
        f"{int(too_late.sum()):,} rows use a value dated after their decision month"
    )


def perturb_future(panel: pd.DataFrame, cutoff: pd.Period, seed: int = 0) -> pd.DataFrame:
    """Replace everything reported after `cutoff` with noise."""
    rng = np.random.default_rng(seed)
    future = panel["period"] > cutoff
    n = int(future.sum())
    perturbed = panel.copy()
    perturbed.loc[future, "dq_status"] = rng.integers(0, 7, n).astype(float)
    perturbed.loc[future, "current_upb"] = perturbed.loc[future, "current_upb"] * rng.uniform(0.5, 1.5, n)
    perturbed.loc[future, "zero_balance_code"] = rng.choice(["", "01", "09"], n)
    perturbed.loc[future, "exits_in_default"] = perturbed.loc[future, "zero_balance_code"] == "09"
    return perturbed


def features_that_see_the_future(
    build_features: Callable[[pd.DataFrame], pd.DataFrame],
    panel: pd.DataFrame,
    feature_columns: list[str],
    cutoff: pd.Period,
) -> dict[str, float]:
    """Rebuild features after scrambling the future; anything that moves on or before the cutoff leaked.

    Returns, for each leaking column, the share of pre-cutoff rows whose
    value changed. An empty dict means every feature ignored the future.
    """
    original = build_features(panel)
    rebuilt = build_features(perturb_future(panel, cutoff))
    past = panel["period"] <= cutoff

    leaks = {}
    for column in feature_columns:
        before = original.loc[past, column]
        after = rebuilt.loc[past, column]
        changed = ~((before == after) | (before.isna() & after.isna()))
        if changed.any():
            leaks[column] = float(changed.mean())
    return leaks


def single_feature_scores(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_columns: list[str],
    fit_and_score: Callable[[pd.DataFrame, pd.DataFrame, list[str]], dict[str, float]],
) -> pd.DataFrame:
    """Validation PR-AUC of a model given one feature at a time."""
    rows = [
        {"feature": column, **fit_and_score(train, valid, [column])}
        for column in feature_columns
    ]
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False, ignore_index=True)
