"""The clean pipeline end to end, printing each intermediate.

Usage:
    python run_clean.py --data data/synthetic
    python run_clean.py --data data/freddie   # the real sample files
"""

import argparse
from pathlib import Path

import torch

from datasets import LoanHistoryDataset, LoanMonthDataset
from leakage_checks import (
    assert_no_calendar_gaps,
    assert_sorted_panel,
    assert_split_gaps,
    features_that_see_the_future,
)
from model import predict_proba, score, train_logistic
from pipeline import (
    FEATURE_COLUMNS,
    LABEL,
    apply_preprocessing,
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

SPLIT_WINDOWS = {
    "train": ("2005-06", "2007-12"),
    "valid": ("2009-01", "2009-12"),
    "test": ("2011-01", "2011-12"),
}


def show(name: str, frame) -> None:
    print(f"\n== {name}: {frame.shape[0]:,} rows x {frame.shape[1]} columns")
    print(frame.head(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/synthetic"))
    args = parser.parse_args()

    origination = load_origination(args.data)
    show("origination", origination)

    performance = load_performance(args.data)
    show("performance", performance)

    sorted_panel = sort_panel(performance)
    assert_sorted_panel(sorted_panel)

    panel = complete_monthly_calendar(sorted_panel)
    assert_no_calendar_gaps(panel)
    print(f"\nunreported months filled in: {int((~panel['is_reported']).sum()):,}")

    data_end = panel["period"].max()
    labelled = build_label(panel, data_end)
    show("labelled", labelled[["loan_id", "period", "dq_status", LABEL, "label_is_mature"]])

    features = build_features(panel, origination)
    show("features", features)

    leaks = features_that_see_the_future(
        lambda p: build_features(p, origination), panel, FEATURE_COLUMNS, cutoff=data_end - 48
    )
    print(f"\nfeatures that see the future: {leaks or 'none'}")

    rows = assemble_rows(labelled, features, FEATURE_COLUMNS)
    splits = split_by_time(rows, SPLIT_WINDOWS)
    assert_split_gaps(splits)
    for name, split in splits.items():
        print(
            f"{name:>5}: {split['period'].min()} to {split['period'].max()}, "
            f"{len(split):,} rows, {split['loan_id'].nunique():,} loans, "
            f"default rate {split[LABEL].mean():.2%}"
        )

    prep = fit_preprocessing(splits["train"], FEATURE_COLUMNS)
    x_train, y_train, keys_train = to_tensors(splits["train"], prep)
    x_valid, y_valid, _ = to_tensors(splits["valid"], prep)
    print(f"\nx_train {tuple(x_train.shape)} {x_train.dtype}, y_train {tuple(y_train.shape)}")

    train_set = LoanMonthDataset(x_train, y_train, keys_train)
    print(f"train_set[0] traces back to {train_set.key(0)}")

    model = train_logistic(train_set)
    print(f"\nvalidation: {score(y_valid, predict_proba(model, x_valid))}")

    # The sequence view of the same rows: the last six months per decision.
    panel_x = apply_preprocessing(features, prep)
    history = LoanHistoryDataset(
        panel_x,
        panel["loan_id"],
        decision_rows=splits["train"].index.to_numpy(),
        labels=splits["train"][LABEL].to_numpy(),
        window=6,
    )
    sequence, mask, label = history[0]
    print(f"\nhistory[0]: sequence {tuple(sequence.shape)}, real months {int(mask.sum())}, label {label.item()}")
    batch = torch.utils.data.DataLoader(history, batch_size=256)
    sequences, masks, labels = next(iter(batch))
    print(f"one batch: {tuple(sequences.shape)} = [batch, months, features]")


if __name__ == "__main__":
    main()
