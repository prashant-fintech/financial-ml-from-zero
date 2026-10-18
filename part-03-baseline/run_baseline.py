"""Develop and freeze first; open the 2011 test period in a separate command."""

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch

from data import (
    FEATURE_COLUMNS, LABEL, PART_2, WINDOWS, apply_preprocessing, development_splits,
    fit_preprocessing, prep_from_dict, prep_to_dict, prepare_rows, split_by_time, to_tensors,
)
from reporting import save_report, write_json
from risk_model import capacity_threshold, export_model, fit_logistic, predict

HERE = Path(__file__).resolve().parent


def fingerprint(paths):
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: (p.parent.name, p.name)):
        digest.update(f"{path.parent.name}/{path.name}\n".encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def code_fingerprint():
    return fingerprint([*HERE.glob("*.py"), *PART_2.glob("*.py")])


def data_fingerprint(directory):
    return fingerprint([*directory.glob("sample_orig_*.txt"), *directory.glob("sample_svcg_*.txt")])


def benchmark_predictions(frame, frozen):
    return {
        "constant": np.full(len(frame), frozen["train_rate"]),
        "delinquency_only": frame["dq_status_now"].map(
            {int(k): v for k, v in frozen["delinquency_rates"].items()}
        ).fillna(frozen["train_rate"]).to_numpy(dtype=float),
    }


def develop(data_dir, output):
    if (output / "frozen.json").exists() or (output / "test-started.json").exists():
        raise FileExistsError("This run is frozen. Use its existing report; do not retune on the test result.")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    splits = development_splits(data_dir)
    train, valid = splits["train"], splits["valid"]
    prep = fit_preprocessing(train, FEATURE_COLUMNS)
    x_train, y_train, _ = to_tensors(train, prep)
    x_valid, y_valid, _ = to_tensors(valid, prep)
    train_rate = float(y_train.double().mean())
    positive_weight = float((len(y_train) - y_train.sum()) / y_train.sum())
    unweighted, plain_fit = fit_logistic(x_train, y_train)
    weighted, weighted_fit = fit_logistic(x_train, y_train, positive_weight)
    specifications = {
        "unweighted": export_model(unweighted),
        "weighted_raw": export_model(weighted),
        "weighted_corrected": export_model(weighted, math.log(positive_weight)),
    }
    # Twenty pseudo-observations at the training base rate smooth the three groups.
    groups = train.groupby("dq_status_now")[LABEL].agg(["sum", "count"])
    rates = (groups["sum"] + 20 * train_rate) / (groups["count"] + 20)
    frozen = {
        "schema_version": 1, "train_rate": train_rate, "positive_weight": positive_weight,
        "delinquency_rates": {str(int(k)): float(v) for k, v in rates.items()},
        "preprocessing": prep_to_dict(prep), "windows": WINDOWS,
        "development_as_of": "2010-12", "test_as_of": "2012-12",
        "selection_rule": "Lowest validation log loss among unweighted and weighted_corrected",
        "review_fraction_on_validation": .10,
        "fit_diagnostics": {"unweighted": plain_fit, "weighted": weighted_fit},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "torch": torch.__version__, "sklearn": sklearn.__version__},
        "development_splits": {
            name: {"rows": len(frame), "loans": int(frame["loan_id"].nunique()),
                   "events": int(frame[LABEL].sum()), "event_rate": float(frame[LABEL].mean())}
            for name, frame in splits.items()
        },
    }
    predictions = benchmark_predictions(valid, frozen)
    predictions.update({name: predict(spec, x_valid.numpy()) for name, spec in specifications.items()})
    metrics = save_report(output, "validation", y_valid.numpy(), predictions)
    candidates = metrics.set_index("model").loc[["unweighted", "weighted_corrected"], "log_loss"]
    selected = str(candidates.idxmin())
    frozen["selected_name"] = selected
    frozen["selected_model"] = specifications[selected]
    frozen["thresholds"] = {
        name: capacity_threshold(predictions[name]) for name in ("constant", "delinquency_only", selected)
    }
    frozen["code_sha256"] = code_fingerprint()
    frozen["data_sha256"] = data_fingerprint(data_dir)
    write_json(output / "frozen.json", frozen)
    print(f"\nFrozen: {selected}; positive weight={positive_weight:.6f}; threshold={frozen['thresholds'][selected]:.6f}")
    print("No test examples were built. Next: evaluate with this same data and output directory.")


def evaluate(data_dir, output):
    frozen_path = output / "frozen.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen["code_sha256"] != code_fingerprint() or frozen["data_sha256"] != data_fingerprint(data_dir):
        raise ValueError("Code or input files changed after freezing. Do not reuse this run.")
    # Reserve the evaluation before touching test labels; even an interrupted run is visible.
    with (output / "test-started.json").open("x", encoding="utf-8") as handle:
        json.dump({"frozen_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest()}, handle)
    rows = prepare_rows(data_dir, frozen["test_as_of"])
    test = split_by_time(rows, {"test": tuple(frozen["windows"]["test"])})["test"]
    if test.empty:
        raise ValueError("The test period is empty")
    x = apply_preprocessing(test, prep_from_dict(frozen["preprocessing"]))
    selected = frozen["selected_name"]
    predictions = benchmark_predictions(test, frozen)
    predictions[selected] = predict(frozen["selected_model"], x)
    save_report(output, "test", test[LABEL].to_numpy(), predictions, frozen["thresholds"])
    trace = test[["loan_id", "period", LABEL]].copy()
    trace["probability"] = predictions[selected]
    trace["review"] = trace["probability"] >= frozen["thresholds"][selected]
    trace.to_csv(output / "test-predictions.csv", index=False)
    print("\nTest evaluation complete. Inspect the saved files; this run cannot be evaluated again.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["develop", "evaluate"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/baseline"))
    args = parser.parse_args()
    try:
        {"develop": develop, "evaluate": evaluate}[args.phase](args.data, args.out)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
