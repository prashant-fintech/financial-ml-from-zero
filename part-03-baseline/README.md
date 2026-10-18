# Part 3: a probability-of-default baseline you can trust

Companion code for [Financial machine learning from zero, part 3](https://singhprashant.in/writing/financial-ml-03-default-baseline).

This folder imports the unchanged loader, features, labels, preprocessing and
checks from `../part-02-leakage`. Keep both folders together. Everything runs
on a CPU; no account, downloaded mortgage data or GPU is needed.

## Setup

From this folder, create an environment with Python 3.12:

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` on macOS/Linux. Then:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest -v test_baseline
```

`requirements-repro.txt` records the direct dependency versions used for the
published tables (Python 3.12.6, CPU PyTorch). Use it instead of
`requirements.txt` to match that environment more closely. It is not a lock
of transitive dependencies; small floating-point differences across platforms
are possible. The saved `frozen.json` also records runtime versions.

For that pinned run, install the matching CPU wheel explicitly:

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-repro.txt
```

## Run in order

```bash
python ../part-02-leakage/make_synthetic_freddie.py --out data/synthetic --loans 12000 --seed 7
python run_baseline.py develop --data data/synthetic --out artifacts/baseline
python run_baseline.py evaluate --data data/synthetic --out artifacts/baseline
```

If you already generated the seed-7 panel in part 2, skip generation and use
`--data ../part-02-leakage/data/synthetic` in both commands. The example
results use the 12,000-loan panel; small samples need not contain both classes.

The development command discards monthly performance reports after December
2010 *before* building labels or features. It trains on June 2005–December
2007 and selects on January–December 2009, with the same twelve-month label
horizon and gaps as part 2. Preprocessing and class weights use training rows
only. Two logistic regressions are fitted with full-batch L-BFGS and a
gradient convergence check. There is no random split or stochastic training.

The selection rule is fixed in code: choose the lowest validation log loss
between ordinary logistic regression and weighted logistic regression with
the analytic log-odds correction. Raw weighted scores are a diagnostic, not
a candidate PD estimate. Constant and delinquency-only baselines are always
reported. A score threshold, chosen on validation, reviews at most 10% of
rows there; ties stay together. This is a teaching review budget, not a
credit approval policy or an estimate of the value of an intervention.

`frozen.json` stores the selected coefficients, preprocessing, benchmarks,
thresholds, package versions and hashes of the source and input files. The
evaluation command loads that file, scores the 2011 test period using
outcomes through December 2012, and writes the report. It fits nothing.
It creates `test-started.json` exclusively before building test examples and
refuses a second evaluation. Development also refuses to overwrite a frozen
run. Inspect saved files to revisit results.

These guards prevent accidental reuse, not deliberate deletion or creating
a new output folder. If evaluation is interrupted, the marker remains: inspect
the failure before deciding whether a fresh run is justified. Do not use a new
folder to tune against the same exposed test period. Part 2 already disclosed
the test base rate; this is a disciplined tutorial evaluation, not a claim
of a fully blind research benchmark.

## Files and outputs

| File | Purpose |
|---|---|
| `data.py` | Reuses part 2 and imposes the development observation cutoff |
| `risk_model.py` | Logistic fitting, corrected probabilities, metrics and tied-score policy |
| `run_baseline.py` | Separate develop/freeze and evaluate commands |
| `reporting.py` | CSV reports and calibration figures |
| `test_baseline.py` | Checks odds correction, ties, cutoff isolation and the evaluation guard |
| `example-results/` | Aggregate output from the run quoted in the post |

Under `artifacts/baseline/` you get validation and test metrics, fixed-bin
calibration tables and PNGs; the frozen model; the test review confusion
counts; and `test-predictions.csv` containing loan ID, decision month,
outcome, selected probability and review flag. Generated data and artifacts
are ignored by Git. Only small aggregate example results are committed.

The column called `average_precision` is scikit-learn's non-interpolated AP,
the implementation called PR-AUC in part 2. Brier and log loss measure overall
probability quality, not calibration alone. Calibration bins are fixed before
evaluation; their CSV includes row and event counts. No calibrator is fitted
on validation or test. The weighted correction removes an idealised training
odds shift; it does not promise calibration under distribution drift.

## Limits

All results are synthetic and measure **loan-months**, not independent
borrowers or prevented defaults. Some loans appear at multiple decision
months and across time splits. This evaluates later monitoring of a portfolio,
not generalisation to entirely unseen borrowers. We do not attach row-wise
confidence intervals to correlated observations.

The wrapper deliberately rejects missing monthly reports, unknown status
and unexplained early reporting stops. Part 2's global label-maturity rule
alone cannot establish a known negative after such censoring. The synthetic
generator satisfies these checks. Real-data use requires reviewing the file
layout, closure codes, missing reports and reporting delays; this post does
not validate that extension.
