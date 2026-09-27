# Part 2: tensors, time-ordered batching, and why leakage hides in `.shift()`

Companion code for
[Financial machine learning from zero, part 2](https://singhprashant.in/writing/financial-ml-02-tensors-batching-leakage).

| File | What it does |
|---|---|
| `freddie_layout.py` | Column positions and sentinel values for the Freddie Mac sample files |
| `make_synthetic_freddie.py` | Writes a synthetic panel in the same layout, so everything runs without the real data |
| `pipeline.py` | The clean pipeline: load, sort, complete the calendar, label, features, split, preprocess, tensors |
| `datasets.py` | `LoanMonthDataset` (flat rows) and `LoanHistoryDataset` (last N months per decision) |
| `model.py` | A plain logistic regression used as a measuring instrument |
| `leakage_checks.py` | Sorting, calendar, split-gap and point-in-time assertions, the future-perturbation check, and single-feature scores |
| `run_clean.py` | Runs the clean pipeline and prints every intermediate |
| `break_it.py` | Breaks the pipeline one line at a time and measures each break |

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; use `source .venv/bin/activate` elsewhere
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Run on synthetic data

```bash
python make_synthetic_freddie.py --out data/synthetic --loans 12000
python run_clean.py --data data/synthetic
python break_it.py --data data/synthetic
```

The synthetic panel has a known data-generating process, so it shows the
*direction and rough size* of each leak clearly. Its numbers say nothing
about real mortgage risk.

## Run on the real Freddie Mac data

1. Register (free) for the Single-Family Loan-Level Dataset on Freddie
   Mac's website and download the **sample** files for vintages 2005 to
   2010. Freddie Mac's terms do not allow redistribution, which is why
   they are not in this repository.
2. Put the `sample_orig_YYYY.txt` and `sample_svcg_YYYY.txt` files in
   `data/freddie/`.
3. Check the column positions in `freddie_layout.py` against the user
   guide for your release.
4. `python run_clean.py --data data/freddie`

`break_it.py` also expects `bureau_snapshot_latest.csv`, which only the
synthetic generator produces. Freddie Mac has no refreshed credit score.
For the real-world version of that mistake, see Lending Club's
`last_fico_range_high` and `last_fico_range_low` columns, discussed in
the post.
