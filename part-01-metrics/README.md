# Part 1: the vocabulary, the traps, and the metrics that matter

Companion code for
[Financial machine learning from zero, part 1](https://singhprashant.in/writing/financial-ml-01-vocabulary-traps-metrics).
The post has no code by design; this folder makes its metric tables and
its traps runnable.

| File | What it does |
|---|---|
| `metrics.py` | Every classification, calibration, drift, and business metric from the post's tables, each written out in a few lines of numpy |
| `test_metrics.py` | Checks `metrics.py` against scikit-learn and scipy wherever they have an equivalent |
| `traps.py` | The post's traps as seven small experiments with printed numbers |

The market-track metrics (Sharpe, drawdown, IC) are left for the
market posts later in the series.

## Setup and run

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; use `source .venv/bin/activate` elsewhere
pip install -r requirements.txt
python test_metrics.py
python traps.py                   # or: python traps.py 3 5
```

Everything runs in about a minute on a laptop CPU. No PyTorch yet; that
starts in part 2.

## The seven traps

1. **Accuracy lies when defaults are rare.** Flagging nobody scores 99%
   accuracy at a 1% base rate, and a PR-AUC equal to the base rate.
2. **ROC-AUC stays flat while the event gets rarer.** The same
   separation gives the same ROC-AUC at 10%, 1%, and 0.1%; PR-AUC falls
   with the base rate.
3. **Ranking is not calibration.** Dividing every probability by three
   leaves Gini and KS unchanged and makes Brier and log loss worse.
4. **Every cut-off is a trade-off.** Approval rate against bad rate for
   credit; recall against good transactions blocked per fraud caught.
5. **A random split looks better than the future.** Same model, same
   data, a random split vs an out-of-time split, repeated over ten
   simulated economies.
6. **You only see defaults on loans you approved.** A model trained on
   approved loans is calibrated on them and badly understates risk on
   the applicants the old policy declined.
7. **PSI is a drift alarm, not a quality score.** PSI for growing shifts
   in the scored population, against the common 0.10 and 0.25 bands.

All data is simulated with a known answer, so each effect is visible.
The numbers illustrate the traps; they are not estimates for any real
portfolio.
