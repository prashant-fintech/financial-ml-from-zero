"""The metrics from part 1, written out so each one can be read in a few lines.

Every function takes plain numpy arrays: `y` holds 0/1 outcomes (1 = the
event: default, fraud, a confirmed case), `p` holds predicted
probabilities or scores, and `flagged` holds 0/1 decisions. Each
docstring ends with the direction that is better, as in part 1's tables.

test_metrics.py checks these against scikit-learn wherever it has an
equivalent.
"""

import numpy as np

# --- Counting outcomes at a decision threshold --------------------------------


def confusion_counts(y: np.ndarray, flagged: np.ndarray) -> dict[str, int]:
    """True/false positives and negatives for 0/1 decisions."""
    return {
        "tp": int(np.sum((flagged == 1) & (y == 1))),
        "fp": int(np.sum((flagged == 1) & (y == 0))),
        "tn": int(np.sum((flagged == 0) & (y == 0))),
        "fn": int(np.sum((flagged == 0) & (y == 1))),
    }


def accuracy(y: np.ndarray, flagged: np.ndarray) -> float:
    """Share of decisions that are correct. Better: depends on the base rate."""
    c = confusion_counts(y, flagged)
    return (c["tp"] + c["tn"]) / len(y)


def precision(y: np.ndarray, flagged: np.ndarray) -> float:
    """Of the cases flagged, the share that were real. Better: higher."""
    c = confusion_counts(y, flagged)
    flagged_total = c["tp"] + c["fp"]
    return c["tp"] / flagged_total if flagged_total else 0.0


def recall(y: np.ndarray, flagged: np.ndarray) -> float:
    """Of the real cases, the share that were flagged. Better: higher."""
    c = confusion_counts(y, flagged)
    positives = c["tp"] + c["fn"]
    return c["tp"] / positives if positives else 0.0


def f1(y: np.ndarray, flagged: np.ndarray) -> float:
    """Harmonic mean of precision and recall. Better: higher."""
    p, r = precision(y, flagged), recall(y, flagged)
    return 2 * p * r / (p + r) if (p + r) else 0.0


# --- Ranking quality -----------------------------------------------------------


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Probability that a random positive scores above a random negative. Better: higher.

    Computed directly from that definition via ranks (the Mann-Whitney U
    statistic), with ties counted as half.
    """
    ranks = _average_ranks(p)
    n_pos = int(np.sum(y == 1))
    n_neg = len(y) - n_pos
    rank_sum_pos = ranks[y == 1].sum()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def gini(y: np.ndarray, p: np.ndarray) -> float:
    """The credit-risk name for ranking power: 2 x ROC-AUC - 1. Better: higher."""
    return 2 * roc_auc(y, p) - 1


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    """Largest gap between the score distributions of bads and goods. Better: higher."""
    thresholds = np.unique(p)
    bads, goods = p[y == 1], p[y == 0]
    cdf_bad = np.searchsorted(np.sort(bads), thresholds, side="right") / len(bads)
    cdf_good = np.searchsorted(np.sort(goods), thresholds, side="right") / len(goods)
    return float(np.max(np.abs(cdf_good - cdf_bad)))


def pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Area under the precision-recall curve, as average precision. Better: higher.

    Walk down the list from the highest score; at each real case, note the
    precision so far. The average of those precisions is the area. A model
    with no skill scores about the base rate.
    """
    order = np.argsort(-p, kind="stable")
    y_sorted = y[order]
    p_sorted = p[order]
    # Treat tied scores as one step, like a threshold would.
    last_of_tie = np.r_[p_sorted[1:] != p_sorted[:-1], True]
    cum_tp = np.cumsum(y_sorted)[last_of_tie]
    cum_flagged = (np.arange(len(y)) + 1)[last_of_tie]
    precision_at_step = cum_tp / cum_flagged
    recall_at_step = cum_tp / y.sum()
    recall_gain = np.diff(np.r_[0.0, recall_at_step])
    return float(np.sum(recall_gain * precision_at_step))


def precision_at_k(y: np.ndarray, p: np.ndarray, k: int) -> float:
    """Of the k highest-scored cases, the share that were real. Better: higher."""
    top_k = np.argsort(-p, kind="stable")[:k]
    return float(y[top_k].mean())


# --- Probability quality -------------------------------------------------------


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    """Average penalty for the probability given to what happened. Better: lower."""
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared gap between predicted probability and the 0/1 outcome. Better: lower."""
    return float(np.mean((p - y) ** 2))


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    """Predicted vs observed rate in equal-count score buckets. Better: closer to equal."""
    order = np.argsort(p, kind="stable")
    rows = []
    for bucket in np.array_split(order, n_bins):
        rows.append(
            {
                "n": len(bucket),
                "mean_predicted": float(p[bucket].mean()),
                "observed_rate": float(y[bucket].mean()),
            }
        )
    return rows


# --- Drift -----------------------------------------------------------------------


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population stability index between a reference and a new score sample. Better: lower.

    Bins are deciles of the reference sample. Empty bins get a small floor
    so the logarithm stays finite.
    """
    edges = np.quantile(expected, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    expected_share = np.histogram(expected, edges)[0] / len(expected)
    actual_share = np.histogram(actual, edges)[0] / len(actual)
    expected_share = np.clip(expected_share, 1e-6, None)
    actual_share = np.clip(actual_share, 1e-6, None)
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


# --- What the business feels ----------------------------------------------------


def approval_rate(pd_scores: np.ndarray, cutoff: float) -> float:
    """Share of applicants approved when everyone below the PD cut-off is accepted."""
    return float(np.mean(pd_scores < cutoff))


def bad_rate_at_cutoff(y: np.ndarray, pd_scores: np.ndarray, cutoff: float) -> float:
    """Default rate among the applicants approved at this cut-off. Better: lower."""
    approved = pd_scores < cutoff
    return float(y[approved].mean()) if approved.any() else 0.0


def expected_credit_loss(pd: np.ndarray, ead: np.ndarray, lgd: np.ndarray) -> float:
    """Sum over the book of PD x exposure at default x loss given default. Better: lower."""
    return float(np.sum(pd * ead * lgd))


def false_positives_per_case_caught(y: np.ndarray, flagged: np.ndarray) -> float:
    """Good cases flagged for each real one caught: blocked good transactions
    per fraud stopped, or wasted alerts per confirmed AML case. Better: lower."""
    c = confusion_counts(y, flagged)
    return c["fp"] / c["tp"] if c["tp"] else float("inf")


# --- Helpers -----------------------------------------------------------------------


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks with ties sharing their average rank."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2 + 1
        start = end + 1
    return ranks
