"""Check metrics.py against scikit-learn and scipy.

Run with `python test_metrics.py` (or `pytest test_metrics.py`).
"""

import numpy as np
from scipy.stats import ks_2samp
from sklearn import metrics as sk

import metrics as m

rng = np.random.default_rng(0)


def sample(n: int = 5000, base_rate: float = 0.05, ties: bool = False) -> tuple[np.ndarray, np.ndarray]:
    y = (rng.random(n) < base_rate).astype(int)
    p = 1 / (1 + np.exp(-(rng.normal(0, 1, n) + 1.2 * y - 3)))
    if ties:
        p = np.round(p, 2)  # a scorecard with coarse score bands
    return y, p


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def test_threshold_metrics() -> None:
    y, p = sample()
    flagged = (p > 0.1).astype(int)
    assert close(m.accuracy(y, flagged), sk.accuracy_score(y, flagged))
    assert close(m.precision(y, flagged), sk.precision_score(y, flagged))
    assert close(m.recall(y, flagged), sk.recall_score(y, flagged))
    assert close(m.f1(y, flagged), sk.f1_score(y, flagged))


def test_ranking_metrics() -> None:
    for ties in (False, True):
        y, p = sample(ties=ties)
        assert close(m.roc_auc(y, p), sk.roc_auc_score(y, p))
        assert close(m.gini(y, p), 2 * sk.roc_auc_score(y, p) - 1)
        assert close(m.pr_auc(y, p), sk.average_precision_score(y, p))
        assert close(m.ks_statistic(y, p), ks_2samp(p[y == 1], p[y == 0]).statistic)


def test_probability_metrics() -> None:
    y, p = sample()
    assert close(m.log_loss(y, p), sk.log_loss(y, p), tol=1e-12)
    assert close(m.brier_score(y, p), sk.brier_score_loss(y, p))


def test_psi_is_zero_for_identical_samples() -> None:
    scores = rng.normal(0, 1, 10000)
    assert close(m.psi(scores, scores), 0.0)


def test_precision_at_k() -> None:
    y = np.array([1, 0, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    assert close(m.precision_at_k(y, p, 3), 2 / 3)


if __name__ == "__main__":
    tests = [value for name, value in dict(globals()).items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
