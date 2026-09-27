"""The traps from part 1, each as a small experiment with printed numbers.

Every experiment uses simulated data with a known answer, so the size of
each effect is visible. The numbers illustrate the traps; they are not
estimates for any real portfolio.

Usage:
    python traps.py              # all traps
    python traps.py 3 5          # only traps 3 and 5
"""

import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import metrics as m

SEED = 7


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def table(headers: list[str], rows: list[list[str]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("|" + "---|" * len(headers))
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print()


def title(n: int, text: str) -> None:
    print(f"\n## Trap {n}: {text}\n")


# --- 1 ---------------------------------------------------------------------------


def trap_1_accuracy_lies() -> None:
    title(1, "accuracy lies when defaults are rare")
    rng = np.random.default_rng(SEED)
    y = (rng.random(100_000) < 0.01).astype(int)
    flag_nobody = np.zeros_like(y)
    constant_score = np.full(len(y), 0.01)

    table(
        ["Model", "Accuracy", "Recall", "PR-AUC"],
        [[
            'Predict "no default" for everyone',
            f"{m.accuracy(y, flag_nobody):.1%}",
            f"{m.recall(y, flag_nobody):.0%}",
            f"{m.pr_auc(y, constant_score):.3f}",
        ]],
    )
    print(f"Base rate {y.mean():.2%}. The model catches nothing and is 99% accurate;")
    print("its PR-AUC equals the base rate, which is the no-skill baseline.")


# --- 2 ---------------------------------------------------------------------------


def trap_2_roc_flatters_rare_events() -> None:
    title(2, "ROC-AUC stays flat while the event gets rarer; PR-AUC does not")
    rng = np.random.default_rng(SEED)
    n = 400_000
    rows = []
    for base_rate in (0.10, 0.01, 0.001):
        y = (rng.random(n) < base_rate).astype(int)
        # The same separation between goods and bads at every base rate.
        score = rng.normal(0, 1, n) + 1.5 * y
        rows.append([
            f"{base_rate:.1%}",
            f"{m.roc_auc(y, score):.3f}",
            f"{m.gini(y, score):.3f}",
            f"{m.pr_auc(y, score):.3f}",
        ])
    table(["Base rate", "ROC-AUC", "Gini", "PR-AUC"], rows)
    print("The model is equally good at separating the two groups in every row.")
    print("ROC-AUC reports that. PR-AUC reports what an investigator would see:")
    print("at 0.1%, most of what the model flags is still a false alarm.")


# --- 3 ---------------------------------------------------------------------------


def simulated_book(rng: np.random.Generator, n: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
    """True default probabilities for a retail book, and the outcomes they produce."""
    true_pd = sigmoid(rng.normal(-3.2, 1.1, n))
    y = (rng.random(n) < true_pd).astype(int)
    return true_pd, y


def trap_3_ranking_is_not_calibration() -> None:
    title(3, "a model can rank perfectly well and still be wrong about the probability")
    rng = np.random.default_rng(SEED)
    true_pd, y = simulated_book(rng)
    calibrated = true_pd
    understated = true_pd / 3  # same order, every probability a third of the truth

    table(
        ["Model", "Gini", "KS", "Brier", "Log loss"],
        [
            ["Calibrated", f"{m.gini(y, calibrated):.3f}", f"{m.ks_statistic(y, calibrated):.3f}",
             f"{m.brier_score(y, calibrated):.4f}", f"{m.log_loss(y, calibrated):.4f}"],
            ["Understated by 3x", f"{m.gini(y, understated):.3f}", f"{m.ks_statistic(y, understated):.3f}",
             f"{m.brier_score(y, understated):.4f}", f"{m.log_loss(y, understated):.4f}"],
        ],
    )
    top_calibrated = m.reliability_table(y, calibrated)[-1]
    top_understated = m.reliability_table(y, understated)[-1]
    table(
        ["Riskiest 10% of the book", "Model says", "Observed default rate"],
        [
            ["Calibrated", f"{top_calibrated['mean_predicted']:.1%}", f"{top_calibrated['observed_rate']:.1%}"],
            ["Understated by 3x", f"{top_understated['mean_predicted']:.1%}", f"{top_understated['observed_rate']:.1%}"],
        ],
    )
    print("Gini and KS cannot tell these models apart. Brier and log loss can,")
    print("and a loan priced on the second model is priced for a third of its risk.")


# --- 4 ---------------------------------------------------------------------------


def trap_4_the_cutoff_is_a_trade_off() -> None:
    title(4, "every cut-off trades one number the business feels for another")
    rng = np.random.default_rng(SEED)
    true_pd, y = simulated_book(rng)
    rows = []
    for cutoff in (0.02, 0.05, 0.10, 0.20):
        rows.append([
            f"PD < {cutoff:.0%}",
            f"{m.approval_rate(true_pd, cutoff):.1%}",
            f"{m.bad_rate_at_cutoff(y, true_pd, cutoff):.2%}",
        ])
    table(["Approve if", "Approval rate", "Bad rate among approved"], rows)

    # Fraud: 0.2% of transactions, a decent but imperfect score.
    n = 500_000
    fraud = (rng.random(n) < 0.002).astype(int)
    score = rng.normal(0, 1, n) + 2.5 * fraud
    rows = []
    for target_recall in (0.50, 0.80, 0.95):
        threshold = np.quantile(score[fraud == 1], 1 - target_recall)
        flagged = (score >= threshold).astype(int)
        rows.append([
            f"{m.recall(fraud, flagged):.0%}",
            f"{m.precision(fraud, flagged):.1%}",
            f"{m.false_positives_per_case_caught(fraud, flagged):.0f}",
        ])
    table(["Fraud caught (recall)", "Precision", "Good transactions blocked per fraud caught"], rows)
    print("Neither table has a right answer. The cut-off is a business decision;")
    print("the model's job is to make every row of these tables better.")


# --- 5 ---------------------------------------------------------------------------


def split_comparison(seed: int) -> dict[str, dict[str, float]]:
    """Fit the same model under a random split and an out-of-time split."""
    rng = np.random.default_rng(seed)
    n_months, per_month = 48, 3000
    month = np.repeat(np.arange(n_months), per_month)
    # The economy: a mean-reverting shock shared by every loan booked in a month.
    economy = np.zeros(n_months)
    for t in range(1, n_months):
        economy[t] = 0.7 * economy[t - 1] + rng.normal(0, 0.6)
    # A month-level variable the model can see: it identifies the month
    # without explaining its default rate.
    market_rate = 5 + 0.05 * np.arange(n_months) + rng.normal(0, 0.3, n_months)

    risk_score = rng.normal(0, 1, len(month))
    dti = rng.normal(0, 1, len(month))
    true_logit = -3.3 + 0.9 * risk_score + 0.4 * dti + economy[month]
    y = (rng.random(len(month)) < sigmoid(true_logit)).astype(int)
    x = np.column_stack([risk_score, dti, market_rate[month]])

    def fit_and_score(train: np.ndarray, test: np.ndarray) -> dict[str, float]:
        model = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
        model.fit(x[train], y[train])
        p = model.predict_proba(x[test])[:, 1]
        return {"base_rate": y[test].mean(), "roc_auc": m.roc_auc(y[test], p), "pr_auc": m.pr_auc(y[test], p)}

    shuffled = rng.permutation(len(month))
    cut = int(0.8 * len(month))
    return {
        "random": fit_and_score(shuffled[:cut], shuffled[cut:]),
        "out_of_time": fit_and_score(np.flatnonzero(month < 36), np.flatnonzero(month >= 36)),
    }


def trap_5_random_split_flatters() -> None:
    title(5, "a random split looks better than the future will be")
    result = split_comparison(SEED)
    table(
        ["Validation", "Test base rate", "ROC-AUC", "PR-AUC"],
        [
            [label, f"{r['base_rate']:.2%}", f"{r['roc_auc']:.3f}", f"{r['pr_auc']:.3f}"]
            for label, r in (
                ("Random 80/20 split of all 48 months", result["random"]),
                ("Train months 1-36, test months 37-48", result["out_of_time"]),
            )
        ],
    )

    # One simulated economy could be a fluke, so repeat with ten others.
    gaps = []
    for seed in range(10):
        r = split_comparison(seed)
        gaps.append(r["random"]["roc_auc"] - r["out_of_time"]["roc_auc"])
    print(
        f"Across ten simulated economies the random split scored higher in "
        f"{sum(g > 0 for g in gaps)} of 10, by {min(gaps):.3f} to {max(gaps):.3f} "
        f"ROC-AUC (mean {np.mean(gaps):.3f})."
    )
    print()
    print("With a random split, every test month also appears in training, so the")
    print("model learns each month's default level from the market rate. The future")
    print("has months it has never seen. Only the second row is a forecast. Compare")
    print("ROC-AUC here, not PR-AUC: the two test sets have different base rates.")


# --- 6 ---------------------------------------------------------------------------


def trap_6_you_only_see_defaults_you_approved() -> None:
    title(6, "a model trained on approved loans has never seen the riskiest applicants")
    rng = np.random.default_rng(SEED)
    n = 200_000
    bureau = rng.normal(0, 1, n)  # in the data
    underwriter_view = rng.normal(0, 1, n)  # used by the old policy, never recorded
    true_pd = sigmoid(-3.0 + 0.9 * bureau + 0.9 * underwriter_view)
    y = (rng.random(n) < true_pd).astype(int)

    # The old policy approved the 60% it judged safest, using both.
    old_view = bureau + underwriter_view
    approved = old_view < np.quantile(old_view, 0.60)

    model = LogisticRegression()
    model.fit(bureau[approved].reshape(-1, 1), y[approved])
    p = model.predict_proba(bureau.reshape(-1, 1))[:, 1]

    rows = []
    for name, group in (("Approved (training data)", approved), ("Declined (never labelled)", ~approved)):
        rows.append([name, f"{p[group].mean():.1%}", f"{y[group].mean():.1%}"])
    table(["Applicants", "Model's average PD", "Actual default rate"], rows)
    print("On the loans it learned from, the model is calibrated. On the people the")
    print("old policy declined, it has no idea how bad they are: the information that")
    print("declined them never reached the training data.")


# --- 7 ---------------------------------------------------------------------------


def trap_7_psi_as_a_drift_alarm() -> None:
    title(7, "PSI: a drift alarm, not a quality score")
    rng = np.random.default_rng(SEED)
    development = rng.normal(0, 1, 50_000)
    rows = []
    for label, shift in (
        ("Same population", 0.0),
        ("Small shift (+0.2 sd)", 0.2),
        ("Moderate shift (+0.4 sd)", 0.4),
        ("Large shift (+0.6 sd)", 0.6),
    ):
        new_population = rng.normal(shift, 1, 50_000)
        value = m.psi(development, new_population)
        band = "stable" if value < 0.10 else ("watch" if value < 0.25 else "review")
        rows.append([label, f"{value:.3f}", band])
    table(["Scored population vs development", "PSI", "Common reading"], rows)
    print("The bands (below 0.10 stable, 0.10-0.25 watch, above 0.25 review) are a")
    print("widely used rule of thumb, not a law. PSI says the population moved;")
    print("it does not say whether the model got worse.")


TRAPS = {
    1: trap_1_accuracy_lies,
    2: trap_2_roc_flatters_rare_events,
    3: trap_3_ranking_is_not_calibration,
    4: trap_4_the_cutoff_is_a_trade_off,
    5: trap_5_random_split_flatters,
    6: trap_6_you_only_see_defaults_you_approved,
    7: trap_7_psi_as_a_drift_alarm,
}

if __name__ == "__main__":
    chosen = [int(a) for a in sys.argv[1:]] or list(TRAPS)
    for n in chosen:
        TRAPS[n]()
