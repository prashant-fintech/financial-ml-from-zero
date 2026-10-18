"""Small, inspectable probability models and a review policy."""

import math

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torch import nn


class LogisticRegression(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1, dtype=torch.float64)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x.to(torch.float64)).squeeze(-1)


def fit_logistic(x, y, positive_weight=1.0):
    if not math.isfinite(positive_weight) or positive_weight <= 0:
        raise ValueError("positive_weight must be finite and positive")
    if set(y.tolist()) != {0.0, 1.0}:
        raise ValueError("Training requires both outcomes")
    if not torch.isfinite(x).all():
        raise ValueError("Features contain non-finite values")
    model = LogisticRegression(x.shape[1])
    target = y.to(torch.float64)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, dtype=torch.float64))
    optimiser = torch.optim.LBFGS(
        model.parameters(), max_iter=500, tolerance_grad=1e-9,
        tolerance_change=1e-12, line_search_fn="strong_wolfe",
    )

    def closure():
        optimiser.zero_grad()
        loss = loss_fn(model(x), target)
        loss.backward()
        return loss

    optimiser.step(closure)
    final_loss = float(closure().detach())
    gradient = max(float(p.grad.abs().max()) for p in model.parameters())
    if not math.isfinite(final_loss) or not math.isfinite(gradient) or gradient > 1e-5:
        raise RuntimeError(f"Fit did not converge: loss={final_loss}, max gradient={gradient}")
    return model, {"loss": final_loss, "max_abs_gradient": gradient}


def export_model(model, logit_offset=0.0):
    return {
        "weights": model.linear.weight.detach().numpy().ravel().tolist(),
        "bias": float(model.linear.bias.detach().item()),
        "logit_offset": float(logit_offset),
    }


def predict(spec, x):
    # Float64 throughout scoring avoids turning large logits into artificial ties.
    values = torch.as_tensor(np.asarray(x), dtype=torch.float64)
    weights = torch.tensor(spec["weights"], dtype=torch.float64)
    logits = values @ weights + spec["bias"] - spec["logit_offset"]
    return torch.sigmoid(logits).numpy()


def probability_metrics(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Labels and probabilities must be non-empty matching vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected binary labels and finite probabilities in [0, 1]")
    both_classes = len(np.unique(y)) == 2
    return {
        "rows": len(y), "events": int(y.sum()), "event_rate": float(y.mean()),
        "mean_prediction": float(p.mean()),
        "average_precision": float(average_precision_score(y, p)) if y.sum() else None,
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else None,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def reliability_table(y, p):
    edges = np.array([0, .01, .02, .05, .10, .20, .40, .60, .80, 1.0])
    frame = pd.DataFrame({"outcome": y, "prediction": p})
    frame["bucket"] = pd.cut(frame["prediction"], edges, include_lowest=True)
    return frame.groupby("bucket", observed=True).agg(
        rows=("outcome", "size"), events=("outcome", "sum"),
        mean_prediction=("prediction", "mean"), event_rate=("outcome", "mean"),
    ).reset_index()


def capacity_threshold(p, fraction=0.10):
    """Largest whole score groups within the development review budget; never split ties."""
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected a non-empty probability vector")
    if not 0 < fraction < 1:
        raise ValueError("Review fraction must be between 0 and 1")
    values, counts = np.unique(p, return_counts=True)
    values, counts = values[::-1], counts[::-1]
    allowed = np.cumsum(counts) <= math.floor(len(p) * fraction)
    # A finite cutoff above one means no review, including for a constant model.
    return float(values[allowed][-1]) if allowed.any() else float(np.nextafter(1.0, 2.0))


def review_metrics(y, p, threshold):
    y = np.asarray(y, dtype=int)
    flags = np.asarray(p) >= threshold
    tp = int((flags & (y == 1)).sum())
    fp = int((flags & (y == 0)).sum())
    fn = int((~flags & (y == 1)).sum())
    return {
        "threshold": threshold, "reviewed": int(flags.sum()),
        "review_rate": float(flags.mean()), "true_positives": tp,
        "false_positives": fp, "false_negatives": fn,
        "true_negatives": int((~flags & (y == 0)).sum()),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }
