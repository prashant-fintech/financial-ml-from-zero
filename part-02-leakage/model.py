"""A deliberately plain model, so that score changes come from the data, not the model.

Part 3 of the series builds the probability-of-default baseline
properly. Here logistic regression is a measuring instrument.
"""

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from datasets import LoanMonthDataset


class LogisticRegression(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)  # logits


def train_logistic(train: LoanMonthDataset, epochs: int = 4, batch_size: int = 2048, seed: int = 0) -> LogisticRegression:
    torch.manual_seed(seed)
    model = LogisticRegression(train.x.shape[1])
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    # Shuffling inside the training window is fine: every row already
    # carries only what was known at its own decision month.
    loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    for _ in range(epochs):
        for x, y in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimiser.step()
    return model


def fit_to_convergence(x: torch.Tensor, y: torch.Tensor, seed: int = 0) -> LogisticRegression:
    """Full-batch L-BFGS: the exact optimum, so score differences come from the data.

    Mini-batch Adam for a few epochs can stop short of the optimum, and how
    short depends on feature scaling. That turns preprocessing choices into
    score differences that have nothing to do with leakage.
    """
    torch.manual_seed(seed)
    model = LogisticRegression(x.shape[1])
    optimiser = torch.optim.LBFGS(model.parameters(), max_iter=500, line_search_fn="strong_wolfe")
    loss_fn = nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        return loss

    optimiser.step(closure)
    return model


@torch.no_grad()
def predict_proba(model: LogisticRegression, x: torch.Tensor) -> np.ndarray:
    return torch.sigmoid(model(x)).numpy()


def score(y_true: torch.Tensor, p: np.ndarray) -> dict[str, float]:
    y = y_true.numpy()
    return {
        "base_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
    }
