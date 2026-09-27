"""PyTorch datasets over the loan-month panel."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LoanMonthDataset(Dataset):
    """One example per (loan, decision month): a flat feature vector and a label."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, keys: pd.DataFrame):
        assert len(x) == len(y) == len(keys)
        self.x = x
        self.y = y
        self.keys = keys

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[i], self.y[i]

    def key(self, i: int) -> tuple[str, pd.Period]:
        row = self.keys.iloc[i]
        return row["loan_id"], row["period"]


class LoanHistoryDataset(Dataset):
    """One example per decision month: the loan's last `window` months as a sequence.

    `panel_x` holds preprocessed features for every row of a panel sorted
    by (loan_id, period) with a complete monthly calendar. `decision_rows`
    are positions in that panel. Each example is the slice ending at the
    decision row, so it cannot reach past it. Loans with shorter histories
    are left-padded with zeros and a mask marks the real months.
    """

    def __init__(
        self,
        panel_x: np.ndarray,
        panel_loan_ids: pd.Series,
        decision_rows: np.ndarray,
        labels: np.ndarray,
        window: int = 6,
    ):
        self.panel_x = torch.from_numpy(panel_x)
        is_first_month = panel_loan_ids.ne(panel_loan_ids.shift()).to_numpy()
        # Position of each row's loan's first month in the panel.
        self.loan_start = np.maximum.accumulate(np.where(is_first_month, np.arange(len(panel_loan_ids)), 0))
        self.decision_rows = decision_rows
        self.labels = torch.from_numpy(labels.astype(np.float32))
        self.window = window

    def __len__(self) -> int:
        return len(self.decision_rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        end = self.decision_rows[i] + 1  # inclusive of the decision month
        start = max(self.loan_start[self.decision_rows[i]], end - self.window)
        history = self.panel_x[start:end]

        n_features = self.panel_x.shape[1]
        sequence = torch.zeros(self.window, n_features)
        mask = torch.zeros(self.window, dtype=torch.bool)
        sequence[self.window - len(history):] = history
        mask[self.window - len(history):] = True
        return sequence, mask, self.labels[i]
