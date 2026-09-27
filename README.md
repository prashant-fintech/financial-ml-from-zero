# Financial machine learning from zero

Companion code for the
[financial machine learning series](https://singhprashant.in/writing/category/machine-learning)
on singhprashant.in: machine learning for credit scoring, fraud detection,
and anti-money-laundering analytics, built in PyTorch and runnable on a
laptop CPU.

Each post that has code gets its own folder with a README, a
`requirements.txt`, and scripts you run in order.

| Part | Post | Folder |
|---|---|---|
| 1 | [The vocabulary, the traps, and the metrics that matter](https://singhprashant.in/writing/financial-ml-01-vocabulary-traps-metrics) | No code |
| 2 | [Tensors, time-ordered batching, and why leakage hides in `.shift()`](https://singhprashant.in/writing/financial-ml-02-tensors-batching-leakage) | [`part-02-leakage`](part-02-leakage) |

## Data

Nothing under `data/` is committed. Part 2 includes a generator for a
synthetic panel in Freddie Mac's file layout, so everything runs without
downloading anything. To use the real Freddie Mac Single-Family
Loan-Level Dataset, register (free) on Freddie Mac's website and follow
the instructions in the part's README; its terms do not allow
redistribution.
