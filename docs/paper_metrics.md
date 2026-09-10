# Posterior metrics

These reusable metrics score caller-supplied posterior draws and finite loss
tables. They do not establish calibration, online utility, or a scientific
workflow.

## Marginal CRPS

`marginal_crps(..., estimator="empirical")` computes the exact CRPS of a
finite empirical distribution, optionally with nonnegative draw weights.
Weights are max-rescaled before summation, so finite very-large weights remain
well-defined; zero-mass rows are rejected.

Contract: `samples` has shape `(cases, draws, targets)` and `truth` has shape
`(cases, targets)`, with no broadcasting. Optional `weights` has shape
`(cases, draws)` and is finite, nonnegative, and strictly positive per row.
The return value has shape `(cases, targets)`. The `"unbiased"` estimator
requires unweighted iid draws and at least two draws.

`estimator="unbiased"` is the iid finite-sample U-estimator with a single
fixed truth value. The triangle inequality gives
`sum_{i != j}|x_i - x_j| <= 2*(M-1)*sum_i|x_i - y|`, so it is nonnegative for
every finite sample in exact arithmetic; floating-point roundoff can cause
tiny negative values. Its unbiasedness does not require negative realizations.

## Decisions

`brier_score` accepts finite nonnegative probabilities of shape
`(cases, classes)` whose rows sum to one, plus integer `labels` of shape
`(cases,)`. It returns shape `(cases,)`, with the sum of squared errors over
all classes. For two classes this is exactly factor 2 times the scalar binary
convention `(p_positive - y)**2`.

`expected_action_losses` accepts a finite loss table of shape
`(draws, actions)` and optional finite nonnegative `weights` of shape
`(draws,)` (strictly positive total). It returns shape `(actions,)`, the
posterior-weighted mean loss per action. `bayes_action_set` has the same input
contract and returns `(min_risk, minimizers)`, where `minimizers` is a sorted
integer array containing every exact minimizing action. Their weights use the
same stable normalization and validation rules.

Tiny executable example:

```python
import numpy as np
from skbel.metrics import bayes_action_set, marginal_crps

draws = np.array([[[0.0], [2.0]]])
print(marginal_crps(draws, np.array([[1.0]])))  # [[0.5]]
print(bayes_action_set(np.array([[1.0, 3.0], [1.0, 3.0]])))  # (1.0, [0])
```
