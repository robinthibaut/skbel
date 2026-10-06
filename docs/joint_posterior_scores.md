# Joint posterior scores

`skbel.metrics.energy_score(samples, truth, weights=None)` scores a *joint*
empirical posterior against a realized truth vector, one float64 value per case
(lower is better). It complements `marginal_crps`, which scores each target
independently and cannot see dependence between targets.

## Definition

For one case with normalized weights `w_i` and draws `x_i`:

```
ES = sum_i w_i ||x_i - y||_2  -  0.5 * sum_ij w_i w_j ||x_i - x_j||_2
```

`||.||_2` is the Euclidean distance across all targets jointly.

## Inputs

| Argument  | Shape                        | Rules |
|-----------|------------------------------|-------|
| `samples` | `(cases, draws, targets)`    | real numeric, finite, no empty axis |
| `truth`   | `(cases, targets)`           | real numeric, finite, exact shape |
| `weights` | `None` or `(cases, draws)`   | finite, non-negative, positive total per case, exact shape (no broadcasting) |

Wrong shapes, bool/complex/string dtypes and non-finite values are rejected.
Weights are renormalized per case. Zero-weight draws are validated, then
excluded before any distance is computed. If supplied positive weight mass is
lost (underflows) during rescaling or normalization, the call raises.

## Numerical behaviour

- Norms are computed with peak scaling, so representable distances such as
  `[3e200, 4e200] -> 5e200` do not overflow while squaring.
- Differences, norms or objectives that are not representable in float64 raise
  `ValueError`. There is no clipping, epsilon or silent scaling.
- The score is non-negative in exact arithmetic, but float64 roundoff can give
  tiny negative values. No strict positivity is guaranteed.
- Pairwise distances are computed in chunks sized toward a target buffer
  (`_PAIR_CHUNK_ELEMENTS`). At least one row is always processed, so a single
  row (`draws * targets`) can exceed the target, and the norm computation holds
  a few buffers of that size. Work is still `O(draws^2 * targets)` per case;
  this is not a bound on arbitrary input size.

## Interpretation limits

- It scores the *supplied* empirical law only. It is not an unbiased score of
  an unknown continuous generator, not decision risk, and not a proof of joint
  calibration.
- Units and scaling of targets are the caller's choice. Heterogeneous physical
  coordinates are never normalized here.

## Dependence control (hand algebra)

Take equal-weight `Q+ = {(0,0), (1,1)}` and `Q- = {(0,1), (1,0)}`. Their
univariate marginals are identical, and each marginal CRPS is `1/4` at every
binary truth coordinate. For truth `(0,0)` or `(1,1)`:

- `ES(Q+) = sqrt(2)/4`
- `ES(Q-) = 1 - sqrt(2)/4`
- gap `= 1 - sqrt(2)/2`

This is derived by hand and checked in `test_joint_posterior_scores.py` against
an independent explicit-loop oracle. It is a unit-test control, not calibration
evidence.
