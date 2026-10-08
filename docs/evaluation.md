# From a fitted BEL to scores, events and decisions

`skbel.evaluation` connects a fitted `BEL` to the metrics in `skbel.metrics`
with five functions:

| Function | Input | Output |
| --- | --- | --- |
| `sample_bel_posterior(bel, observations, n_draws)` | fitted `BEL`, `(cases, features)` | draws `(cases, n_draws, targets)` in original target units |
| `score_samples(samples, truth, levels=...)` | draws, truth `(cases, targets)` | per-case CRPS, interval coverage, optional energy score |
| `event_probabilities(indicators)` | boolean `(cases, draws, events)` | `(cases, events)` |
| `event_brier_scores(probabilities, outcomes, convention=...)` | `(cases, events)`, boolean `(cases, events)` | retrospective Brier scores |
| `decision_risks(losses)` | loss table `(cases, draws, actions)` | expected losses, minimum risk, every Bayes action |

Only the first function uses `BEL`. The others accept draws from any sampler.
They are functions, not `BEL` methods, and they add no fitting, refitting or
simulation. Events and losses are not derived from targets here: the caller
evaluates them on the draws, in the caller's units, and passes the arrays in.

## Sampling a fitted BEL

`sample_bel_posterior` calls the public `BEL.predict` once with
`return_samples=True` and `inverse_transform=True`, so the draws are in the
original target units. It requires:

- a fitted `BEL` (it never calls `fit`) with no active `x_observation`
  override, because `predict` would then ignore the supplied observations;
- observations as a real, finite `(cases, features)` array, with `features`
  equal to the fitted predictor pre-processing's `n_features_in_` when that is
  available; a single row must still be passed as `(1, features)`;
- `n_draws` as a positive integer.

The output is always a float64 array. Real floating output of another
precision from `predict` is converted to float64 first (float16 and float32
widen exactly); boolean, integer, complex, object and string output raises.
The converted output must have shape `(cases, n_draws, targets)` and be finite,
so a value that overflows float64 raises too. When the fitted target
pre-processing reports `n_features_in_`, `targets` must equal it.

`predict` keeps its normal side effects on the model. It stores `mode` (when
given), `noise` (`None` resets the multiplier to 0.01), `n_posts`, the
projected observations and the posterior state of the mode. In `tm` mode the
transport maps are optimized again on every call. If `bel.seed` is `None`, a
seed is taken from operating-system entropy and stored in `bel.seed`.

Random streams follow `BEL.random_sample`: row `i` of `observations` draws from
the stream of position `i` under `bel.seed`. The same call with the same seed
repeats the draws. Reordering or subsetting the rows changes which stream a
case receives, so the draws of a case are not stable under reordering. Two
further properties of the existing samplers carry over: in `tm` mode with
`n_draws` equal to the number of training rows, the draws are a deterministic
map of the training samples; in `kde` mode a nearly linear component
(correlation of at least 0.999) gives the same value for every draw.

## Scores

`score_samples(samples, truth, levels=..., weights=None,
crps_estimator="empirical", joint=False)` calls `marginal_crps`,
`interval_coverage` and, only when `joint=True`, `energy_score`, all with the
same draws, truth and weights. Their conventions are unchanged: `"unbiased"`
CRPS needs unweighted draws and at least two of them, intervals are closed
inverse-CDF central intervals, and weights are renormalized per case. `levels`
must be given; there is no default.

All results are per case. Nothing is averaged over cases or pooled over
targets; use `summarize_coverage` on the returned `coverage` to aggregate. The
CRPS is in the units of each target. The energy score adds Euclidean distances
across targets, so it mixes their units; it is off by default, and the caller
chooses any scaling before calling.

## Events and their Brier scores

`event_probabilities(indicators, weights=None)` takes booleans of shape
`(cases, draws, events)`, for example
`np.stack([samples[:, :, 0] > 0], axis=-1)`. A draw exactly at a threshold
counts according to the comparison the caller wrote. Without weights the
result is the fraction of draws in each event. With weights it is the event
weight divided by the event weight plus the non-event weight, which stays in
`[0, 1]` without clipping. Integer or float indicators are rejected.

`event_brier_scores(probabilities, outcomes, convention=...)` compares those
probabilities with boolean outcomes of the same shape. The convention is
required:

- `"binary"`: every event on its own, `(p - y) ** 2`, shape `(cases, events)`.
- `"multiclass"`: the events are exclusive and exhaustive classes; each
  probability row sums to 1 and each outcome row has exactly one `True`. The
  result, shape `(cases,)`, is `brier_score`, which sums over every class: for
  two classes it is twice the binary value.

These scores are retrospective. They are not decision losses and do not enter
`decision_risks`.

## Decision risks

`decision_risks(losses, weights=None)` takes the loss of every action under
every draw, shape `(cases, draws, actions)`, and reduces each case with
`expected_action_losses` and `bayes_action_set`. It returns
`expected_losses` `(cases, actions)`, `min_risk` `(cases,)` and
`bayes_actions`, one sorted array per case with every action whose expected
loss equals the minimum exactly. Exact ties are all reported, and none is
picked. Near ties caused by rounding are not ties.

The truth is not an argument, so no retrospective score can change the
actions. Every loss and weight is validated, including those of zero-weight
draws, and an expected loss that overflows raises.

## Worked chain

The example fits a small `BEL` on synthetic linear-Gaussian data and runs the
whole chain on held-out synthetic cases. The event, the losses and the two
actions are illustrative choices made by the caller.

```python
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skbel import BEL
from skbel.evaluation import (
    decision_risks,
    event_brier_scores,
    event_probabilities,
    sample_bel_posterior,
    score_samples,
)
from skbel.metrics import summarize_coverage

rng = np.random.default_rng(0)
y = rng.normal(size=(200, 2))
x = np.column_stack([y[:, 0] + 0.3 * y[:, 1], y[:, 1] - 0.2 * y[:, 0], y.sum(axis=1)])
x = x + 0.2 * rng.normal(size=x.shape)
x_train, y_train, x_test, y_test = x[:150], y[:150], x[150:], y[150:]

bel = BEL(
    mode="mvn",
    X_pre_processing=Pipeline([("scale", StandardScaler())]),
    Y_pre_processing=Pipeline([("scale", StandardScaler())]),
    regression_model=CCA(n_components=2),
    random_state=1,
)
bel.fit(x_train, y_train)

# 1. Draws in original target units, shape (50, 200, 2).
samples = sample_bel_posterior(bel, x_test, 200)

# 2. Per-case scores; aggregation is a separate, explicit step.
scores = score_samples(samples, y_test, levels=[0.5, 0.9])
print("mean CRPS per target", scores.crps.mean(axis=0))
print("coverage", summarize_coverage(scores.coverage).coverage)

# 3. A caller-defined event: the first target is positive.
indicators = samples[:, :, :1] > 0
p_event = event_probabilities(indicators)
brier = event_brier_scores(p_event, y_test[:, :1] > 0, convention="binary")
print("mean binary Brier", brier.mean())

# 4. Two caller-defined actions: act (loss 1 if the first target is negative)
#    or wait (loss 0.5 whatever happens). The truth is not used here.
losses = np.stack(
    [np.where(samples[:, :, 0] < 0, 1.0, 0.0), np.full(samples.shape[:2], 0.5)], axis=-1
)
risks = decision_risks(losses)
print("Bayes actions of the first cases", risks.bayes_actions[:3])
```

Run the code to see the numbers; this page does not quote any.

## Limits

- The functions describe the supplied draws, events and losses. Good scores on
  synthetic cases do not show that a model is calibrated for other data, and
  the example makes no physical, calibration or design claim.
- `noise` in `mvn` mode is the existing multiplier of the projected predictor
  covariance, not a measurement standard deviation in physical units.
- Coverage and CRPS are marginal: they do not establish joint calibration.
- Bayes action sets use exact float64 equality, as `bayes_action_set` does.
