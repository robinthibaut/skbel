# Neural posterior ensemble

`skbel.neural` is an optional conditional posterior for paired arrays: given
predictors `X`, shape `(rows, features)`, and targets `Y`, shape
`(rows, targets)`, it learns a Gaussian mixture over the whole target vector
and returns joint posterior draws, shape `(cases, draws, targets)`. The draws
go straight into {doc}`evaluation`, {mod}`skbel.metrics` and {doc}`design`.
Nothing in the module knows about the physics or the coordinates of the
targets: draws come back in the coordinates of `Y`, and any transform to or
from those coordinates is the caller's.

Training needs PyTorch, which is an optional extra:

```sh
pip install 'skbel[neural]'
```

Importing `skbel` or `skbel.neural`, the recalibrator and the NumPy sampler of
precomputed mixtures work without it. Fitting, loading or evaluating a network
without PyTorch raises an `ImportError` that names the extra.

## The model

Each member is a {class}`~skbel.neural.posterior.MixturePosterior`:

- Predictors pass through a `StandardScaler` and a randomized `PCA(n_pca)`,
  both fitted on the training rows only.
- A multilayer perceptron with SiLU hidden layers (float64, CPU) outputs, per
  case, the weights, means and shape of an `n_components` Gaussian mixture
  over the joint target vector. With `covariance="diagonal"` each component
  has independent log-scales `lo + (hi - lo) * sigmoid(raw)`; with
  `covariance="full"` it has a Cholesky factor `L` with that diagonal and free
  strictly-lower entries, so the component covariance is `L L^T`.
- Adam minimizes the mean joint negative log density. The **last**
  `n_validation` rows, in the order given, are held out internally and choose
  the number of epochs by early stopping (`patience`, `min_improvement`). The
  scaler, PCA and network are then refitted on every row for that number of
  epochs with the same seeds.
- Nothing is jittered, clipped or repaired. A non-finite loss, parameter or
  mixture output raises.

{class}`~skbel.neural.posterior.NeuralPosterior` is the scikit-learn estimator.
It fits `n_members` such members with independent seeds and samples their
equal-weight mixture: every draw picks a member with probability
`1 / n_members`, then a component of that member by its weight, then one
Gaussian vector of that component shared by all targets.

```python
from skbel.neural import NeuralPosterior

posterior = NeuralPosterior(n_members=5, covariance="full", random_state=0)
posterior.fit(X_train, Y_train)
draws = posterior.sample(X_test, 200, case_ids=test_ids, seed=1)  # (cases, 200, targets)
```

The defaults (`n_components=20`, `hidden=(128, 128, 128)`, `n_pca=64`,
`learning_rate=1e-3`, `batch_size=512`, `max_epochs=200`,
`n_validation=1024`, `patience=20`, `min_improvement=1e-4`,
`log_scale_bounds=(-7, 3)`) assume thousands of training rows; small problems
need smaller values, as in the example below.

## Seeds and reproducibility

- Member `m` uses three seeds, `pca`, `init` and `shuffle`. By default they are
  derived from `random_state` with
  {func}`~skbel.neural.posterior.member_seeds` (`SeedSequence(random_state,
  spawn_key=(m, stream))`), so adding members never changes earlier ones.
  `member_seeds` gives them explicitly, for example to reproduce particular
  members. `random_state` must be an integer.
- Training, sampling and the recalibrator use local generators only; they
  leave NumPy's legacy global state and PyTorch's global generator unchanged
  and check that they did.
- `sample` requires explicit `case_ids` and `seed`. Case `c` draws from
  `SeedSequence(seed, spawn_key=(c,))` alone, so its draws do not depend on
  which other cases are evaluated, their order or the batch size `chunk`
  (up to floating-point rounding of batched network passes; member
  labels are exact). This differs from
  {func}`~skbel.evaluation.sample_bel_posterior`, whose streams follow row
  positions.
- `sample(..., return_labels=True)` also returns the `member` and `component`
  label of every draw. `predict_mixtures` returns each member's mixture
  parameters, and {func}`~skbel.neural.posterior.sample_ensemble_latent`
  samples such precomputed mixtures with NumPy alone.

## Persistence

`posterior.save(directory)` writes `member_<i>.npz` files and an
`ensemble.json` record into a new directory and refuses to overwrite.
`NeuralPosterior.load(directory)` or `NeuralPosterior.load([member files])`
rebuilds the estimator without training, reading its parameters and seeds
from the members. Files hold arrays and JSON strings only and are read with
`allow_pickle=False`; an ensemble record must name equal weights and plain
`.npz` member files inside its directory. A loaded estimator has
`random_state=None` and explicit `member_seeds`, so a clone refits the same
members.

## Design comparison

`NeuralPosterior` keeps its parameters unchanged until `fit`, so
{func}`sklearn.base.clone` gives an unfitted copy and it can be the template of
{func}`~skbel.design.fit_design` and {func}`~skbel.design.fit_designs`, which
fit one fresh clone per design:

```python
from skbel.design import fit_designs
from skbel.evaluation import score_samples

refits = fit_designs(
    NeuralPosterior(n_members=5, random_state=0), bank, designs, train_rows=fit_rows
)
for refit in refits:
    draws = refit.model.sample(refit.features(test_observations), 200, case_ids=test_ids, seed=1)
    scores = score_samples(draws, test_targets, levels=[0.5, 0.9])
```

## Quantile recalibration

{class}`~skbel.neural.posterior.QuantileRecalibrator` adjusts sample-based
posteriors marginally, per target and per caller-defined group.

**Fitting** uses held-out cases with known truths, for example a development
set that the networks did not train on:

```python
from skbel.neural import QuantileRecalibrator

recal = QuantileRecalibrator().fit(
    dev_draws, dev_truth, dev_groups, seed=2, min_cases=100, case_ids=dev_ids
)
```

1. For every case and target, a randomized PIT value
   `u = (#draws < truth + U (#draws == truth + 1)) / (M + 1)` is computed
   exactly. The tie-breaker `U` lies on a finite open grid keyed by
   (case ID, target), so `u` is strictly inside (0, 1) without clipping.
2. For every target and every group label with at least `min_cases` cases,
   the map `H` is the piecewise-linear empirical mid-CDF of those PIT values,
   with exact end points `(0, 0)` and `(1, 1)`; it is strictly increasing.
3. A label with fewer cases is recorded INCOMPLETE (`recal.incomplete()`).
   There is no pooled fallback, smoothing or extra sampling.

**Application** takes draws and *routed* group labels only, never truths:

```python
adjusted = recal.transform(test_draws, test_groups)
```

For each draw, its within-case midrank probability `p` is mapped to
`v = H^-1(p)` of its routed group, and the draw is replaced by the empirical
`v`-quantile of the same case and target. Draw indices are kept, so the joint
draw order shared across targets is preserved and ties stay ties. An identity
map returns the draws exactly. Routing a case to an INCOMPLETE or unknown
group raises.

Limits of the recalibration:

- It is **marginal**. Each target is recalibrated on its own; the dependence
  between targets is only carried along by keeping the draw indices, not
  recalibrated. Joint scores such as the energy score are not guaranteed to
  improve.
- Its guarantee is about the fitting population of each group. At
  application time the true group of a case is usually unknown, so groups are
  routed by a caller rule that uses information available before the truth
  (a proxy). When the route differs from the group the truth belongs to, the
  case receives another group's map, and calibration holds only as far as
  the routing matches.
- Adjusted values are draws of the same case: the transform cannot widen the
  range of the raw draws.

## Example

`examples/neural_posterior.py` runs the whole chain on a seeded synthetic
bank with tiny networks: two designs refitted with
{func}`~skbel.design.fit_designs`, recalibration on development rows with a
routing rule that reads the observations only, then
{func}`~skbel.evaluation.score_samples`,
{func}`~skbel.evaluation.event_probabilities` and
{func}`~skbel.evaluation.decision_risks` on test rows.

Results on synthetic data show that the chain runs; they do not show that the
model is calibrated for other data or settings.

## API

See {doc}`api/neural`.
