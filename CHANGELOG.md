# Changelog

## 2.3.0 - 2026-10-09

### Changed behavior

- **Posterior sampling randomness.** `BEL.random_sample` draws each observation
  row from its own generator, derived from the model's `random_state` and the
  row index. This applies to the `mvn`, `kde` and `tm` modes. Repeated calls
  with the same seed return the same samples, a row selected with `obs_n`
  receives the same samples as that row of the full batch, and rows with
  identical observations receive independent draws.
- **Exception in `tm` mode.** When a component's requested draw count equals
  the number of training rows of its map, that component maps the training
  samples instead of drawing independent random reference values. These
  samples use no random numbers, do not depend on the seed and are not
  independent across rows.
- Setting `seed` or `random_state` only stores the value, which must be a
  non-negative integer or `None`. NumPy's global random state is no longer read
  or reseeded by the setters or by sampling. If the seed is `None`, sampling
  takes a seed from operating-system entropy and stores it, so later calls
  reproduce those draws until the seed is changed.
- **The same seed now gives different numerical samples than in 2.2.0.** Code
  that relied on `bel.seed = s` also seeding other NumPy-based steps, such as a
  randomized PCA fit, must seed those steps explicitly.
- `skbel.algorithms.it_sampling` accepts an optional `rng` argument. Without it,
  it still uses NumPy's global random state.
- **Paired rows and pass-through transforms.** With the default no-op
  pipelines, `BEL.fit` keeps the paired predictor and target rows, including
  single-row inputs, and rejects inputs with differing row counts with
  `ValueError`. The forward `BEL.transform` returns the inputs unchanged in
  their given order, for paired, predictor-only and target-only calls. Configured
  pre- and post-processing is still applied.
- **Target-only CCA transforms.** `BEL.cca_pc_transform` with only `X` or only
  `Y` returns the same values as the fitted CCA's own `transform`, for one row
  or several rows at once, and does not modify the caller's arrays.
- **MVN predictor dimension.** In `mode="mvn"` the projected noise covariance
  uses the predictor dimension actually retained by the fitted pre-processing,
  including integer, default and variance-fraction PCA, rather than an assumed
  width.
- **Selected-observation conditioning.** In `kde` and `tm` modes,
  `random_sample(..., obs_n=i)` conditions on the selected observation row
  and its own fitted functions. `obs_n` may be negative, an out-of-range index
  raises, and the caller's array is not modified.
- **Scalar CDF queries.** The CDF returned by `skbel.algorithms.statistics.get_cdf`
  returns a float for a Python or NumPy scalar, an array for an iterable
  (including an empty one), and gives the same value for a scalar and a
  one-element sequence.
- **Unsupervised DCT fit.** `DiscreteCosineTransform2D.fit(X)` no longer needs a
  target; `fit(X, y)` still works and ignores `y`. Fitting does not modify the
  caller's array.
- **3D grid cell centers.** `skbel.spatial.grid_parameters` with a `z_lim`
  spanning several layers returns three-column cell centers, ordered by layer,
  then row, then column, and offset from any origin. A single layer still
  returns two columns.
- `BEL.predict(..., mode="mvn", noise=value)` validates and uses an explicit
  finite, non-negative scalar for that call. `noise=None` keeps the historical
  default multiplier of `0.01`. The value multiplies the identity covariance in
  PCA-score space before projection through the CCA rotations.
- `CompositePCA(scale=True)` fits a separate `StandardScaler` per block's PCA
  scores during `fit`. `transform` uses the retained training scalers and never
  fits on an evaluation batch. `inverse_transform` undoes the score scaling and
  then each PCA inverse transform, and raises `ValueError` on block-count or
  width mismatches. The `scale=False` path is unchanged.

### Added

- **Calibration checks** (`skbel.metrics`): `sbc_ranks`, `sbc_rank_histogram`,
  `empirical_pit`, `interval_coverage`, `summarize_coverage` and `case_rng`.
  Tie-breaking is reproducible per case and target and does not use NumPy's
  global random state. All checks are marginal.
- **Posterior metrics** (`skbel.metrics`): `energy_score` for joint empirical
  posteriors, plus weighted `marginal_crps`, `brier_score`,
  `expected_action_losses` and `bayes_action_set`.
- **Evaluation helpers** (`skbel.evaluation`): `sample_bel_posterior`,
  `score_samples`, `event_probabilities`, `event_brier_scores` and
  `decision_risks`.
- **Design comparison** (`skbel.design`): `SimulationBank`, `cadence_window`,
  `fit_design` and `fit_designs`, which select measurement subsets from one
  shared simulation bank and refit a fresh model per design.
- **Decision tools** (`skbel.metrics`): `rank_prospective_measurements` and
  `finite_acquisition_policy`; `skbel.metrics.robust.robust_acquisition_policy`
  for a finite set of candidate models.
- **Optional neural posterior** (`skbel.neural`): `NeuralPosterior`, an ensemble
  of Gaussian-mixture networks over the joint target vector, with seeded
  per-case sampling, save and load, and `QuantileRecalibrator` for marginal
  recalibration per caller-defined group. It requires the optional `neural`
  extra (`pip install 'skbel[neural]'`). The core package does not require
  PyTorch, and importing `skbel` or `skbel.neural` works without it.
- **Prediction capsules**: `skbel.learning.portable` (linear-MVN moments) and
  `skbel.learning.portable_kde` (KDE) export a fitted `BEL` to a data-only,
  deterministic JSON document that is restored without fitting or pickle.
- Documentation pages for the above and a `neural_posterior` example.

### Experimental

- The prediction capsules and the exact acquisition planners
  (`finite_acquisition_policy`, `robust_acquisition_policy`) are experimental
  and their interfaces may change in a later release.
- Capsules support limited model profiles and raise an error otherwise. Input
  is limited to 4 MiB and a SHA-256 digest is an integrity check against an
  externally supplied value, not tamper-proofing. Capsule contents remain
  derived from the training data.
- The exact planners keep their existing caps and reject larger inputs with
  `ValueError` rather than pruning. `finite_acquisition_policy` supports
  horizon 0, 1 or 2. `robust_acquisition_policy` accepts at most 8 models, 32
  atoms, 3 candidates, 3 actions and 2 outcome labels per candidate. The caps
  are acceptance bounds, not guarantees of runtime or memory.
- None of these tools establishes calibration or validity of a posterior for a
  particular application.

### Packaging

- The source distribution now includes `CHANGELOG.md`.
