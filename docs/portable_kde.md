# Portable KDE prediction capsule

`skbel.learning.portable_kde` compiles a fitted affine `BEL` model in KDE mode
into a data-only **prediction capsule**. The capsule conditions **new** raw
predictor rows and returns non-Gaussian draws in original target units without
the training rows, fitted estimators, callbacks, unpickling or any fitting.

```python
import numpy as np

from skbel.learning.portable_kde import KDEPredictionCapsule, export_kde

capsule = export_kde(fitted_bel, bandwidths=[0.5, 0.5])  # raises KDEPredictionError if unsupported
data = capsule.to_bytes()                                # deterministic UTF-8 JSON
digest = capsule.sha256()                                # keep this out of band

restored = KDEPredictionCapsule.from_bytes(data, expected_sha256=digest)
U = np.random.default_rng(0).random((len(X_obs), 100, 2))  # caller-owned uniforms
draws = restored.sample(X_obs, U)                          # (n_cases, 100, R)
first = restored.sample(X_obs, U, obs_n=0)                 # (1, 100, R), same row of X_obs and U
```

The module is imported explicitly; it is not re-exported from `skbel` or
`skbel.learning`. Importing it still imports the whole SKBEL package and its
dependencies.

## Supported profile

`export_kde(bel, bandwidths)` accepts only:

- an exact `BEL` instance (no subclasses) with `mode="kde"`;
- an exact, fitted `sklearn.cross_decomposition.CCA` with exactly 2 components;
- `X_pre_processing` / `Y_pre_processing` built only from exact `Pipeline`,
  `StandardScaler` and `PCA(whiten=False)` steps (`"passthrough"` allowed);
- passthrough `X_post_processing` / `Y_post_processing`;
- no cached observation override (`x_observation`, `x_pre_processed`,
  `y_pre_processed` all `None`);
- finite paired `X_f` / `Y_f` with 2 to 32 rows, and raw predictor and target
  widths `P, R` between 2 and 4;
- one finite positive bandwidth per canonical component.

Everything else (transport-map or MVN modes, power transforms, scaled
post-processing, other regression models, ...) raises `KDEPredictionError`
before any density is fitted.

The compiler **does fit**: per component it repeats the branch choice of
`BEL.predict` (signed Pearson correlation `>= 0.999` between the canonical
scores selects a `LinearRegression`; otherwise one Gaussian `KernelDensity` is
fitted). The bandwidths are used as given. The default `GridSearchCV`
bandwidth search of `BEL.predict` is **not** reproduced, so a capsule matches a
`BEL` whose KDE functions were built with the same fixed bandwidths, not one
that searched its own. The affine maps come from one public `transform` and
one public `inverse_transform` call on a zero-plus-basis array. The fitted
model, its processors and its prediction caches are not changed.

## State and conventions

For raw rows `x` the canonical data are `d = x A_x + b_x`; a canonical target
`z` reconstructs to `y = z A_y + b_y`. Each canonical component is either

- `{"kind": "linear", "slope", "intercept"}`: a point mass at
  `slope * d_j + intercept` (its uniform channel is validated but unused); or
- `{"kind": "pdf", "bandwidth", "x_axis", "y_axis", "density"}`: the joint
  density of (canonical data, canonical target) on the existing grid, with
  `density[row, col]` evaluated at `(x_axis[col], y_axis[row])`.

The density table follows `kde_params`: Gaussian kernel, Euclidean metric,
`atol=1e-4`, `rtol=0`, breadth-first, leaf size 40, a `200 x 200`
`meshgrid(indexing="xy")` grid over `[min - bandwidth, max + bandwidth]`
(`cut=1`, no clip), `exp(score_samples)`, values below `1e-8` set to `0`.

For a query value `d_j` of a PDF component the capsule repeats the existing
conditioning exactly:

1. Cross-section: pixel coordinates `col = 200 (d_j - x_min) / ptp(x)` and rows
   from `0` to `200` at 129 points (the pixel **count** is used, as in
   `posterior_conditional`, not count minus one); `ndimage.map_coordinates`
   with `order=3`, `mode="constant"`, `cval=0`, `prefilter=True`; target line
   `linspace(y_min, y_max, 129)`.
2. Scaling: values with magnitude below `1e-8` set to `0`, multiplied by
   `1 / simpson(|post|, line)`, the `1e-8` cutoff applied again.
3. CDF: normalization `A = romb(post, dx)`; normalized density
   `interp(s) / A` with values below `1e-3` set to `0`; at each of the 129 line
   points the CDF is `0` at the lower end, `1` at the upper end, `0` within
   `1e-4` of the lower end and otherwise the Romberg integral over 129 prefix
   samples.
4. Inverse: `np.interp(U, cdf, line)`. At knots and on CDF plateaus this returns
   the rightmost tied knot, as the existing sampler does.

The components are sampled independently (the existing factorized canonical
approximation); correlation between original targets comes only from the
affine reconstruction. Uncertainty discarded by PCA/CCA truncation is not added
back. Fixed bandwidth means that the KDE is fixed, not the law: none of this
is a calibrated or full joint posterior.

### Narrower failure behaviour

A query is accepted only when the result is a valid law. The call fails with
`KDEPredictionError` if `d_j` lies outside `x_axis`, if the cross-section is
zero, if after the cutoffs any density value is negative or not finite, if
`A < 1e-3`, if the normalized density is zero everywhere, or if the CDF is not
finite, leaves `[0, 1]` or decreases. The existing `BEL` sampler would instead
return zeros, or sample from such a table as it is. The capsule does not clip,
jitter, extrapolate or fall back to a Gaussian. A query inside the support is
therefore not guaranteed to succeed. This narrower behaviour belongs to the
portable profile; it does not say that every legacy output is invalid.

`sample(X_obs, U, obs_n=None)` accepts a finite real `(n_cases >= 1, P)` array
and uniforms `U` in `[0, 1]` of shape `(n_cases, n_samples >= 1, 2)`, one
channel per canonical component. The capsule never draws random numbers and
never touches NumPy's global random state. `obs_n` selects the same row of
`X_obs` and `U` (negative values count from the end). Results are freshly
allocated arrays the caller owns.

## Wire format

`to_bytes()` returns compact JSON (sorted keys, no NaN) with exactly these keys:
`schema` (`"skbel.kde-prediction-capsule/v1"`), `convention`
(`"gaussian-euclidean-fixed-bw-grid200-cut1-count-pixels-cubic-constant0-prefilter-129-romberg-cutoffs-linear-cdf-inverse/v1"`),
`dimensions` (`predictor`, `canonical`, `target`), `A_x`, `b_x`, `A_y`, `b_y`
and `components`. Linear records hold exactly `kind`, `slope`, `intercept`.
PDF records hold exactly `kind`, `bandwidth`, `x_axis` (200), `y_axis` (200)
and `density` (200 x 200). No training rows, query caches, random state, class
names or paths appear. Equal capsules give equal bytes.

`from_bytes(data, expected_sha256=None)` accepts `bytes` only and:

1. rejects more than 4 MiB (`MAX_BYTES`) before doing anything else;
2. if given, compares the SHA-256 digest with the expected one before parsing;
3. rejects invalid UTF-8/JSON, duplicate keys, `NaN`/`Infinity`, nesting deeper
   than 12, unknown or missing keys, other schema versions or conventions,
   Boolean/string/null/nested entries and wrongly shaped arrays;
4. runs the same validation as `from_state` and direct construction: finite
   float64 values, the dimension limits, one record per component, positive
   bandwidths, strictly increasing evenly spaced 200-point axes and
   non-negative density tables with no values in `(0, 1e-8)` and some positive
   mass. All state is copied and stored read-only.

The constructor checks numeric structure only. It cannot promise that every
future cross-section is a valid law, so each query validates its own law. The
loader never uses pickle, joblib, `eval`, `exec`, dynamic imports, files or the
network, and the capsule API reads and writes no paths. The caller owns
transport and storage.

## Capabilities and limitations

- **SHA-256 digest.** A matching SHA-256 only shows that the bytes equal what
  an externally supplied digest describes. A consistent, rehashed alteration
  is a valid capsule of a different model. This is no tamper-proofing and no
  security sandbox beyond the parsing rules above.
- **Data sensitivity.** The density tables are derived from the paired training
  data and remain sensitive. They are smoothed, but they are not de-identified.
- **Not probability calibration.** Draws follow the existing KDE convention.
  Agreement with a same-profile BEL establishes implementation fidelity, not
  independent validation of probabilities. These tests do not validate
  probabilities against real observations.
- **Subset of joblib.** A trusted joblib checkpoint restores arbitrary
  Python objects, including default bandwidth search. The capsule only
  transfers numeric conditional-prediction state.
- **Different from MVN export.** `skbel.learning.portable` transfers
  Gaussian moments of `mode="mvn"`; this capsule transfers the KDE/linear
  conditional laws of `mode="kde"`. Each is for its respective mode.
- **Numeric precision.** Floating-point agreement is tested on small fixtures (absolute `1e-10`), not
  proved across library versions or extreme magnitudes. The canonical query is
  computed through the stored affine map rather than the fitted pipeline, so
  values extremely close to one of the cutoffs could be treated differently.

## Tests

`skbel/testing/test_portable_kde.py` uses one literal 16-row fixture (two
predictor and two target features, a linear and a curved target, two raw
queries, a fixed uniform tensor, no random numbers). It checks: parity with
the public `BEL` using the same fixed-bandwidth functions and the same
uniforms; a nested fresh interpreter in which every estimator fit raises;
selected-row and ownership behaviour; a literal bimodal-plus-point-mass state
compared with the existing statistics helpers and explicit affine arithmetic,
including bimodal-shape checks; an exact-rational piecewise-linear inverse-CDF
oracle; fail-closed query laws; hostile documents; unsupported profiles; and a
measured fit and call budget.
