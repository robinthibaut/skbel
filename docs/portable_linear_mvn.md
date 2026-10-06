# Portable linear-MVN prediction capsule

`skbel.learning.portable` compiles a fitted affine `BEL` model into a small,
data-only **prediction capsule**. The capsule restores original-space Gaussian
prediction moments for new predictor rows without refitting, sampling,
unpickling or importing anything from the bytes.

```python
from skbel.learning.portable import LinearMVNCapsule, export_linear_mvn

capsule = export_linear_mvn(fitted_bel)  # raises LinearMVNError if unsupported
mean, cov = capsule.predict_moments(X_obs)  # (n, R), (n, R, R); noise=None -> 0.01
data = capsule.to_bytes()  # deterministic UTF-8 JSON
digest = capsule.sha256()  # keep this out of band

restored = LinearMVNCapsule.from_bytes(data, expected_sha256=digest)
```

The module is imported explicitly; it is not re-exported from `skbel` or
`skbel.learning`. Importing it still imports the whole SKBEL package and its
dependencies: this is not a standalone NumPy-only installation.

## Supported profile

`export_linear_mvn(bel)` accepts only:

- an exact `BEL` instance (no subclasses) with `mode="mvn"`;
- an exact, fitted `sklearn.cross_decomposition.CCA` with at least 2 components;
- `X_pre_processing` / `Y_pre_processing` built only from exact `Pipeline`,
  `StandardScaler` and `PCA(whiten=False)` steps (`"passthrough"` allowed);
- passthrough `X_post_processing` / `Y_post_processing`;
- no cached observation override (`x_observation`, `x_pre_processed`,
  `y_pre_processed` all `None`);
- finite paired `X_f` / `Y_f` of shape `(N, Q)` whose aggregate blocks are
  symmetric, positive definite and well conditioned (condition number at most
  `1e12`);
- `2 <= Q <= min(P, R, 32)` and `P, R <= 128`, where `P` is the raw predictor
  width and `R` the original target width.

Everything else (power transforms, KDE or transport-map modes, custom
callables, unfitted or singular models, ...) raises `LinearMVNError` before a
capsule exists. The compiler calls the public `transform` and
`inverse_transform` once each on a zero-plus-basis array to read the affine
maps. It does not modify the model or its caches, and it keeps no training
rows, estimator objects, seeds or query caches.

## State and conventions

For raw rows `x` the canonical data are `d = x A_x + b_x`; a canonical target
`z` reconstructs to `y = z A_y + b_y`. The ten stored arrays are:

| name | shape | meaning |
| --- | --- | --- |
| `A_x`, `b_x` | `(P, Q)`, `(Q,)` | raw predictor to canonical data |
| `A_y`, `b_y` | `(Q, R)`, `(R,)` | canonical target to original target |
| `mu_y`, `C_y` | `(Q,)`, `(Q, Q)` | canonical target mean and covariance |
| `G` | `(Q, Q)` | least-squares map target to data (`Y_f G.T ~ X_f`) |
| `mu_e`, `C_e` | `(Q,)`, `(Q, Q)` | mean and covariance of the residual |
| `B` | `(Q, Q)` | `x_rotations_.T @ x_rotations_` |

The producer statistics follow `mvn_inference` exactly: coefficient and
target-mean magnitudes below `1e-8` are set to `0`, covariances use `ddof=1`
with centred residuals, and the data covariance is `s * B` with `s = noise`
(default `0.01`; finite, non-negative scalar, no booleans). This is the
existing canonical-space convention, not a physical-space noise model.

Prediction builds the joint block `[[C_y, C_y G.T], [G C_y, G C_y G.T + C_e + s B]]`,
takes its pseudoinverse (NumPy default cutoff, as in `mvn_inference`) and
applies the same posterior mean and covariance formulas. For well-conditioned
positive definite blocks this equals ordinary Schur conditioning. The joint
condition number must be at most `1e12` or the call fails; there is no jitter,
clipping, symmetrization or covariance repair. The original-space covariance is
`A_y.T C_z A_y` and may be rank deficient when `Q < R`. Uncertainty discarded by
the PCA/CCA truncation is **not** added back.

`predict_moments(X_obs, noise=None)` accepts only a finite real 2D array of
shape `(n_cases >= 1, P)` (no broadcasting, booleans, objects or complex
numbers) and returns freshly allocated arrays the caller owns.

## Wire format

`to_bytes()` returns compact JSON (sorted keys, no NaN) with exactly these
keys: `schema` (`"skbel.linear-mvn-capsule/v1"`), `inference_convention`,
`dimensions` (`predictor`, `target`, `canonical`) and the ten arrays above. No
class names, paths, URLs or other metadata appear. Equal capsules give equal
bytes.

`from_bytes(data, expected_sha256=None)` accepts `bytes` only and:

1. rejects more than 4 MiB (`MAX_BYTES`) before doing anything else;
2. if given, compares the SHA-256 digest with the expected one before parsing;
3. rejects absurd bracket nesting, invalid UTF-8/JSON, duplicate keys,
   `NaN`/`Infinity`, unknown or missing keys, other schema versions or
   conventions, boolean/string/null/nested array entries, ragged or wrongly
   shaped arrays and dimensions outside the limits;
4. runs the same validation as direct construction (finite values, symmetric
   positive definite well-conditioned `C_y`, `C_e`, `B`) and copies all state.

The loader never uses pickle, joblib, `eval`, `exec`, dynamic imports, files or
the network, and the capsule API reads and writes no paths; the caller owns
transport and storage.

## What this does not claim

- **Not authenticity.** A matching SHA-256 only shows that the bytes equal what
  an externally supplied digest describes. Anyone who can alter the bytes can
  rehash them, so this is not tamper-proofing, and there is no security-sandbox
  guarantee beyond the parsing rules above.
- **Not privacy.** The aggregate statistics are learned from the paired
  training data and remain sensitive; exporting is not a privacy guarantee.
- **Not calibration.** Moments are computed under the existing MVN
  convention; nothing here validates them against real data.
- **Narrower than joblib.** A trusted joblib checkpoint restores arbitrary
  Python objects; the capsule only transfers numeric prediction state and
  cannot retrain, sample or rebuild the full `BEL`.
- Floating-point agreement is tested on small fixtures (absolute `1e-10`), not
  proved across library versions or extreme magnitudes.

## Tests

`skbel/testing/test_portable_linear_mvn.py` uses one literal fixture (16 rows,
4 predictor features, 2 canonical components, no random numbers): parity with
the actual public `BEL` and a freshly created trusted joblib copy, an
independent exact-rational Schur oracle, a nested fresh-interpreter restore
with no fitting, hostile-document, unsupported-profile and ownership tests,
and a measured call budget.
