"""Portable, data-only prediction capsule for the affine linear-MVN BEL profile.

``export_linear_mvn`` compiles a fitted ``BEL`` (mode ``"mvn"``, CCA, affine
scaler/PCA pre-processing, passthrough post-processing) into a
:class:`LinearMVNCapsule`: ten small numeric arrays that are enough to compute
original-space Gaussian prediction moments for new predictor rows.  The capsule
holds no training rows, estimator objects, seeds or caches and its wire format
is strict, bounded JSON.  Restoration never fits, samples, unpickles or imports
anything.

This is deliberately narrower than a trusted joblib checkpoint.  It makes no
privacy, calibration, authenticity or sandboxing claim: learned aggregate
statistics stay sensitive, a SHA-256 digest only detects change against an
externally supplied expectation, and uncertainty discarded by PCA/CCA
truncation is not added back to the original-space covariance.
"""

import hashlib
import hmac
import json
from numbers import Real

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .bel import BEL

__all__ = [
    "INFERENCE_CONVENTION",
    "MAX_BYTES",
    "SCHEMA",
    "LinearMVNCapsule",
    "LinearMVNError",
    "export_linear_mvn",
]

SCHEMA = "skbel.linear-mvn-capsule/v1"
INFERENCE_CONVENTION = (
    "mvn-precision-block-pinv;ddof-1;zero-threshold-1e-8;"
    "noise-scale-times-x-rotations-gram;default-noise-0.01;original-affine-moments"
)
MAX_BYTES = 4 * 1024 * 1024
DEFAULT_NOISE = 0.01
MAX_FEATURES = 128
MAX_CANONICAL = 32

_MAX_CONDITION = 1e12
_SYMMETRY_RTOL = 1e-12
_ZERO_THRESHOLD = 1e-8
_MAX_BRACKETS = 512
_MAX_BRACES = 2
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_NOISE_MESSAGE = "noise must be a finite non-negative scalar"
_ARRAY_NAMES = ("A_x", "b_x", "A_y", "b_y", "mu_y", "C_y", "G", "mu_e", "C_e", "B")
_DIMENSION_NAMES = frozenset(("predictor", "target", "canonical"))
_TOP_LEVEL_KEYS = frozenset(("schema", "inference_convention", "dimensions", *_ARRAY_NAMES))
_AFFINE_STEPS = (StandardScaler, PCA)


class LinearMVNError(ValueError):
    """Unsupported profile, malformed wire data or inconsistent capsule state."""


def _check_dimensions(predictor, canonical, target):
    """Validate the (P, Q, R) dimension triple."""
    for name, value in (
        ("predictor", predictor),
        ("canonical", canonical),
        ("target", target),
    ):
        if type(value) is not int:
            raise LinearMVNError(f"dimension {name} must be an integer")
    if predictor > MAX_FEATURES or target > MAX_FEATURES:
        raise LinearMVNError(f"predictor and target dimensions must not exceed {MAX_FEATURES}")
    if canonical < 2 or canonical > min(predictor, target, MAX_CANONICAL):
        raise LinearMVNError("canonical dimension must satisfy 2 <= Q <= min(P, R, 32)")


def _array_shapes(predictor, canonical, target):
    """Exact shape of each named array for the given dimensions."""
    p, q, r = predictor, canonical, target
    return {
        "A_x": (p, q),
        "b_x": (q,),
        "A_y": (q, r),
        "b_y": (r,),
        "mu_y": (q,),
        "C_y": (q, q),
        "G": (q, q),
        "mu_e": (q,),
        "C_e": (q, q),
        "B": (q, q),
    }


def _reject_booleans(name, value):
    """Reject Boolean items in (nested) lists and tuples before NumPy can coerce them."""
    if isinstance(value, (bool, np.bool_)):
        raise LinearMVNError(f"{name} must hold real numbers (no bool, object or complex)")
    if isinstance(value, (list, tuple)):
        for item in value:
            _reject_booleans(name, item)


def _as_real_array(name, value):
    """Return an owned, finite float64 copy of a real numeric array-like."""
    if isinstance(value, np.ma.MaskedArray):
        raise LinearMVNError(f"{name} must not be a masked array")
    try:
        _reject_booleans(name, value)
    except RecursionError as exc:
        raise LinearMVNError(f"{name} is nested too deeply") from exc
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise LinearMVNError(f"{name} must be a real numeric array") from exc
    if array.dtype.kind not in "iuf":
        raise LinearMVNError(f"{name} must hold real numbers (no bool, object or complex)")
    owned = np.array(array, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(owned)):
        raise LinearMVNError(f"{name} must be finite")
    return owned


def _check_moment_matrices(c_y, c_e, gram):
    """Require symmetric, positive definite, well conditioned moment blocks.

    Nothing is repaired: no symmetrization, jitter or eigenvalue clipping.
    """
    for name, matrix in (("C_y", c_y), ("C_e", c_e), ("B", gram)):
        scale = max(1.0, float(np.max(np.abs(matrix))))
        if float(np.max(np.abs(matrix - matrix.T))) > _SYMMETRY_RTOL * scale:
            raise LinearMVNError(f"{name} must be symmetric")
        try:
            eigenvalues = np.linalg.eigvalsh(matrix)
        except np.linalg.LinAlgError as exc:
            raise LinearMVNError(f"{name} eigenvalues could not be computed") from exc
        if not eigenvalues[0] > 0.0 or eigenvalues[-1] / eigenvalues[0] > _MAX_CONDITION:
            raise LinearMVNError(f"{name} must be positive definite and well conditioned")


def _check_observations(X_obs, predictor):
    """Validate and copy query rows: finite real 2D ``(n_cases >= 1, predictor)``."""
    observed = _as_real_array("X_obs", X_obs)
    if observed.ndim != 2 or observed.shape[0] < 1 or observed.shape[1] != predictor:
        raise LinearMVNError(f"X_obs must have shape (n_cases >= 1, {predictor})")
    return observed


def _check_noise(noise):
    """Same noise-multiplier contract as ``BEL.predict``; ``None`` is the default."""
    if noise is None:
        return DEFAULT_NOISE
    if isinstance(noise, (bool, np.bool_)) or not isinstance(noise, Real):
        raise LinearMVNError(_NOISE_MESSAGE)
    try:
        value = float(noise)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LinearMVNError(_NOISE_MESSAGE) from exc
    if not np.isfinite(value) or value < 0:
        raise LinearMVNError(_NOISE_MESSAGE)
    return value


def _canonical_posterior(arrays, observed, scale):
    """Canonical-space posterior mean ``(n, Q)`` and covariance ``(Q, Q)``.

    Mirrors ``mvn_inference``: joint (target, data) block, precision by
    pseudoinverse, covariance by pseudoinverse of the target precision block.
    """
    q = observed.shape[1]
    c_y, g = arrays["C_y"], arrays["G"]
    s12 = c_y @ g.T
    s21 = g @ c_y
    s22 = g @ c_y @ g.T + scale * arrays["B"] + arrays["C_e"]
    block = np.block([[c_y, s12], [s21, s22]])
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            condition = np.linalg.cond(block)
            if not np.isfinite(condition) or condition > _MAX_CONDITION:
                raise LinearMVNError("joint covariance is singular or ill conditioned")
            delta = np.linalg.pinv(block)
            d11 = delta[:q, :q]
            d12 = delta[:q, q:]
            covariance = np.linalg.pinv(d11)
            residual = observed - arrays["mu_e"] - arrays["mu_y"] @ g.T
            mean = (d11 @ arrays["mu_y"] - residual @ d12.T) @ covariance.T
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        raise LinearMVNError("posterior moments are not representable") from exc
    return mean, covariance


class LinearMVNCapsule:
    """Prediction-only, data-only state of an affine linear-MVN BEL model.

    For raw predictor rows ``x`` the canonical data are ``d = x A_x + b_x`` and
    a canonical target ``z`` reconstructs to ``y = z A_y + b_y``.  ``mu_y``,
    ``C_y`` are the canonical target mean/covariance, ``G`` the least-squares
    map target -> data, ``mu_e``, ``C_e`` the residual mean/covariance and
    ``B = x_rotations.T @ x_rotations`` the noise base matrix.

    All arrays are validated, copied and stored read-only; nothing the caller
    passes in (or receives back) aliases the internal state.
    """

    def __init__(self, *, A_x, b_x, A_y, b_y, mu_y, C_y, G, mu_e, C_e, B):
        """Validate and copy the ten arrays; see the class docstring."""
        supplied = {
            "A_x": A_x,
            "b_x": b_x,
            "A_y": A_y,
            "b_y": b_y,
            "mu_y": mu_y,
            "C_y": C_y,
            "G": G,
            "mu_e": mu_e,
            "C_e": C_e,
            "B": B,
        }
        arrays = {name: _as_real_array(name, value) for name, value in supplied.items()}
        if arrays["A_x"].ndim != 2 or arrays["A_y"].ndim != 2:
            raise LinearMVNError("A_x and A_y must be 2D")
        predictor, canonical = arrays["A_x"].shape
        target = arrays["A_y"].shape[1]
        _check_dimensions(predictor, canonical, target)
        for name, shape in _array_shapes(predictor, canonical, target).items():
            if arrays[name].shape != shape:
                raise LinearMVNError(f"{name} must have shape {shape}")
        _check_moment_matrices(arrays["C_y"], arrays["C_e"], arrays["B"])
        for array in arrays.values():
            array.flags.writeable = False
        self._arrays = arrays
        self._predictor = predictor
        self._canonical = canonical
        self._target = target

    def __repr__(self):
        return (
            f"LinearMVNCapsule(predictor={self._predictor}, "
            f"canonical={self._canonical}, target={self._target})"
        )

    @property
    def predictor_dim(self):
        """Number of raw predictor features ``P``."""
        return self._predictor

    @property
    def canonical_dim(self):
        """Number of canonical components ``Q``."""
        return self._canonical

    @property
    def target_dim(self):
        """Number of original target features ``R``."""
        return self._target

    def arrays(self):
        """Independent writable copies of the ten named arrays."""
        return {name: self._arrays[name].copy() for name in _ARRAY_NAMES}

    def predict_moments(self, X_obs, noise=None):
        """Original-space Gaussian moments for each query row.

        :param X_obs: Finite real array ``(n_cases, P)``; no broadcasting.
        :param noise: ``None`` (default 0.01) or a finite non-negative scalar
            multiplier of ``B``, exactly as in ``BEL.predict(mode="mvn")``.
        :return: ``(mean, covariance)`` with shapes ``(n_cases, R)`` and
            ``(n_cases, R, R)``, freshly allocated and owned by the caller.
            The covariance is ``A_y.T C_z A_y`` and may be rank deficient;
            uncertainty discarded by the PCA/CCA truncation is not added.
        """
        observed = _check_observations(X_obs, self._predictor)
        scale = _check_noise(noise)
        arrays = self._arrays
        canonical_data = observed @ arrays["A_x"] + arrays["b_x"]
        mean_z, cov_z = _canonical_posterior(arrays, canonical_data, scale)
        mean = mean_z @ arrays["A_y"] + arrays["b_y"]
        covariance = arrays["A_y"].T @ cov_z @ arrays["A_y"]
        stacked = np.empty((observed.shape[0],) + covariance.shape, dtype=np.float64)
        stacked[...] = covariance
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(stacked))):
            raise LinearMVNError("posterior moments are not representable")
        return mean, stacked

    def to_bytes(self):
        """Deterministic UTF-8 JSON (schema ``skbel.linear-mvn-capsule/v1``)."""
        payload = {
            "schema": SCHEMA,
            "inference_convention": INFERENCE_CONVENTION,
            "dimensions": {
                "predictor": self._predictor,
                "target": self._target,
                "canonical": self._canonical,
            },
        }
        payload.update({name: self._arrays[name].tolist() for name in _ARRAY_NAMES})
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return text.encode("utf-8")

    def sha256(self):
        """Hex SHA-256 of ``to_bytes()``; integrity against an external copy only."""
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data, expected_sha256=None):
        """Restore a capsule from ``to_bytes`` output, validating everything.

        The size limit is checked first, then the optional external digest, and
        only then is the document parsed.  A matching digest is not proof of
        authenticity: whoever can alter the bytes can also alter the digest.

        :param data: ``bytes`` of at most ``MAX_BYTES``.
        :param expected_sha256: Optional hex digest obtained out of band.
        """
        if type(data) is not bytes:
            raise LinearMVNError("capsule data must be bytes")
        if len(data) > MAX_BYTES:
            raise LinearMVNError(f"capsule data exceeds {MAX_BYTES} bytes")
        if expected_sha256 is not None:
            _check_digest(data, expected_sha256)
        if data.count(b"[") > _MAX_BRACKETS or data.count(b"{") > _MAX_BRACES:
            raise LinearMVNError("capsule data is nested too deeply")
        arrays = _arrays_from_payload(_parse_json(data))
        return cls(**arrays)


def _check_digest(data, expected):
    """Compare the SHA-256 of ``data`` with an externally supplied hex digest."""
    if type(expected) is not str or len(expected) != 64 or not set(expected) <= _HEX_DIGITS:
        raise LinearMVNError("expected_sha256 must be a 64 character hex string")
    if not hmac.compare_digest(hashlib.sha256(data).hexdigest(), expected.lower()):
        raise LinearMVNError("SHA-256 digest does not match the expected value")


def _strict_pairs(pairs):
    """JSON object hook that rejects duplicate keys."""
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise LinearMVNError("duplicate keys are not allowed")
    return dict(pairs)


def _reject_constant(token):
    """JSON constant hook: NaN and the infinities are not valid capsule data."""
    raise LinearMVNError(f"non-finite constant {token} is not allowed")


def _parse_json(data):
    """Decode UTF-8 and parse JSON with strict hooks; nothing is executed."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LinearMVNError("capsule data is not valid UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except LinearMVNError:
        raise
    except (ValueError, RecursionError) as exc:
        raise LinearMVNError("capsule data is not valid JSON") from exc


def _array_from_json(name, value, shape):
    """Strictly decode one nested JSON list into an owned finite float64 array."""
    if len(shape) == 1:
        rows = [value]
    else:
        if type(value) is not list or len(value) != shape[0]:
            raise LinearMVNError(f"{name} has the wrong shape")
        rows = value
    flat = []
    for row in rows:
        if type(row) is not list or len(row) != shape[-1]:
            raise LinearMVNError(f"{name} has the wrong shape")
        flat.extend(row)
    array = np.empty(len(flat), dtype=np.float64)
    for index, item in enumerate(flat):
        if type(item) not in (int, float):
            raise LinearMVNError(f"{name} must hold only JSON numbers")
        try:
            array[index] = float(item)
        except OverflowError as exc:
            raise LinearMVNError(f"{name} holds an unrepresentable number") from exc
    if not np.all(np.isfinite(array)):
        raise LinearMVNError(f"{name} must be finite")
    return array.reshape(shape)


def _arrays_from_payload(payload):
    """Validate the v1 document structure and decode its ten arrays."""
    if type(payload) is not dict or set(payload) != _TOP_LEVEL_KEYS:
        raise LinearMVNError("capsule fields must match the v1 schema exactly")
    if payload["schema"] != SCHEMA:
        raise LinearMVNError("unsupported capsule schema version")
    if payload["inference_convention"] != INFERENCE_CONVENTION:
        raise LinearMVNError("unsupported inference convention")
    dimensions = payload["dimensions"]
    if type(dimensions) is not dict or set(dimensions) != _DIMENSION_NAMES:
        raise LinearMVNError("dimensions must hold exactly predictor, target and canonical")
    predictor = dimensions["predictor"]
    canonical = dimensions["canonical"]
    target = dimensions["target"]
    _check_dimensions(predictor, canonical, target)
    shapes = _array_shapes(predictor, canonical, target)
    return {name: _array_from_json(name, payload[name], shapes[name]) for name in _ARRAY_NAMES}


def _affine_steps(processor, role):
    """Fitted non-passthrough steps of an exact supported pre-processor."""
    if type(processor) is Pipeline:
        entries = list(processor.steps)
    elif type(processor) in _AFFINE_STEPS:
        entries = [("step", processor)]
    else:
        raise LinearMVNError(f"{role} must be an exact Pipeline, StandardScaler or PCA")
    steps = []
    for entry in entries:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise LinearMVNError(f"{role} has a malformed step")
        step = entry[1]
        if isinstance(step, str):
            if step != "passthrough":
                raise LinearMVNError(f"{role} has an unsupported string step")
            continue
        if type(step) not in _AFFINE_STEPS:
            raise LinearMVNError(f"{role} has an unsupported (non-affine) step")
        if type(step) is PCA and step.whiten is not False:
            raise LinearMVNError(f"{role} PCA must have whiten=False")
        try:
            check_is_fitted(step)
        except NotFittedError as exc:
            raise LinearMVNError(f"{role} has an unfitted step") from exc
        steps.append(step)
    return steps


def _require_passthrough(processor, role):
    """Post-processing must be an exact Pipeline of passthrough steps only."""
    if type(processor) is not Pipeline or len(processor.steps) == 0:
        raise LinearMVNError(f"{role} must be a passthrough Pipeline")
    for entry in processor.steps:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise LinearMVNError(f"{role} has a malformed step")
        if not isinstance(entry[1], str) or entry[1] != "passthrough":
            raise LinearMVNError(f"{role} must be passthrough")


def _check_profile(bel):
    """Reject every unsupported profile; return dimensions and copied state."""
    if type(bel) is not BEL:
        raise LinearMVNError("export requires an exact BEL instance (no subclasses)")
    if not isinstance(bel.mode, str) or bel.mode != "mvn":
        raise LinearMVNError("only mode='mvn' is supported")
    cached = (bel.x_observation, bel.x_pre_processed, bel.y_pre_processed)
    if any(item is not None for item in cached):
        raise LinearMVNError("cached observation or pre-processed overrides are unsupported")
    model = bel.regression_model
    if type(model) is not CCA:
        raise LinearMVNError("regression_model must be an exact CCA")
    try:
        check_is_fitted(model)
    except NotFittedError as exc:
        raise LinearMVNError("CCA is not fitted") from exc
    x_steps = _affine_steps(bel.X_pre_processing, "X_pre_processing")
    y_steps = _affine_steps(bel.Y_pre_processing, "Y_pre_processing")
    _require_passthrough(bel.X_post_processing, "X_post_processing")
    _require_passthrough(bel.Y_post_processing, "Y_post_processing")

    x_f = getattr(bel, "X_f", None)
    y_f = getattr(bel, "Y_f", None)
    if not isinstance(x_f, np.ndarray) or not isinstance(y_f, np.ndarray):
        raise LinearMVNError("BEL is not fitted (paired X_f and Y_f are missing)")
    if x_f.ndim != 2 or x_f.shape != y_f.shape or x_f.shape[0] < 2:
        raise LinearMVNError("X_f and Y_f must be paired 2D arrays with at least 2 rows")
    x_f = _as_real_array("X_f", x_f)
    y_f = _as_real_array("Y_f", y_f)
    canonical = x_f.shape[1]
    rotations = _as_real_array("x_rotations_", model.x_rotations_)
    if rotations.ndim != 2 or rotations.shape[1] != canonical:
        raise LinearMVNError("CCA rotations do not match the fitted canonical dimension")
    if model.n_components != canonical:
        raise LinearMVNError("CCA n_components does not match the fitted state")
    predictor = int(x_steps[0].n_features_in_) if x_steps else int(rotations.shape[0])
    if y_steps:
        target = int(y_steps[0].n_features_in_)
    else:
        target = int(np.shape(model.y_loadings_)[0])
    _check_dimensions(predictor, canonical, target)
    return predictor, canonical, target, x_f, y_f, rotations


def _paired_statistics(x_f, y_f):
    """Aggregate statistics with the exact conventions of ``mvn_inference``."""
    try:
        mu_y = np.mean(y_f, axis=0)
        mu_y = np.where(np.abs(mu_y) < _ZERO_THRESHOLD, 0, mu_y)
        c_y = np.cov(y_f.T)
        g = np.linalg.lstsq(y_f, x_f, rcond=None)[0].T
        g = np.where(np.abs(g) < _ZERO_THRESHOLD, 0, g)
        predicted = np.matmul(y_f, g.T)
        mu_e = np.mean(x_f - predicted, axis=0)
        residual = x_f - predicted - np.tile(mu_e, (x_f.shape[0], 1))
        c_e = np.cov(residual.T)
    except np.linalg.LinAlgError as exc:
        raise LinearMVNError("paired statistics could not be computed") from exc
    return mu_y, c_y, g, mu_e, c_e


def _forward_affine(bel, predictor, canonical):
    """Raw predictor -> canonical data map from one public ``transform`` call."""
    basis = np.vstack([np.zeros((1, predictor)), np.eye(predictor)])
    try:
        out = np.asarray(bel.transform(X=basis), dtype=np.float64)
    except (ValueError, TypeError, AttributeError) as exc:
        raise LinearMVNError("public transform failed on the affine basis") from exc
    if out.shape != (predictor + 1, canonical) or not np.all(np.isfinite(out)):
        raise LinearMVNError("public transform returned an unexpected basis image")
    offset = out[0].copy()
    return out[1:] - offset, offset


def _inverse_affine(bel, canonical, target):
    """Canonical -> original target map from one public ``inverse_transform`` call."""
    basis = np.vstack([np.zeros((1, canonical)), np.eye(canonical)])
    try:
        out = np.asarray(bel.inverse_transform(basis[np.newaxis]), dtype=np.float64)
    except (ValueError, TypeError, AttributeError) as exc:
        raise LinearMVNError("public inverse_transform failed on the affine basis") from exc
    if out.shape != (1, canonical + 1, target) or not np.all(np.isfinite(out)):
        raise LinearMVNError("public inverse_transform returned an unexpected basis image")
    offset = out[0, 0].copy()
    return out[0, 1:] - offset, offset


def export_linear_mvn(bel):
    """Compile a fitted ``BEL`` into a data-only :class:`LinearMVNCapsule`.

    Supported: exact ``BEL`` (no subclass), ``mode="mvn"``, exact fitted ``CCA``
    with at least two components, pre-processing built only from exact
    ``Pipeline``/``StandardScaler``/``PCA(whiten=False)`` (passthrough allowed),
    passthrough post-processing, no cached observation override, finite paired
    ``X_f``/``Y_f`` and positive definite, well conditioned aggregate blocks.
    Everything else raises :class:`LinearMVNError` before a capsule exists.

    The affine maps come from the public ``transform``/``inverse_transform`` on
    a zero-plus-basis array; the fitted object and its caches are not changed
    and no training rows, estimators or seeds are retained.
    """
    predictor, canonical, target, x_f, y_f, rotations = _check_profile(bel)
    mu_y, c_y, g, mu_e, c_e = _paired_statistics(x_f, y_f)
    gram = rotations.T @ rotations
    named = (("mu_y", mu_y), ("C_y", c_y), ("G", g), ("mu_e", mu_e), ("C_e", c_e), ("B", gram))
    for name, array in named:
        if not np.all(np.isfinite(array)):
            raise LinearMVNError(f"{name} is not finite")
    _check_moment_matrices(c_y, c_e, gram)
    a_x, b_x = _forward_affine(bel, predictor, canonical)
    a_y, b_y = _inverse_affine(bel, canonical, target)
    return LinearMVNCapsule(
        A_x=a_x,
        b_x=b_x,
        A_y=a_y,
        b_y=b_y,
        mu_y=mu_y,
        C_y=c_y,
        G=g,
        mu_e=mu_e,
        C_e=c_e,
        B=gram,
    )
