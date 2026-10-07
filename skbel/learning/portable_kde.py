"""Portable, data-only conditional prediction capsule for the affine KDE BEL profile.

``export_kde`` compiles a fitted ``BEL`` (mode ``"kde"``, two-component CCA,
affine scaler/PCA pre-processing, passthrough post-processing) and caller-frozen
bandwidths into a :class:`KDEPredictionCapsule`.  Per canonical component the
capsule holds either the fixed-bandwidth joint density table on the existing
200 x 200 grid or the existing near-linear branch (slope and intercept), plus
the affine raw-predictor and original-target maps.

For NEW raw predictor rows and caller-supplied uniforms the capsule conditions
each component with the existing cross-section, normalization and CDF
conventions and returns original-unit draws.  Restoration and sampling never
fit, unpickle, import or call anything from the bytes, and need no training
rows, estimators or callbacks.  The compiler itself does fit the fixed-bandwidth
component densities (and the near-linear regressions).

This is deliberately narrower than a trusted joblib checkpoint.  The default
``GridSearchCV`` bandwidth selection of ``BEL.predict`` is not reproduced, the
canonical components are sampled independently (the existing factorized
approximation, with original-unit correlation coming only from the affine
reconstruction) and uncertainty discarded by PCA/CCA truncation is not added.
Learned density tables stay sensitive; a SHA-256 digest only detects change
against an externally supplied expectation and is no authenticity, privacy or
sandboxing claim.
"""

import hashlib
import hmac
import json

import numpy as np
from scipy import integrate, ndimage
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from ..algorithms.statistics import kde_params, romb
from .bel import BEL

__all__ = [
    "CONVENTION",
    "MAX_BYTES",
    "SCHEMA",
    "KDEPredictionCapsule",
    "KDEPredictionError",
    "export_kde",
]

SCHEMA = "skbel.kde-prediction-capsule/v1"
CONVENTION = (
    "gaussian-euclidean-fixed-bw-grid200-cut1-count-pixels-cubic-constant0-"
    "prefilter-129-romberg-cutoffs-linear-cdf-inverse/v1"
)
MAX_BYTES = 4 * 1024 * 1024
MAX_DEPTH = 12
CANONICAL = 2
MAX_FEATURES = 4
MAX_TRAINING_ROWS = 32
GRID_SIZE = 200
LINE_SIZE = 2**7 + 1
LINEAR_CORRELATION = 0.999

_DENSITY_CUTOFF = 1e-8
_NORMALIZATION_CUTOFF = 1e-3
_PREFIX_MIN_DISTANCE = 1e-4
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_DIMENSION_NAMES = frozenset(("predictor", "canonical", "target"))
_MAP_NAMES = ("A_x", "b_x", "A_y", "b_y")
_TOP_LEVEL_KEYS = frozenset(("schema", "convention", "dimensions", "components", *_MAP_NAMES))
_LINEAR_KEYS = frozenset(("kind", "slope", "intercept"))
_PDF_KEYS = frozenset(("kind", "bandwidth", "x_axis", "y_axis", "density"))
_AFFINE_STEPS = (StandardScaler, PCA)


class KDEPredictionError(ValueError):
    """Unsupported profile, malformed wire data, inconsistent state or invalid query law."""


def _check_dimensions(predictor, canonical, target):
    """Validate the (P, Q, R) dimension triple: Q == 2 and Q <= P, R <= 4."""
    for name, value in (
        ("predictor", predictor),
        ("canonical", canonical),
        ("target", target),
    ):
        if type(value) is not int:
            raise KDEPredictionError(f"dimension {name} must be an integer")
    if canonical != CANONICAL:
        raise KDEPredictionError(f"canonical dimension must be {CANONICAL}")
    for name, value in (("predictor", predictor), ("target", target)):
        if not CANONICAL <= value <= MAX_FEATURES:
            raise KDEPredictionError(
                f"{name} dimension must satisfy {CANONICAL} <= {name} <= {MAX_FEATURES}"
            )


def _reject_booleans(name, value):
    """Reject Boolean items in (nested) lists and tuples before NumPy can coerce them."""
    if isinstance(value, (bool, np.bool_)):
        raise KDEPredictionError(f"{name} must hold real numbers (no bool, object or complex)")
    if isinstance(value, (list, tuple)):
        for item in value:
            _reject_booleans(name, item)


def _as_real_array(name, value):
    """Return an owned, finite float64 copy of a real numeric array-like."""
    if isinstance(value, np.ma.MaskedArray):
        raise KDEPredictionError(f"{name} must not be a masked array")
    try:
        _reject_booleans(name, value)
    except RecursionError as exc:
        raise KDEPredictionError(f"{name} is nested too deeply") from exc
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise KDEPredictionError(f"{name} must be a real numeric array") from exc
    if array.dtype.kind not in "iuf":
        raise KDEPredictionError(f"{name} must hold real numbers (no bool, object or complex)")
    owned = np.array(array, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(owned)):
        raise KDEPredictionError(f"{name} must be finite")
    return owned


def _is_real_scalar(value):
    """Python or NumPy integer/float scalar; Booleans are not numbers here."""
    if isinstance(value, (bool, np.bool_)):
        return False
    return type(value) in (int, float) or isinstance(value, (np.integer, np.floating))


def _state_array(name, value, shape):
    """Owned finite float64 array of exactly ``shape`` from an ndarray or nested lists.

    Nested lists are walked level by level (no recursion, no NumPy coercion), so
    ragged, Boolean, string, null or over-nested entries are rejected.
    """
    if isinstance(value, np.ndarray):
        if isinstance(value, np.ma.MaskedArray):
            raise KDEPredictionError(f"{name} must not be a masked array")
        if value.dtype.kind not in "iuf" or value.shape != shape:
            raise KDEPredictionError(f"{name} must be a real array of shape {shape}")
        array = np.array(value, dtype=np.float64, copy=True)
    else:
        level = [value]
        for size in shape:
            items = []
            for item in level:
                if type(item) not in (list, tuple) or len(item) != size:
                    raise KDEPredictionError(f"{name} must have shape {shape}")
                items.extend(item)
            level = items
        array = np.empty(len(level), dtype=np.float64)
        for index, item in enumerate(level):
            if not _is_real_scalar(item):
                raise KDEPredictionError(f"{name} must hold only real numbers")
            try:
                array[index] = float(item)
            except OverflowError as exc:
                raise KDEPredictionError(f"{name} holds an unrepresentable number") from exc
        array = array.reshape(shape)
    if not np.all(np.isfinite(array)):
        raise KDEPredictionError(f"{name} must be finite")
    return array


def _state_scalar(name, value):
    """Finite float from a real scalar (or 0-d real array)."""
    return float(_state_array(name, value, ())[()])


def _check_axis(name, axis):
    """Axes must be the strictly increasing ``linspace(first, last, 200)`` grid."""
    if not np.all(np.diff(axis) > 0):
        raise KDEPredictionError(f"{name} must be strictly increasing")
    if not np.array_equal(axis, np.linspace(axis[0], axis[-1], GRID_SIZE)):
        raise KDEPredictionError(f"{name} must be an evenly spaced {GRID_SIZE}-point grid")


def _check_density(name, density):
    """Non-negative table, cut off below 1e-8 as compiled, with some positive mass."""
    if np.any(density < 0):
        raise KDEPredictionError(f"{name} must be non-negative")
    if np.any((density > 0) & (density < _DENSITY_CUTOFF)):
        raise KDEPredictionError(f"{name} holds values below the {_DENSITY_CUTOFF} cutoff")
    if not np.any(density > 0):
        raise KDEPredictionError(f"{name} must hold some positive density")


def _validate_component(index, value):
    """Validate and copy one ``linear`` or ``pdf`` component record."""
    label = f"components[{index}]"
    if type(value) is not dict or type(value.get("kind")) is not str:
        raise KDEPredictionError(f"{label} must be an object with a string kind")
    kind = value["kind"]
    if kind == "linear":
        if set(value) != _LINEAR_KEYS:
            raise KDEPredictionError(f"{label} fields must be exactly kind, slope, intercept")
        return {
            "kind": "linear",
            "slope": _state_scalar(f"{label}.slope", value["slope"]),
            "intercept": _state_scalar(f"{label}.intercept", value["intercept"]),
        }
    if kind == "pdf":
        if set(value) != _PDF_KEYS:
            raise KDEPredictionError(
                f"{label} fields must be exactly kind, bandwidth, x_axis, y_axis, density"
            )
        bandwidth = _state_scalar(f"{label}.bandwidth", value["bandwidth"])
        if not bandwidth > 0:
            raise KDEPredictionError(f"{label}.bandwidth must be positive")
        x_axis = _state_array(f"{label}.x_axis", value["x_axis"], (GRID_SIZE,))
        y_axis = _state_array(f"{label}.y_axis", value["y_axis"], (GRID_SIZE,))
        density = _state_array(f"{label}.density", value["density"], (GRID_SIZE, GRID_SIZE))
        _check_axis(f"{label}.x_axis", x_axis)
        _check_axis(f"{label}.y_axis", y_axis)
        _check_density(f"{label}.density", density)
        return {
            "kind": "pdf",
            "bandwidth": bandwidth,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "density": density,
        }
    raise KDEPredictionError(f"{label} has an unsupported kind")


def _validate_state(state):
    """Validate the full v1 state; return dimensions, copied maps and components."""
    if type(state) is not dict or set(state) != _TOP_LEVEL_KEYS:
        raise KDEPredictionError("capsule fields must match the v1 schema exactly")
    if type(state["schema"]) is not str or state["schema"] != SCHEMA:
        raise KDEPredictionError("unsupported capsule schema version")
    if type(state["convention"]) is not str or state["convention"] != CONVENTION:
        raise KDEPredictionError("unsupported capsule convention")
    dimensions = state["dimensions"]
    if type(dimensions) is not dict or set(dimensions) != _DIMENSION_NAMES:
        raise KDEPredictionError("dimensions must hold exactly predictor, canonical and target")
    predictor = dimensions["predictor"]
    canonical = dimensions["canonical"]
    target = dimensions["target"]
    _check_dimensions(predictor, canonical, target)
    shapes = {
        "A_x": (predictor, canonical),
        "b_x": (canonical,),
        "A_y": (canonical, target),
        "b_y": (target,),
    }
    maps = {name: _state_array(name, state[name], shapes[name]) for name in _MAP_NAMES}
    components = state["components"]
    if type(components) not in (list, tuple) or len(components) != canonical:
        raise KDEPredictionError(f"components must be a list of {canonical} records")
    parsed = [_validate_component(index, item) for index, item in enumerate(components)]
    return (predictor, canonical, target), maps, parsed


def _check_observations(X_obs, predictor):
    """Validate and copy query rows: finite real 2D ``(n_cases >= 1, predictor)``."""
    observed = _as_real_array("X_obs", X_obs)
    if observed.ndim != 2 or observed.shape[0] < 1 or observed.shape[1] != predictor:
        raise KDEPredictionError(f"X_obs must have shape (n_cases >= 1, {predictor})")
    return observed


def _check_uniforms(U, n_cases, canonical):
    """Validate and copy uniforms: finite ``(n_cases, n_samples >= 1, Q)`` in [0, 1]."""
    uniforms = _as_real_array("U", U)
    if (
        uniforms.ndim != 3
        or uniforms.shape[0] != n_cases
        or uniforms.shape[1] < 1
        or uniforms.shape[2] != canonical
    ):
        raise KDEPredictionError(
            f"U must have shape ({n_cases}, n_samples >= 1, {canonical}) matching X_obs"
        )
    if np.any((uniforms < 0) | (uniforms > 1)):
        raise KDEPredictionError("U must lie in [0, 1]")
    return uniforms


def _resolve_obs_index(obs_n, n_obs):
    """Row index contract of ``BEL.random_sample``: negative values count from the end."""
    if isinstance(obs_n, (bool, np.bool_)) or not isinstance(obs_n, (int, np.integer)):
        raise KDEPredictionError(f"obs_n must be an integer index, got {obs_n!r}")
    if not -n_obs <= obs_n < n_obs:
        raise KDEPredictionError(f"obs_n={obs_n} is out of range for {n_obs} observations")
    return int(obs_n) % n_obs


def _normalized_density(points, line, post, normalization):
    """``normalize``: linear interpolation divided by A, values below 1e-3 set to 0."""
    ratio = np.interp(points, line, post) / normalization
    return np.where(ratio < _NORMALIZATION_CUTOFF, 0.0, ratio)


def _conditional_law(component, value):
    """Support and tabulated CDF of the existing conditional cross-section at ``value``.

    Mirrors ``posterior_conditional`` (pixel coordinates scaled by the pixel count,
    cubic ``map_coordinates`` with constant-0 boundary and prefilter, 1e-8 cutoffs,
    absolute-Simpson scaling), the second 1e-8 cutoff of ``BEL.predict`` and the
    ``normalize``/``get_cdf`` Romberg CDF on 129 points.  Anything outside the
    portable profile fails instead of being repaired or sampled as zeros.
    """
    x_axis, y_axis, density = component["x_axis"], component["y_axis"], component["density"]
    if not x_axis[0] <= value <= x_axis[-1]:
        raise KDEPredictionError("canonical query lies outside the recorded density support")
    x_world = np.array([value, value])
    y_world = np.array([y_axis.min(), y_axis.max()])
    col = GRID_SIZE * (x_world - x_axis.min()) / np.ptp(x_axis)
    row = GRID_SIZE * (y_world - y_axis.min()) / np.ptp(y_axis)
    row = np.linspace(row[0], row[1], LINE_SIZE)
    col = np.linspace(col[0], col[1], LINE_SIZE)
    post = ndimage.map_coordinates(
        density, np.vstack((row, col)), order=3, mode="constant", cval=0.0, prefilter=True
    )
    line = np.linspace(y_axis.min(), y_axis.max(), LINE_SIZE)
    if not np.all(np.diff(line) > 0):
        raise KDEPredictionError("conditional support is degenerate")

    post[np.abs(post) < _DENSITY_CUTOFF] = 0
    if not post.any():
        raise KDEPredictionError("conditional density is zero at this query")
    area = integrate.simpson(y=np.abs(post), x=line)
    if not (np.isfinite(area) and area > 0):
        raise KDEPredictionError("conditional density cannot be scaled")
    post *= 1 / area
    post[np.abs(post) < _DENSITY_CUTOFF] = 0
    if not np.all(np.isfinite(post)) or np.any(post < 0):
        raise KDEPredictionError("conditional density is negative or not finite")

    normalization = romb(post, np.abs(line[1] - line[0]))
    if not (np.isfinite(normalization) and normalization >= _NORMALIZATION_CUTOFF):
        raise KDEPredictionError("conditional density normalization is below the 1e-3 cutoff")
    lower, upper = line.min(), line.max()
    if not np.any(_normalized_density(line, line, post, normalization) > 0):
        raise KDEPredictionError("normalized conditional density is zero everywhere")
    cdf = np.empty(LINE_SIZE, dtype=np.float64)
    for index, point in enumerate(np.linspace(lower, upper, LINE_SIZE)):
        if point <= lower:
            cdf[index] = 0.0
        elif point >= upper:
            cdf[index] = 1.0
        elif np.abs(point - lower) > _PREFIX_MIN_DISTANCE:
            samples = np.linspace(lower, point, LINE_SIZE)
            step = np.abs(samples[1] - samples[0])
            cdf[index] = romb(_normalized_density(samples, line, post, normalization), step)
        else:
            cdf[index] = 0.0
    if not np.all(np.isfinite(cdf)) or np.any((cdf < 0) | (cdf > 1)):
        raise KDEPredictionError("conditional CDF leaves [0, 1]")
    if np.any(np.diff(cdf) < 0):
        raise KDEPredictionError("conditional CDF decreases")
    return line, cdf


def _inverse_cdf(cdf, support, uniforms):
    """Quantiles ``np.interp(U, cdf, support)`` of a validated tabulated CDF.

    ``cdf`` must be finite, non-decreasing, within [0, 1], start at 0 and end at 1
    on a strictly increasing ``support`` of the same length.  At knots and on CDF
    plateaus NumPy returns the rightmost tied knot, exactly as the existing sampler.
    """
    cdf = _as_real_array("cdf", cdf)
    support = _as_real_array("support", support)
    uniforms = _as_real_array("U", uniforms)
    if cdf.ndim != 1 or cdf.shape[0] < 2 or support.shape != cdf.shape or uniforms.ndim != 1:
        raise KDEPredictionError("cdf and support must be equal-length 1D arrays; U must be 1D")
    if not np.all(np.diff(support) > 0):
        raise KDEPredictionError("support must be strictly increasing")
    if cdf[0] != 0.0 or cdf[-1] != 1.0 or np.any((cdf < 0) | (cdf > 1)):
        raise KDEPredictionError("cdf must lie in [0, 1] and run from 0 to 1")
    if np.any(np.diff(cdf) < 0):
        raise KDEPredictionError("cdf must be non-decreasing")
    if np.any((uniforms < 0) | (uniforms > 1)):
        raise KDEPredictionError("U must lie in [0, 1]")
    return np.interp(uniforms, cdf, support)


class KDEPredictionCapsule:
    """Prediction-only, data-only state of an affine KDE BEL model.

    Raw predictor rows ``x`` map to canonical data ``d = x A_x + b_x``; canonical
    targets ``z`` reconstruct to ``y = z A_y + b_y``.  Component ``j`` is either
    ``{"kind": "linear", "slope", "intercept"}`` (a point mass at
    ``slope * d_j + intercept``) or ``{"kind": "pdf", "bandwidth", "x_axis",
    "y_axis", "density"}`` with ``density[row, col]`` evaluated at
    ``(x_axis[col], y_axis[row])``.

    The constructor validates numeric structure only; every query still validates
    its own conditional law.  All state is copied and stored read-only.
    """

    def __init__(self, state):
        """Validate and copy a v1 state mapping (the decoded wire document)."""
        (predictor, canonical, target), maps, components = _validate_state(state)
        for array in maps.values():
            array.flags.writeable = False
        for component in components:
            if component["kind"] == "pdf":
                for name in ("x_axis", "y_axis", "density"):
                    component[name].flags.writeable = False
        self._predictor = predictor
        self._canonical = canonical
        self._target = target
        self._maps = maps
        self._components = components
        self._bytes = self._encode()
        if len(self._bytes) > MAX_BYTES:
            raise KDEPredictionError(f"capsule data exceeds {MAX_BYTES} bytes")

    @classmethod
    def from_state(cls, state):
        """Build a capsule from a state mapping with the exact wire keys."""
        return cls(state)

    def __repr__(self):
        return (
            f"KDEPredictionCapsule(predictor={self._predictor}, "
            f"canonical={self._canonical}, target={self._target}, kinds={self.kinds})"
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

    @property
    def kinds(self):
        """Branch of each canonical component, ``"linear"`` or ``"pdf"``."""
        return tuple(component["kind"] for component in self._components)

    def state(self):
        """Independent writable copy of the state (accepted by ``from_state``)."""
        components = []
        for component in self._components:
            copied = {}
            for name, value in component.items():
                copied[name] = value.copy() if isinstance(value, np.ndarray) else value
            components.append(copied)
        state = {
            "schema": SCHEMA,
            "convention": CONVENTION,
            "dimensions": {
                "predictor": self._predictor,
                "canonical": self._canonical,
                "target": self._target,
            },
            "components": components,
        }
        state.update({name: self._maps[name].copy() for name in _MAP_NAMES})
        return state

    def sample(self, X_obs, U, obs_n=None):
        """Original-unit conditional draws for new raw predictor rows.

        :param X_obs: Finite real array ``(n_cases, P)``; no broadcasting.
        :param U: Finite uniforms in [0, 1] of shape ``(n_cases, n_samples, Q)``,
            one channel per canonical component.  Channels of linear components are
            validated but not used.  The capsule never draws random numbers.
        :param obs_n: ``None`` for every row, or an integer row index (negative
            values count from the end) selecting the same row of ``X_obs`` and ``U``.
        :return: ``(n_cases, n_samples, R)`` (or ``(1, n_samples, R)`` when
            ``obs_n`` is given), freshly allocated and owned by the caller.
        """
        observed = _check_observations(X_obs, self._predictor)
        uniforms = _check_uniforms(U, observed.shape[0], self._canonical)
        canonical_data = observed @ self._maps["A_x"] + self._maps["b_x"]
        if obs_n is not None:
            index = _resolve_obs_index(obs_n, observed.shape[0])
            canonical_data = canonical_data[index : index + 1]
            uniforms = uniforms[index : index + 1]
        n_cases, n_samples = uniforms.shape[:2]
        draws = np.empty((n_cases, n_samples, self._canonical), dtype=np.float64)
        for i in range(n_cases):
            for j, component in enumerate(self._components):
                value = canonical_data[i, j]
                if component["kind"] == "linear":
                    draws[i, :, j] = value * component["slope"] + component["intercept"]
                else:
                    support, cdf = _conditional_law(component, value)
                    draws[i, :, j] = _inverse_cdf(cdf, support, uniforms[i, :, j])
        samples = draws @ self._maps["A_y"] + self._maps["b_y"]
        if not np.all(np.isfinite(samples)):
            raise KDEPredictionError("samples are not representable")
        return samples

    def _encode(self):
        """Deterministic compact JSON of the validated state."""
        components = []
        for component in self._components:
            if component["kind"] == "linear":
                components.append(
                    {
                        "kind": "linear",
                        "slope": component["slope"],
                        "intercept": component["intercept"],
                    }
                )
            else:
                components.append(
                    {
                        "kind": "pdf",
                        "bandwidth": component["bandwidth"],
                        "x_axis": component["x_axis"].tolist(),
                        "y_axis": component["y_axis"].tolist(),
                        "density": component["density"].tolist(),
                    }
                )
        payload = {
            "schema": SCHEMA,
            "convention": CONVENTION,
            "dimensions": {
                "predictor": self._predictor,
                "canonical": self._canonical,
                "target": self._target,
            },
            "components": components,
        }
        payload.update({name: self._maps[name].tolist() for name in _MAP_NAMES})
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return text.encode("utf-8")

    def to_bytes(self):
        """Deterministic UTF-8 JSON (schema ``skbel.kde-prediction-capsule/v1``)."""
        return self._bytes

    def sha256(self):
        """Hex SHA-256 of ``to_bytes()``; integrity against an external copy only."""
        return hashlib.sha256(self._bytes).hexdigest()

    @classmethod
    def from_bytes(cls, data, expected_sha256=None):
        """Restore a capsule from ``to_bytes`` output, validating everything.

        The size limit is checked first, then the optional external digest, and
        only then is the document parsed (nesting depth at most 12) and validated
        exactly like ``from_state``.  A matching digest is not proof of
        authenticity: whoever can alter the bytes can also alter the digest.

        :param data: ``bytes`` of at most ``MAX_BYTES``.
        :param expected_sha256: Optional hex digest obtained out of band.
        """
        if type(data) is not bytes:
            raise KDEPredictionError("capsule data must be bytes")
        if len(data) > MAX_BYTES:
            raise KDEPredictionError(f"capsule data exceeds {MAX_BYTES} bytes")
        if expected_sha256 is not None:
            _check_digest(data, expected_sha256)
        payload = _parse_json(data)
        if _depth(payload) > MAX_DEPTH:
            raise KDEPredictionError("capsule data is nested too deeply")
        return cls(payload)


def _check_digest(data, expected):
    """Compare the SHA-256 of ``data`` with an externally supplied hex digest."""
    if type(expected) is not str or len(expected) != 64 or not set(expected) <= _HEX_DIGITS:
        raise KDEPredictionError("expected_sha256 must be a 64 character hex string")
    if not hmac.compare_digest(hashlib.sha256(data).hexdigest(), expected.lower()):
        raise KDEPredictionError("SHA-256 digest does not match the expected value")


def _strict_pairs(pairs):
    """JSON object hook that rejects duplicate keys."""
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise KDEPredictionError("duplicate keys are not allowed")
    return dict(pairs)


def _reject_constant(token):
    """JSON constant hook: NaN and the infinities are not valid capsule data."""
    raise KDEPredictionError(f"non-finite constant {token} is not allowed")


def _parse_json(data):
    """Decode UTF-8 and parse JSON with strict hooks; nothing is executed."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise KDEPredictionError("capsule data is not valid UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except KDEPredictionError:
        raise
    except (ValueError, RecursionError) as exc:
        raise KDEPredictionError("capsule data is not valid JSON") from exc


def _depth(value):
    """Container nesting depth of a parsed document (iterative)."""
    if not isinstance(value, (dict, list)):
        return 0
    deepest = 0
    stack = [(value, 1)]
    while stack:
        item, level = stack.pop()
        deepest = max(deepest, level)
        if deepest > MAX_DEPTH:
            break
        children = item.values() if isinstance(item, dict) else item
        stack.extend((child, level + 1) for child in children if isinstance(child, (dict, list)))
    return deepest


def _affine_steps(processor, role):
    """Fitted non-passthrough steps of an exact supported pre-processor."""
    if type(processor) is Pipeline:
        entries = list(processor.steps)
    elif type(processor) in _AFFINE_STEPS:
        entries = [("step", processor)]
    else:
        raise KDEPredictionError(f"{role} must be an exact Pipeline, StandardScaler or PCA")
    steps = []
    for entry in entries:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise KDEPredictionError(f"{role} has a malformed step")
        step = entry[1]
        if isinstance(step, str):
            if step != "passthrough":
                raise KDEPredictionError(f"{role} has an unsupported string step")
            continue
        if type(step) not in _AFFINE_STEPS:
            raise KDEPredictionError(f"{role} has an unsupported (non-affine) step")
        if type(step) is PCA and step.whiten is not False:
            raise KDEPredictionError(f"{role} PCA must have whiten=False")
        try:
            check_is_fitted(step)
        except NotFittedError as exc:
            raise KDEPredictionError(f"{role} has an unfitted step") from exc
        steps.append(step)
    return steps


def _require_passthrough(processor, role):
    """Post-processing must be an exact Pipeline of passthrough steps only."""
    if type(processor) is not Pipeline or len(processor.steps) == 0:
        raise KDEPredictionError(f"{role} must be a passthrough Pipeline")
    for entry in processor.steps:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise KDEPredictionError(f"{role} has a malformed step")
        if not isinstance(entry[1], str) or entry[1] != "passthrough":
            raise KDEPredictionError(f"{role} must be passthrough")


def _check_profile(bel):
    """Reject every unsupported profile; return dimensions and copied paired scores."""
    if type(bel) is not BEL:
        raise KDEPredictionError("export requires an exact BEL instance (no subclasses)")
    if not isinstance(bel.mode, str) or bel.mode != "kde":
        raise KDEPredictionError("only mode='kde' is supported")
    cached = (bel.x_observation, bel.x_pre_processed, bel.y_pre_processed)
    if any(item is not None for item in cached):
        raise KDEPredictionError("cached observation or pre-processed overrides are unsupported")
    model = bel.regression_model
    if type(model) is not CCA:
        raise KDEPredictionError("regression_model must be an exact CCA")
    try:
        check_is_fitted(model)
    except NotFittedError as exc:
        raise KDEPredictionError("CCA is not fitted") from exc
    x_steps = _affine_steps(bel.X_pre_processing, "X_pre_processing")
    y_steps = _affine_steps(bel.Y_pre_processing, "Y_pre_processing")
    _require_passthrough(bel.X_post_processing, "X_post_processing")
    _require_passthrough(bel.Y_post_processing, "Y_post_processing")

    x_f = getattr(bel, "X_f", None)
    y_f = getattr(bel, "Y_f", None)
    if not isinstance(x_f, np.ndarray) or not isinstance(y_f, np.ndarray):
        raise KDEPredictionError("BEL is not fitted (paired X_f and Y_f are missing)")
    if x_f.ndim != 2 or x_f.shape != y_f.shape:
        raise KDEPredictionError("X_f and Y_f must be paired 2D arrays")
    if not 2 <= x_f.shape[0] <= MAX_TRAINING_ROWS:
        raise KDEPredictionError(f"paired training rows must number 2 to {MAX_TRAINING_ROWS}")
    x_f = _as_real_array("X_f", x_f)
    y_f = _as_real_array("Y_f", y_f)
    canonical = x_f.shape[1]
    if canonical != CANONICAL or model.n_components != canonical:
        raise KDEPredictionError(f"exactly {CANONICAL} fitted CCA components are required")
    rotations = _as_real_array("x_rotations_", model.x_rotations_)
    if rotations.ndim != 2 or rotations.shape[1] != canonical:
        raise KDEPredictionError("CCA rotations do not match the fitted canonical dimension")
    predictor = int(x_steps[0].n_features_in_) if x_steps else int(rotations.shape[0])
    if y_steps:
        target = int(y_steps[0].n_features_in_)
    else:
        target = int(np.shape(model.y_loadings_)[0])
    _check_dimensions(predictor, canonical, target)
    return predictor, canonical, target, x_f, y_f


def _check_bandwidths(bandwidths, canonical):
    """One finite positive bandwidth per component (linear components ignore theirs)."""
    if isinstance(bandwidths, (str, bytes)):
        raise KDEPredictionError("bandwidths must be a sequence of real numbers")
    widths = _as_real_array("bandwidths", bandwidths)
    if widths.shape != (canonical,) or not np.all(widths > 0):
        raise KDEPredictionError(f"bandwidths must be {canonical} finite positive numbers")
    return widths


def _compile_component(x_scores, y_scores, bandwidth):
    """Existing ``BEL.predict`` branch choice and fixed-bandwidth density table.

    Correlation >= 0.999 (signed, not absolute) selects the linear branch with the
    fitted ``LinearRegression`` slope and intercept; otherwise one Gaussian
    ``KernelDensity`` fit with the given bandwidth is evaluated on the existing
    200 x 200 grid (cut 1, no clip) and values below 1e-8 are set to 0.
    """
    correlation = np.corrcoef(x_scores, y_scores).diagonal(offset=1)[0]
    if not np.isfinite(correlation):
        raise KDEPredictionError("canonical correlation is not finite")
    if correlation >= LINEAR_CORRELATION:
        fitted = LinearRegression().fit(x_scores.reshape(-1, 1), y_scores.reshape(-1, 1))
        return {
            "kind": "linear",
            "slope": float(np.asarray(fitted.coef_).reshape(-1)[0]),
            "intercept": float(np.asarray(fitted.intercept_).reshape(-1)[0]),
        }
    density, support, used = kde_params(x=x_scores, y=y_scores, bw=float(bandwidth))
    if used != bandwidth:
        raise KDEPredictionError("the density estimator did not keep the fixed bandwidth")
    density = np.array(density, dtype=np.float64, copy=True)
    density[density < _DENSITY_CUTOFF] = 0
    return {
        "kind": "pdf",
        "bandwidth": float(bandwidth),
        "x_axis": np.array(support[0], dtype=np.float64, copy=True),
        "y_axis": np.array(support[1], dtype=np.float64, copy=True),
        "density": density,
    }


def _forward_affine(bel, predictor, canonical):
    """Raw predictor -> canonical data map from one public ``transform`` call."""
    basis = np.vstack([np.zeros((1, predictor)), np.eye(predictor)])
    try:
        out = np.asarray(bel.transform(X=basis), dtype=np.float64)
    except (ValueError, TypeError, AttributeError) as exc:
        raise KDEPredictionError("public transform failed on the affine basis") from exc
    if out.shape != (predictor + 1, canonical) or not np.all(np.isfinite(out)):
        raise KDEPredictionError("public transform returned an unexpected basis image")
    offset = out[0].copy()
    return out[1:] - offset, offset


def _inverse_affine(bel, canonical, target):
    """Canonical -> original target map from one public ``inverse_transform`` call."""
    basis = np.vstack([np.zeros((1, canonical)), np.eye(canonical)])
    try:
        out = np.asarray(bel.inverse_transform(basis[np.newaxis]), dtype=np.float64)
    except (ValueError, TypeError, AttributeError) as exc:
        raise KDEPredictionError("public inverse_transform failed on the affine basis") from exc
    if out.shape != (1, canonical + 1, target) or not np.all(np.isfinite(out)):
        raise KDEPredictionError("public inverse_transform returned an unexpected basis image")
    offset = out[0, 0].copy()
    return out[0, 1:] - offset, offset


def export_kde(bel, bandwidths):
    """Compile a fitted ``BEL`` into a data-only :class:`KDEPredictionCapsule`.

    Supported: exact ``BEL`` (no subclass), ``mode="kde"``, exact fitted ``CCA``
    with exactly two components, pre-processing built only from exact
    ``Pipeline``/``StandardScaler``/``PCA(whiten=False)`` (passthrough allowed),
    passthrough post-processing, no cached observation override, finite paired
    ``X_f``/``Y_f`` with 2 to 32 rows and raw predictor/target widths 2 to 4.
    Everything else raises :class:`KDEPredictionError` before any fit.

    :param bel: The fitted model; its arrays, processors and caches are not changed.
    :param bandwidths: One finite positive bandwidth per canonical component, used
        as given (no bandwidth search).  Linear components ignore theirs.
    """
    predictor, canonical, target, x_f, y_f = _check_profile(bel)
    widths = _check_bandwidths(bandwidths, canonical)
    components = [
        _compile_component(x_f.T[j], y_f.T[j], float(widths[j])) for j in range(canonical)
    ]
    a_x, b_x = _forward_affine(bel, predictor, canonical)
    a_y, b_y = _inverse_affine(bel, canonical, target)
    return KDEPredictionCapsule(
        {
            "schema": SCHEMA,
            "convention": CONVENTION,
            "dimensions": {"predictor": predictor, "canonical": canonical, "target": target},
            "A_x": a_x,
            "b_x": b_x,
            "A_y": a_y,
            "b_y": b_y,
            "components": components,
        }
    )
