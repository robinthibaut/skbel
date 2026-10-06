"""Tests for the portable data-only linear-MVN capsule (``skbel.learning.portable``).

Finite call budget (one shared literal fixture, no random draws):

* BEL fits: 1.
* Original ``BEL.predict`` calls: 3 (default noise, explicit noise, trusted joblib copy).
* Capsule ``predict_moments`` calls: 8 (default, explicit, repeat after mutating the
  returned arrays, restored copy, rational oracle, nested fresh interpreter, one invalid
  shape, one mixed Boolean).
* ``export_linear_mvn`` calls: 9 (1 valid, 8 invalid profiles that fail before any basis call).
* Public basis calls (``transform``/``inverse_transform``): 5 (2 inside the one valid export,
  3 in the reference mapping).

``TestZCallBudget`` measures the in-process counts and checks them against the caps.
"""

import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from functools import cache, wraps
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

import skbel
from skbel import BEL
from skbel.learning import portable
from skbel.learning.portable import LinearMVNCapsule, LinearMVNError

ATOL = 1e-10
NOISE = 0.5

X_TRAIN = np.array(
    [
        [-1.38, 0.71, -0.63, -3.21],
        [-1.28, -0.49, -1.92, 0.05],
        [-0.85, 1.16, 0.40, -3.22],
        [-0.84, -0.94, -1.67, 1.21],
        [-0.31, 0.17, -0.16, -0.88],
        [-0.17, 1.58, 1.21, -3.24],
        [-0.11, -0.30, -0.27, 0.73],
        [0.37, 0.83, 1.25, -1.40],
        [0.45, -1.25, -0.88, 3.13],
        [0.93, 0.09, 1.11, 0.27],
        [0.91, -0.08, 0.76, 1.46],
        [1.34, 1.07, 2.30, -0.81],
        [1.48, -0.71, 0.86, 3.15],
        [1.98, 0.40, 2.42, 0.82],
        [0.0, -1.46, -1.73, 3.01],
        [0.54, 1.35, 2.09, -2.22],
    ]
)
Y_TRAIN = np.array(
    [
        [-3.49, -1.36, -1.33],
        [-2.02, -1.16, -0.30],
        [-2.82, -0.49, -1.30],
        [-0.75, -0.97, 0.48],
        [-0.83, -0.56, -0.25],
        [-2.04, 0.71, -1.98],
        [0.73, -0.48, 0.54],
        [-0.59, 1.12, -0.99],
        [2.51, -0.27, 1.87],
        [1.22, 1.19, 0.0],
        [2.56, 0.57, 0.98],
        [1.25, 2.0, -0.58],
        [4.12, 0.99, 1.96],
        [2.99, 2.42, 0.18],
        [1.54, -1.20, 1.64],
        [-0.37, 1.48, -1.43],
    ]
)
QUERY = np.array([[0.5, 0.3, 0.8, -0.2], [-0.7, 1.0, 0.4, -2.1]])

# Exactly representable (dyadic) payload used by the independent rational oracle.
RATIONAL = {
    "A_x": [[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]],
    "b_x": [0.25, -0.5],
    "A_y": [[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]],
    "b_y": [0.5, -0.25, 1.0],
    "mu_y": [0.25, -0.5],
    "C_y": [[2.0, 0.5], [0.5, 1.0]],
    "G": [[0.5, 0.25], [-0.25, 1.0]],
    "mu_e": [0.125, -0.0625],
    "C_e": [[1.0, 0.25], [0.25, 0.5]],
    "B": [[1.0, 0.5], [0.5, 2.0]],
}
RATIONAL_QUERY = [[1.0, 2.0, -1.0], [0.5, 0.0, 2.0]]
RATIONAL_NOISE = 0.25
DIMS = {"predictor": 3, "target": 3, "canonical": 2}

CALL_LIMITS = {"fit": 1, "predict": 4, "capsule_predict": 8, "export": 10, "basis": 20}
_CALLS = {"fit": 0, "predict": 0, "capsule_predict": 0, "export": 0, "basis": 0}
_PATCHED = []

_CHILD = """
import json
import sys

import numpy as np
from sklearn.cross_decomposition import CCA

from skbel import BEL
from skbel.learning.portable import LinearMVNCapsule


def _forbidden(*args, **kwargs):
    raise RuntimeError("fit or predict called in the restoring interpreter")


BEL.fit = BEL.predict = _forbidden
CCA.fit = _forbidden
data = sys.stdin.buffer.read()
capsule = LinearMVNCapsule.from_bytes(data, expected_sha256=sys.argv[1])
mean, cov = capsule.predict_moments(np.array(json.loads(sys.argv[2])), noise=float(sys.argv[3]))
print(json.dumps({"mean": mean.tolist(), "cov": cov.tolist()}))
"""


class _SubBEL(BEL):
    """Exact-type profile checks must reject subclasses."""


def _count(owner, name, key):
    original = getattr(owner, name)

    @wraps(original)
    def counted(*args, **kwargs):
        _CALLS[key] += 1
        return original(*args, **kwargs)

    setattr(owner, name, counted)
    _PATCHED.append((owner, name, original))


def setUpModule():
    _count(BEL, "fit", "fit")
    _count(BEL, "predict", "predict")
    _count(BEL, "transform", "basis")
    _count(BEL, "inverse_transform", "basis")
    _count(portable, "export_linear_mvn", "export")
    _count(LinearMVNCapsule, "predict_moments", "capsule_predict")


def tearDownModule():
    while _PATCHED:
        owner, name, original = _PATCHED.pop()
        setattr(owner, name, original)


def _make_bel():
    return BEL(
        mode="mvn",
        X_pre_processing=Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=3, svd_solver="full")),
            ]
        ),
        Y_pre_processing=Pipeline([("scaler", StandardScaler())]),
        regression_model=CCA(n_components=2, max_iter=2000),
        n_comp_cca=2,
    )


@cache
def _fixture():
    """The single fit of this module; never predicted or exported directly."""
    bel = _make_bel()
    bel.fit(X_TRAIN, Y_TRAIN)
    return bel


def _canonical_moments(bel):
    return bel.posterior_mean.copy(), bel.posterior_covariance.copy()


def _original_covariances(linear, canonical_covariances):
    return np.array([linear.T @ cov @ linear for cov in canonical_covariances])


@cache
def _reference():
    """Actual public BEL and trusted joblib moments, mapped with the public inverse."""
    pristine = _fixture()
    buffer = io.BytesIO()
    joblib.dump(pristine, buffer)  # newly created artificial fixture bytes only
    buffer.seek(0)
    trusted = joblib.load(buffer)
    bel = copy.deepcopy(pristine)

    bel.predict(QUERY, noise=None, return_samples=False)
    default = _canonical_moments(bel)
    default_noise = bel.noise
    bel.predict(QUERY, noise=NOISE, return_samples=False)
    noisy = _canonical_moments(bel)
    trusted.predict(QUERY, noise=NOISE, return_samples=False)
    trusted_noisy = _canonical_moments(trusted)

    mapped = bel.inverse_transform(np.concatenate([default[0], noisy[0]]))[:, 0, :]
    trusted_mapped = trusted.inverse_transform(trusted_noisy[0])[:, 0, :]
    basis = np.vstack([np.zeros((1, 2)), np.eye(2)])
    image = bel.inverse_transform(basis[np.newaxis])[0]
    linear = image[1:] - image[0]
    return {
        "default_noise": default_noise,
        "default": {
            "canonical_mean": default[0],
            "mean": mapped[:2],
            "cov": _original_covariances(linear, default[1]),
        },
        "noise": {
            "canonical_mean": noisy[0],
            "mean": mapped[2:],
            "cov": _original_covariances(linear, noisy[1]),
        },
        "trusted": {
            "canonical_mean": trusted_noisy[0],
            "mean": trusted_mapped,
            "cov": _original_covariances(linear, trusted_noisy[1]),
        },
    }


def _same_bel_state(left, right):
    """Whether the fitted state, attribute set and absence of caches are unchanged."""
    if set(vars(left)) != set(vars(right)):
        return False
    if hasattr(right, "posterior_mean") or hasattr(right, "noise"):
        return False
    pairs = [(left.X_f, right.X_f), (left.Y_f, right.Y_f)]
    left_model, right_model = left.regression_model, right.regression_model
    for name in ("x_rotations_", "x_loadings_", "y_loadings_", "x_weights_", "y_weights_"):
        pairs.append((getattr(left_model, name), getattr(right_model, name)))
    left_x, right_x = left.X_pre_processing, right.X_pre_processing
    pairs.append((left_x["pca"].components_, right_x["pca"].components_))
    pairs.append((left_x["scaler"].mean_, right_x["scaler"].mean_))
    pairs.append((left_x["scaler"].scale_, right_x["scaler"].scale_))
    return all(np.array_equal(first, second) for first, second in pairs)


@cache
def _exported():
    """The single valid export, from a private copy that is mutated afterwards."""
    source = copy.deepcopy(_fixture())
    snapshot = copy.deepcopy(source)
    capsule = portable.export_linear_mvn(source)
    unchanged = _same_bel_state(snapshot, source)
    before = capsule.to_bytes()
    source.X_f[...] = 0.0
    source.Y_f[...] = 0.0
    source.regression_model.x_rotations_[...] = 0.0
    source.regression_model.y_loadings_[...] = 0.0
    return SimpleNamespace(
        capsule=capsule,
        unchanged=unchanged,
        before=before,
        after=capsule.to_bytes(),
    )


@cache
def _capsule_results():
    capsule = _exported().capsule
    default = capsule.predict_moments(QUERY)
    noisy = capsule.predict_moments(QUERY, noise=NOISE)
    saved = (noisy[0].copy(), noisy[1].copy())
    owned = [array.flags.owndata and array.flags.writeable for array in default + noisy]
    shared = np.shares_memory(noisy[0], noisy[1])
    noisy[0][...] = np.nan
    noisy[1][...] = np.nan
    again = capsule.predict_moments(QUERY, noise=NOISE)
    data = capsule.to_bytes()
    restored = LinearMVNCapsule.from_bytes(data, expected_sha256=capsule.sha256())
    return {
        "default": default,
        "noise": saved,
        "again": again,
        "restored": restored.predict_moments(QUERY, noise=NOISE),
        "owned": owned,
        "shared": shared,
    }


@cache
def _child_result():
    """Restore and predict in a nested fresh interpreter (no fit there)."""
    capsule = _exported().capsule
    root = os.path.dirname(os.path.dirname(os.path.abspath(skbel.__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [root, env.get("PYTHONPATH")]))
    command = [
        sys.executable,
        "-c",
        _CHILD,
        capsule.sha256(),
        json.dumps(QUERY.tolist()),
        repr(NOISE),
    ]
    with tempfile.TemporaryDirectory() as directory:
        completed = subprocess.run(
            command,
            input=capsule.to_bytes(),
            capture_output=True,
            cwd=directory,
            env=env,
            timeout=120,
            check=False,
        )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr.decode("utf-8", "replace")[-2000:])
    last_line = completed.stdout.decode("utf-8").strip().splitlines()[-1]
    return json.loads(last_line)


# Independent rational Schur/affine oracle (exact Fraction arithmetic, no production helper).
def _rows(value):
    rows = value if isinstance(value[0], list) else [value]
    return [[Fraction(item) for item in row] for row in rows]


def _transpose(matrix):
    return [list(column) for column in zip(*matrix, strict=True)]


def _matmul(left, right):
    columns = list(zip(*right, strict=True))
    product = []
    for row in left:
        product.append([sum(a * b for a, b in zip(row, col, strict=True)) for col in columns])
    return product


def _add(first, second):
    rows = []
    for row_a, row_b in zip(first, second, strict=True):
        rows.append([x + y for x, y in zip(row_a, row_b, strict=True)])
    return rows


def _sub(first, second):
    rows = []
    for row_a, row_b in zip(first, second, strict=True):
        rows.append([x - y for x, y in zip(row_a, row_b, strict=True)])
    return rows


def _scale(matrix, factor):
    return [[factor * item for item in row] for row in matrix]


def _inverse2(matrix):
    (a, b), (c, d) = matrix
    det = a * d - b * c
    return [[d / det, -b / det], [-c / det, a / det]]


def _rational_oracle(arrays, queries, noise):
    """Schur conditioning of the joint Gaussian (y, d) followed by affine reconstruction."""
    m = {name: _rows(value) for name, value in arrays.items()}
    g_t = _transpose(m["G"])
    s22 = _matmul(_matmul(m["G"], m["C_y"]), g_t)
    s22 = _add(s22, m["C_e"])
    s22 = _add(s22, _scale(m["B"], Fraction(noise)))
    gain = _matmul(_matmul(m["C_y"], g_t), _inverse2(s22))
    c_post = _sub(m["C_y"], _matmul(_matmul(gain, m["G"]), m["C_y"]))
    covariance = _matmul(_matmul(_transpose(m["A_y"]), c_post), m["A_y"])
    mu_d = _add(_matmul(m["mu_y"], g_t), m["mu_e"])
    means = []
    for row in queries:
        data = _add(_matmul(_rows(row), m["A_x"]), m["b_x"])
        mu_z = _add(m["mu_y"], _matmul(_sub(data, mu_d), _transpose(gain)))
        means.append(_add(_matmul(mu_z, m["A_y"]), m["b_y"])[0])
    as_floats = [[float(item) for item in row] for row in means]
    cov_floats = [[float(item) for item in row] for row in covariance]
    return np.array(as_floats), np.array(cov_floats)


# Hand-built wire documents (encoded without the production encoder).
def _wire_payload():
    return {
        "schema": portable.SCHEMA,
        "inference_convention": portable.INFERENCE_CONVENTION,
        "dimensions": dict(DIMS),
        **copy.deepcopy(RATIONAL),
    }


def _encode(payload):
    return json.dumps(payload).encode("utf-8")


def _with(**overrides):
    payload = _wire_payload()
    payload.update(overrides)
    return _encode(payload)


def _without(key):
    payload = _wire_payload()
    del payload[key]
    return _encode(payload)


def _entry(name, index, value):
    matrix = copy.deepcopy(RATIONAL[name])
    holder = matrix
    for position in index[:-1]:
        holder = holder[position]
    holder[index[-1]] = value
    return _with(**{name: matrix})


def _synthetic(predictor, canonical, target):
    identity = np.eye(canonical).tolist()
    return {
        "schema": portable.SCHEMA,
        "inference_convention": portable.INFERENCE_CONVENTION,
        "dimensions": {"predictor": predictor, "target": target, "canonical": canonical},
        "A_x": np.zeros((predictor, canonical)).tolist(),
        "b_x": [0.0] * canonical,
        "A_y": np.zeros((canonical, target)).tolist(),
        "b_y": [0.0] * target,
        "mu_y": [0.0] * canonical,
        "C_y": identity,
        "G": np.zeros((canonical, canonical)).tolist(),
        "mu_e": [0.0] * canonical,
        "C_e": identity,
        "B": identity,
    }


def _kwargs(**overrides):
    arrays = {name: np.array(value, dtype=np.float64) for name, value in RATIONAL.items()}
    arrays.update(overrides)
    return arrays


_WIRE_BYTES = _with()


def _count_numbers(value):
    if isinstance(value, list):
        return sum(_count_numbers(item) for item in value)
    return 1


# Profile mutators; each receives a private deep copy of the fitted fixture.
def _use_subclass(bel):
    bel.__class__ = _SubBEL


def _use_kde_mode(bel):
    bel.mode = "kde"


def _use_nonlinear_pre(bel):
    bel.X_pre_processing = Pipeline([("power", PowerTransformer())])


def _use_scaled_post(bel):
    bel.Y_post_processing = Pipeline([("scaler", StandardScaler())])


def _use_unfitted_cca(bel):
    bel.regression_model = CCA(n_components=2)


def _use_observation_override(bel):
    bel.x_observation = np.zeros((1, 3))


def _use_nonfinite_state(bel):
    bel.X_f[0, 0] = np.nan


def _use_singular_state(bel):
    bel.Y_f[:, 1] = bel.Y_f[:, 0]


class TestPublicBelParity(unittest.TestCase):
    def test_capsule_matches_actual_public_bel_for_default_and_explicit_noise(self):
        reference, results = _reference(), _capsule_results()
        self.assertEqual(reference["default_noise"], 0.01)
        for key in ("default", "noise"):
            with self.subTest(key=key):
                mean, cov = results[key]
                self.assertEqual(mean.shape, (2, 3))
                self.assertEqual(cov.shape, (2, 3, 3))
                np.testing.assert_allclose(mean, reference[key]["mean"], rtol=0, atol=ATOL)
                np.testing.assert_allclose(cov, reference[key]["cov"], rtol=0, atol=ATOL)
        default_cov, noisy_cov = results["default"][1], results["noise"][1]
        self.assertFalse(np.allclose(default_cov, noisy_cov, rtol=0, atol=1e-6))

    def test_capsule_matches_trusted_joblib_baseline(self):
        reference, results = _reference(), _capsule_results()
        trusted, plain = reference["trusted"], reference["noise"]
        np.testing.assert_allclose(
            trusted["canonical_mean"], plain["canonical_mean"], rtol=0, atol=ATOL
        )
        mean, cov = results["noise"]
        np.testing.assert_allclose(mean, trusted["mean"], rtol=0, atol=ATOL)
        np.testing.assert_allclose(cov, trusted["cov"], rtol=0, atol=ATOL)


class TestIndependentOracle(unittest.TestCase):
    def test_rational_schur_affine_oracle(self):
        capsule = LinearMVNCapsule.from_bytes(_WIRE_BYTES)
        mean, cov = capsule.predict_moments(np.array(RATIONAL_QUERY), noise=RATIONAL_NOISE)
        expected_mean, expected_cov = _rational_oracle(RATIONAL, RATIONAL_QUERY, RATIONAL_NOISE)
        self.assertEqual(mean.shape, (2, 3))
        self.assertEqual(cov.shape, (2, 3, 3))
        np.testing.assert_allclose(mean, expected_mean, rtol=0, atol=ATOL)
        for case_cov in cov:
            np.testing.assert_allclose(case_cov, expected_cov, rtol=0, atol=ATOL)


class TestExport(unittest.TestCase):
    def test_export_leaves_source_unchanged_and_owns_its_state(self):
        exported = _exported()
        self.assertTrue(exported.unchanged)
        self.assertEqual(exported.before, exported.after)

    def test_unsupported_profiles_fail_closed(self):
        cases = [
            ("subclass", _use_subclass),
            ("kde mode", _use_kde_mode),
            ("nonlinear pre-processing", _use_nonlinear_pre),
            ("non-passthrough post-processing", _use_scaled_post),
            ("unfitted CCA", _use_unfitted_cca),
            ("cached observation override", _use_observation_override),
            ("non-finite paired state", _use_nonfinite_state),
            ("singular paired state", _use_singular_state),
        ]
        for label, mutate in cases:
            with self.subTest(label):
                broken = copy.deepcopy(_fixture())
                mutate(broken)
                with self.assertRaises(LinearMVNError):
                    portable.export_linear_mvn(broken)

    def test_errors_are_value_errors(self):
        self.assertTrue(issubclass(LinearMVNError, ValueError))


class TestWireFormat(unittest.TestCase):
    def test_schema_constant(self):
        self.assertEqual(portable.SCHEMA, "skbel.linear-mvn-capsule/v1")

    def test_exported_bytes_are_deterministic_strict_and_data_only(self):
        capsule = _exported().capsule
        data = capsule.to_bytes()
        self.assertEqual(data, capsule.to_bytes())
        self.assertLess(len(data), portable.MAX_BYTES)
        document = json.loads(data)
        expected_keys = {"schema", "inference_convention", "dimensions", *RATIONAL}
        self.assertEqual(set(document), expected_keys)
        self.assertEqual(document["schema"], "skbel.linear-mvn-capsule/v1")
        self.assertEqual(document["dimensions"], {"predictor": 4, "target": 3, "canonical": 2})
        # P*Q + Q + Q*R + R + 2*Q + 4*Q*Q numbers: no training rows (16 > P, R) or caches.
        count = sum(_count_numbers(document[name]) for name in RATIONAL)
        self.assertEqual(count, 4 * 2 + 2 + 2 * 3 + 3 + 2 * 2 + 4 * 2 * 2)
        stripped = data.replace(b"skbel.linear-mvn-capsule/v1", b"")
        for token in (b"sklearn", b"numpy", b"joblib", b"pickle", b"BEL", b"/"):
            self.assertNotIn(token, stripped)

    def test_roundtrip_is_exact_and_digest_is_external_integrity_only(self):
        capsule = _exported().capsule
        data = capsule.to_bytes()
        digest = hashlib.sha256(data).hexdigest()
        self.assertEqual(capsule.sha256(), digest)
        restored = LinearMVNCapsule.from_bytes(data, expected_sha256=digest)
        self.assertEqual(restored.to_bytes(), data)
        for name, array in capsule.arrays().items():
            np.testing.assert_array_equal(restored.arrays()[name], array)
        self.assertEqual(
            (restored.predictor_dim, restored.canonical_dim, restored.target_dim), (4, 2, 3)
        )
        LinearMVNCapsule.from_bytes(data, expected_sha256=digest.upper())
        for bad in ("0" * 64, digest[:-1], digest + "0", "g" * 64, digest.encode(), 12):
            with self.subTest(bad=bad), self.assertRaises(LinearMVNError):
                LinearMVNCapsule.from_bytes(data, expected_sha256=bad)
        # The digest is checked before parsing: garbage with a wrong digest fails on the digest.
        with self.assertRaisesRegex(LinearMVNError, "SHA-256"):
            LinearMVNCapsule.from_bytes(b"not json", expected_sha256=digest)
        # A rehashed, still valid alteration is accepted: SHA-256 is not authenticity.
        self.assertIn(b"0.25", _WIRE_BYTES)
        altered = _WIRE_BYTES.replace(b"0.25", b"0.75", 1)
        original_digest = hashlib.sha256(_WIRE_BYTES).hexdigest()
        with self.assertRaises(LinearMVNError):
            LinearMVNCapsule.from_bytes(altered, expected_sha256=original_digest)
        LinearMVNCapsule.from_bytes(altered, expected_sha256=hashlib.sha256(altered).hexdigest())

    def test_rejects_hostile_and_malformed_documents(self):
        self.assertIn(b"0.25", _WIRE_BYTES)
        hook = "__import__('builtins').setattr(__import__('sys'), '_capsule_test_executed', True)"
        cases = {
            "empty": b"",
            "not json": b"not json",
            "utf8 bom": b"\xef\xbb\xbf" + _WIRE_BYTES,
            "invalid utf8": b"\xff\xfe" + _WIRE_BYTES,
            "trailing data": _WIRE_BYTES + b"{}",
            "top-level list": b"[]",
            "top-level number": b"1",
            "duplicate key": _WIRE_BYTES.replace(b"{", b'{"schema": "x", ', 1),
            "missing key": _without("G"),
            "unknown key": _with(producer="sklearn.cross_decomposition.CCA"),
            "unknown version": _with(schema="skbel.linear-mvn-capsule/v2"),
            "schema not a string": _with(schema=1),
            "unknown convention": _with(inference_convention="other"),
            "dimension missing": _with(dimensions={"predictor": 3, "target": 3}),
            "dimension extra": _with(dimensions={**DIMS, "extra": 1}),
            "dimension bool": _with(dimensions={**DIMS, "predictor": True}),
            "dimension float": _with(dimensions={**DIMS, "predictor": 3.0}),
            "dimension mismatch": _with(dimensions={**DIMS, "predictor": 4}),
            "dimension not an object": _with(dimensions=[3, 3, 2]),
            "bool entry": _entry("A_x", (0, 0), True),
            "mixed boolean entry": _entry("b_x", (0,), True),
            "string entry": _entry("b_x", (0,), "0.5"),
            "null entry": _entry("b_x", (0,), None),
            "nested entry": _entry("b_x", (0,), [0.5]),
            "object entry": _entry("b_x", (0,), {}),
            "executable string entry": _entry("b_y", (0,), hook),
            "executable schema": _with(schema=hook),
            "huge integer": _entry("b_x", (0,), 10**400),
            "enormous integer literal": _WIRE_BYTES.replace(b"0.25", b"9" * 5000, 1),
            "ragged matrix": _with(C_y=[[2.0, 0.5], [0.5]]),
            "missing row": _with(A_x=[[1.0, 0.0], [0.0, 1.0]]),
            "flat matrix": _with(C_y=[2.0, 0.5, 0.5, 1.0]),
            "nested vector": _with(b_x=[[0.25, -0.5]]),
            "short vector": _with(b_x=[0.25]),
            "scalar vector": _with(b_x=0.25),
            "NaN token": _WIRE_BYTES.replace(b"0.25", b"NaN", 1),
            "Infinity token": _WIRE_BYTES.replace(b"0.25", b"Infinity", 1),
            "negative Infinity token": _WIRE_BYTES.replace(b"0.25", b"-Infinity", 1),
            "overflowing float": _WIRE_BYTES.replace(b"0.25", b"1e999", 1),
            "asymmetric C_y": _with(C_y=[[2.0, 0.5], [0.6, 1.0]]),
            "zero C_y": _with(C_y=[[0.0, 0.0], [0.0, 0.0]]),
            "indefinite C_e": _with(C_e=[[1.0, 2.0], [2.0, 1.0]]),
            "negative B": _with(B=[[1.0, 0.0], [0.0, -1.0]]),
            "singular B": _with(B=[[1.0, 1.0], [1.0, 1.0]]),
            "ill-conditioned C_y": _with(C_y=[[1.0, 0.0], [0.0, 1e-13]]),
            "bracket bomb": b"[" * 100_000 + b"]" * 100_000,
            "unterminated bracket bomb": b"[" * 100_000,
            "brace bomb": b'{"a":' * 1000 + b"1" + b"}" * 1000,
            "oversize blanks": b" " * (portable.MAX_BYTES + 1),
            "oversize padded document": _WIRE_BYTES + b" " * portable.MAX_BYTES,
        }
        for label, data in cases.items():
            with self.subTest(label), self.assertRaises(LinearMVNError):
                LinearMVNCapsule.from_bytes(data)
        self.assertFalse(hasattr(sys, "_capsule_test_executed"))

    def test_rejects_non_bytes_inputs(self):
        for label, value in (
            ("str", _WIRE_BYTES.decode("utf-8")),
            ("bytearray", bytearray(_WIRE_BYTES)),
            ("memoryview", memoryview(_WIRE_BYTES)),
            ("none", None),
        ):
            with self.subTest(label), self.assertRaises(LinearMVNError):
                LinearMVNCapsule.from_bytes(value)

    def test_size_and_dimension_limits(self):
        padded = _WIRE_BYTES + b" " * (portable.MAX_BYTES - len(_WIRE_BYTES))
        self.assertEqual(len(padded), portable.MAX_BYTES)
        LinearMVNCapsule.from_bytes(padded)
        largest = LinearMVNCapsule.from_bytes(_encode(_synthetic(128, 32, 128)))
        self.assertEqual(
            (largest.predictor_dim, largest.canonical_dim, largest.target_dim), (128, 32, 128)
        )
        for dims in ((129, 32, 128), (128, 32, 129), (128, 33, 128), (3, 1, 3), (2, 3, 3)):
            with self.subTest(dims=dims), self.assertRaises(LinearMVNError):
                LinearMVNCapsule.from_bytes(_encode(_synthetic(*dims)))


class TestFreshInterpreter(unittest.TestCase):
    def test_nested_interpreter_restores_and_predicts_without_fit(self):
        child = _child_result()
        mean, cov = _capsule_results()["noise"]
        np.testing.assert_allclose(child["mean"], mean, rtol=0, atol=ATOL)
        np.testing.assert_allclose(child["cov"], cov, rtol=0, atol=ATOL)


class TestOwnershipAndInputs(unittest.TestCase):
    def test_outputs_are_owned_and_unaffected_by_mutating_earlier_results(self):
        results = _capsule_results()
        self.assertTrue(all(results["owned"]))
        self.assertFalse(results["shared"])
        np.testing.assert_array_equal(results["again"][0], results["noise"][0])
        np.testing.assert_array_equal(results["again"][1], results["noise"][1])
        np.testing.assert_array_equal(results["restored"][0], results["noise"][0])
        np.testing.assert_array_equal(results["restored"][1], results["noise"][1])
        self.assertEqual(results["default"][0].dtype, np.float64)

    def test_constructor_and_accessors_copy_state(self):
        supplied = _kwargs()
        built = LinearMVNCapsule(**supplied)
        before = built.to_bytes()
        for array in supplied.values():
            array[...] = 7.0
        self.assertEqual(built.to_bytes(), before)
        for array in built.arrays().values():
            array[...] = 7.0
        self.assertEqual(built.to_bytes(), before)
        self.assertEqual(LinearMVNCapsule(**_kwargs()).to_bytes(), before)

    def test_direct_construction_validates_like_restoration(self):
        bad = {
            "bool array": _kwargs(b_x=np.array([True, False])),
            "mixed boolean list": _kwargs(b_x=[True, 0.0]),
            "mixed numpy boolean list": _kwargs(b_x=[np.True_, 0.0]),
            "nested mixed boolean": _kwargs(C_y=[[2.0, 0.5], [0.5, True]]),
            "complex array": _kwargs(b_x=np.array([1.0 + 0j, 0.5])),
            "object array": _kwargs(b_x=np.array([0.25, -0.5], dtype=object)),
            "nan": _kwargs(b_x=np.array([np.nan, 0.5])),
            "wrong shape": _kwargs(b_x=np.array([0.25])),
            "1D A_x": _kwargs(A_x=np.array([1.0, 0.0])),
            "asymmetric": _kwargs(C_y=np.array([[2.0, 0.5], [0.6, 1.0]])),
            "indefinite": _kwargs(C_e=np.array([[1.0, 2.0], [2.0, 1.0]])),
            "masked": _kwargs(b_x=np.ma.masked_array([0.25, -0.5], mask=[False, True])),
        }
        for label, kwargs in bad.items():
            with self.subTest(label), self.assertRaises(LinearMVNError):
                LinearMVNCapsule(**kwargs)

    def test_observation_validation(self):
        invalid = {
            "bool": np.ones((2, 3), dtype=bool),
            "object": np.ones((2, 3), dtype=object),
            "complex": np.ones((2, 3), dtype=complex),
            "string": [["a", "b", "c"]],
            "mixed boolean": [[True, 0.0, 0.0]],
            "one dimension": np.ones(3),
            "three dimensions": np.ones((1, 1, 3)),
            "no rows": np.ones((0, 3)),
            "no columns": np.ones((2, 0)),
            "wrong width": np.ones((2, 4)),
            "nan": np.array([[np.nan, 0.0, 0.0]]),
            "inf": np.array([[np.inf, 0.0, 0.0]]),
            "ragged": [[1.0, 2.0, 3.0], [1.0]],
            "masked": np.ma.masked_array(np.ones((1, 3)), mask=[[False, True, False]]),
            "none": None,
        }
        for label, value in invalid.items():
            with self.subTest(label), self.assertRaises(LinearMVNError):
                portable._check_observations(value, 3)
        source = np.ones((2, 3), dtype=int)
        checked = portable._check_observations(source, 3)
        checked[0, 0] = 9.0
        self.assertEqual(source[0, 0], 1)
        self.assertEqual(checked.dtype, np.float64)

    def test_noise_validation_matches_bel_contract(self):
        invalid = [
            True,
            np.bool_(False),
            -0.1,
            np.inf,
            -np.inf,
            np.nan,
            10**1000,
            [0.1],
            np.array([0.1]),
            "0.1",
        ]
        for noise in invalid:
            with self.subTest(noise=noise), self.assertRaises(LinearMVNError):
                portable._check_noise(noise)
        self.assertEqual(portable._check_noise(None), 0.01)
        self.assertEqual(portable._check_noise(0), 0.0)
        self.assertEqual(portable._check_noise(1), 1.0)
        self.assertEqual(portable._check_noise(np.float64(0.5)), 0.5)

    def test_invalid_public_prediction_call_is_rejected(self):
        with self.assertRaises(LinearMVNError):
            _exported().capsule.predict_moments(np.zeros((1, 5)))

    def test_public_prediction_rejects_mixed_boolean_query(self):
        with self.assertRaises(LinearMVNError):
            _exported().capsule.predict_moments([[True, 0.0, 0.0, 0.0]])


class TestZCallBudget(unittest.TestCase):
    def test_finite_call_counts_stay_within_fixture_limits(self):
        self.assertLessEqual(_CALLS["fit"], CALL_LIMITS["fit"])
        self.assertLessEqual(_CALLS["predict"], CALL_LIMITS["predict"])
        # One more capsule prediction happens in the nested fresh interpreter.
        self.assertLessEqual(_CALLS["capsule_predict"] + 1, CALL_LIMITS["capsule_predict"])
        self.assertLessEqual(_CALLS["export"], CALL_LIMITS["export"])
        self.assertLessEqual(_CALLS["basis"], CALL_LIMITS["basis"])


if __name__ == "__main__":
    unittest.main()
