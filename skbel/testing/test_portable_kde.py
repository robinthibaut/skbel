"""Tests for the portable KDE prediction capsule (``skbel.learning.portable_kde``).

Uses one fixed literal fixture with no random variates; bounded fitting occurs during the producer/reference checks. The fresh consumer performs no fitting:

* BEL and CCA fits: 1 each; StandardScaler fits: 2; PCA fits: 0.
* Fixed-bandwidth component fits (``KernelDensity`` + ``LinearRegression``): at most 2
  in the independent source reference, at most 2 inside the single valid export,
  at most 4 in total.  ``GridSearchCV`` and ``TransportMap`` are throwing spies.
* Original ``BEL.predict`` (``return_samples=False``, precomputed reference functions): 1;
  ``BEL.random_sample`` with the frozen uniforms: 1.
* Public ``transform``/``inverse_transform``: 4 (query projection, baseline inverse and
  the two basis calls of the export), max 8.
* Valid capsule ``sample`` calls: 5 trained in-process (joint, two selected rows, repeat
  after mutating the result, restored copy) + 2 in the nested fresh interpreter +
  2 on the literal bimodal/point law = 9, max 16.  Invalid query, profile and wire
  cases never fit.

``TestZCallBudget`` measures the in-process counts and checks them against the limits.
Set ``SKBEL_PORTABLE_KDE_ARTIFACT_DIR`` to a directory to also write the frozen wire
documents and fixture numbers (at most 4 MiB in total) for an independent check.
"""

import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from functools import cache, wraps
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from scipy import interpolate
from sklearn.cross_decomposition import CCA, PLSCanonical
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

import skbel
from skbel import BEL
from skbel.algorithms.statistics import it_sampling, kde_params, posterior_conditional
from skbel.learning import portable_kde
from skbel.learning.portable_kde import KDEPredictionCapsule, KDEPredictionError
from skbel.tmaps import TransportMap

source_statistics = importlib.import_module("skbel.algorithms.statistics")

ATOL = 1e-10
# Coarse a-priori tolerance of the analytic bimodal shape checks (not a parity tolerance).
SHAPE_ATOL = 0.05
SEED = 17
N_SAMPLES = 8
BANDWIDTHS = [0.5, 0.5]
ARTIFACT_ENV = "SKBEL_PORTABLE_KDE_ARTIFACT_DIR"

X_TRAIN = np.array(
    [
        [-3.0, -3.0],
        [-3.0, -1.0],
        [-3.0, 1.0],
        [-3.0, 3.0],
        [-1.0, -3.0],
        [-1.0, -1.0],
        [-1.0, 1.0],
        [-1.0, 3.0],
        [1.0, -3.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [1.0, 3.0],
        [3.0, -3.0],
        [3.0, -1.0],
        [3.0, 1.0],
        [3.0, 3.0],
    ]
)
# First target is the first predictor; the second is a curved function of the second.
Y_TRAIN = np.array(
    [
        [-3.0, 8.25],
        [-3.0, 0.75],
        [-3.0, 1.25],
        [-3.0, 9.75],
        [-1.0, 8.25],
        [-1.0, 0.75],
        [-1.0, 1.25],
        [-1.0, 9.75],
        [1.0, 8.25],
        [1.0, 0.75],
        [1.0, 1.25],
        [1.0, 9.75],
        [3.0, 8.25],
        [3.0, 0.75],
        [3.0, 1.25],
        [3.0, 9.75],
    ]
)
QUERIES = np.array([[-0.5, -0.5], [0.5, 0.5]])
# (cases, samples, canonical component); channels of linear components are ignored.
U = np.array(
    [
        [
            [0.0, 0.125],
            [0.125, 0.25],
            [0.25, 0.375],
            [0.375, 0.5],
            [0.5, 0.625],
            [0.625, 0.75],
            [0.875, 0.875],
            [1.0, 1.0],
        ],
        [
            [0.0625, 1.0],
            [0.1875, 0.875],
            [0.3125, 0.75],
            [0.4375, 0.625],
            [0.5625, 0.5],
            [0.6875, 0.375],
            [0.8125, 0.25],
            [0.9375, 0.0],
        ],
    ]
)

# Literal bimodal PDF + point-mass numeric state (no training, no fit).
AXIS = np.linspace(-2.0, 2.0, 200)
BIMODAL_A_Y = [[2.0, 1.0], [-1.0, 3.0]]
BIMODAL_B_Y = [0.5, -1.0]
BIMODAL_SLOPE = 2.0
BIMODAL_INTERCEPT = -0.25

TEST_LIMITS = {
    "bel_fit": 1,
    "cca_fit": 1,
    "scaler_fit": 2,
    "component_fit": 4,
    "predict": 1,
    "random_sample": 1,
    "public_map": 8,
    "export": 24,
    "valid_sample": 16,
}
CHILD_SAMPLE_CALLS = 2
_CALLS = {
    key: 0
    for key in (
        "bel_fit",
        "cca_fit",
        "scaler_fit",
        "pca_fit",
        "kde_fit",
        "linear_fit",
        "grid_search",
        "transport_map",
        "predict",
        "random_sample",
        "public_map",
        "export",
        "sample",
        "sample_invalid",
    )
}
_PATCHED = []

_CHILD = """
import json
import sys

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skbel import BEL
from skbel.learning.portable_kde import KDEPredictionCapsule
from skbel.tmaps import TransportMap


def _forbidden(*args, **kwargs):
    raise RuntimeError("fitting or live prediction called in the restoring interpreter")


estimators = (BEL, CCA, PCA, StandardScaler, Pipeline, KernelDensity, LinearRegression)
for owner in (*estimators, GridSearchCV):
    owner.fit = _forbidden
    if hasattr(owner, "fit_transform"):
        owner.fit_transform = _forbidden
StandardScaler.partial_fit = _forbidden
BEL.predict = BEL.random_sample = BEL.kde_init = _forbidden
TransportMap.optimize = _forbidden

before = np.random.get_state()
data = sys.stdin.buffer.read()
capsule = KDEPredictionCapsule.from_bytes(data, expected_sha256=sys.argv[1])
queries = np.array(json.loads(sys.argv[2]))
uniforms = np.array(json.loads(sys.argv[3]))
full = capsule.sample(queries, uniforms)
selected = capsule.sample(queries, uniforms, obs_n=1)
after = np.random.get_state()
unchanged = (
    before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
)
print(json.dumps({"full": full.tolist(), "selected": selected.tolist(), "rng": bool(unchanged)}))
"""


class _SubBEL(BEL):
    """Exact-type profile checks must reject subclasses."""


def _count(owner, name, key, forbid=False):
    original = getattr(owner, name)
    owned = name in vars(owner)

    @wraps(original)
    def counted(*args, **kwargs):
        _CALLS[key] += 1
        if forbid:
            raise AssertionError(f"{name} must not be called")
        return original(*args, **kwargs)

    setattr(owner, name, counted)
    _PATCHED.append((owner, name, original, owned))


def _count_sample():
    original = KDEPredictionCapsule.sample

    @wraps(original)
    def counted(self, *args, **kwargs):
        _CALLS["sample"] += 1
        try:
            return original(self, *args, **kwargs)
        except KDEPredictionError:
            _CALLS["sample_invalid"] += 1
            raise

    KDEPredictionCapsule.sample = counted
    _PATCHED.append((KDEPredictionCapsule, "sample", original, True))


def setUpModule():
    _count(BEL, "fit", "bel_fit")
    _count(CCA, "fit", "cca_fit")
    _count(StandardScaler, "fit", "scaler_fit")
    _count(PCA, "fit", "pca_fit")
    _count(KernelDensity, "fit", "kde_fit")
    _count(LinearRegression, "fit", "linear_fit")
    _count(GridSearchCV, "fit", "grid_search", forbid=True)
    _count(TransportMap, "optimize", "transport_map", forbid=True)
    _count(BEL, "predict", "predict")
    _count(BEL, "random_sample", "random_sample")
    _count(BEL, "transform", "public_map")
    _count(BEL, "inverse_transform", "public_map")
    _count(portable_kde, "export_kde", "export")
    _count_sample()


def tearDownModule():
    while _PATCHED:
        owner, name, original, owned = _PATCHED.pop()
        if owned:
            setattr(owner, name, original)
        else:
            delattr(owner, name)


def _same_rng(first, second):
    return (
        first[0] == second[0]
        and np.array_equal(first[1], second[1])
        and tuple(first[2:]) == tuple(second[2:])
    )


@cache
def _trained():
    """The single paired fit of this module."""
    bel = BEL(
        mode="kde",
        X_pre_processing=Pipeline([("scale", StandardScaler())]),
        Y_pre_processing=Pipeline([("scale", StandardScaler())]),
        regression_model=CCA(n_components=2, scale=False, max_iter=1000, tol=1e-10),
        random_state=SEED,
    )
    bel.fit(X_TRAIN, Y_TRAIN)
    return bel


def _component_fits():
    return _CALLS["kde_fit"] + _CALLS["linear_fit"]


def _reference_functions(bel, canonical_queries):
    """Fixed-bandwidth functions built from the source helpers exactly as ``BEL.predict``."""
    n_obs = canonical_queries.shape[0]
    functions = np.zeros((n_obs, 2), dtype="object")
    kinds = []
    for comp in range(2):
        x_scores, y_scores = bel.X_f.T[comp], bel.Y_f.T[comp]
        corr = np.corrcoef(x_scores, y_scores).diagonal(offset=1)[0]
        if corr >= 0.999:
            fun = LinearRegression().fit(x_scores.reshape(-1, 1), y_scores.reshape(-1, 1))
            column = [{"kind": "linear", "function": fun, "bandwidth": 0}] * n_obs
            kinds.append("linear")
        else:
            dens, support, bw = kde_params(x=x_scores, y=y_scores, bw=BANDWIDTHS[comp])
            dens[dens < 1e-8] = 0
            column = []
            for dp in canonical_queries:
                hp, sup = posterior_conditional(
                    X_obs=dp.T[comp], dens=dens, support=support, k=2**7 + 1
                )
                hp[np.abs(hp) < 1e-8] = 0
                fun = interpolate.interp1d(sup, hp, kind="linear")
                column.append({"kind": "pdf", "function": fun, "bandwidth": bw})
            kinds.append("pdf")
        functions[:, comp] = column
    return functions, tuple(kinds)


@cache
def _baseline():
    """Live same-profile BEL: one predict with reference functions, one frozen-U sample."""
    bel = _trained()
    before = _component_fits()
    canonical_queries = bel.transform(X=QUERIES)
    functions, kinds = _reference_functions(bel, canonical_queries)
    reference_fits = _component_fits() - before
    bel.predict(QUERIES, n_posts=N_SAMPLES, return_samples=False, precomputed_kde=functions)

    pdf_channels = [j for j in range(2) if kinds[j] == "pdf"]
    queue = [U[i, :, j].copy() for i in range(len(QUERIES)) for j in pdf_channels]
    requests = []

    def queued_uniform(low, high, size):
        requests.append((low, high, size))
        return queue.pop(0)

    original_uniform = source_statistics.uniform
    rng_state = np.random.get_state()
    source_statistics.uniform = queued_uniform
    try:
        canonical = bel.random_sample(X_obs_f=bel.X_obs_f, n_posts=N_SAMPLES)
    finally:
        source_statistics.uniform = original_uniform
        np.random.set_state(rng_state)
    return SimpleNamespace(
        kinds=kinds,
        canonical=canonical,
        samples=bel.inverse_transform(canonical),
        requests=requests,
        leftover=len(queue),
        reference_fits=reference_fits,
        rng_restored=_same_rng(rng_state, np.random.get_state()),
        uniform_restored=source_statistics.uniform is original_uniform,
    )


_MODEL_ARRAYS = (
    "x_rotations_",
    "y_rotations_",
    "x_loadings_",
    "y_loadings_",
    "x_weights_",
    "y_weights_",
    "_x_mean",
    "_y_mean",
    "_x_std",
    "_y_std",
)


def _tracked_arrays(bel):
    arrays = {"X_f": bel.X_f, "Y_f": bel.Y_f, "X_obs_f": bel.X_obs_f}
    for role, pipeline in (("x", bel.X_pre_processing), ("y", bel.Y_pre_processing)):
        arrays[f"{role}_mean"] = pipeline["scale"].mean_
        arrays[f"{role}_scale"] = pipeline["scale"].scale_
    for name in _MODEL_ARRAYS:
        arrays[name] = getattr(bel.regression_model, name)
    return arrays


def _snapshot(bel):
    arrays = _tracked_arrays(bel)
    return SimpleNamespace(
        attributes=set(vars(bel)),
        identities={name: id(value) for name, value in arrays.items()},
        values={name: np.array(value, copy=True) for name, value in arrays.items()},
        objects=(bel.X_pre_processing, bel.Y_pre_processing, bel.regression_model),
        functions=bel.kde_functions,
        entries=[id(item) for item in bel.kde_functions.ravel()],
        scalars=(bel.mode, bel.n_posts, bel.noise, bel.seed),
    )


def _changes(snapshot, bel):
    """Names of every tracked source array, processor or cache that changed."""
    changed = []
    if set(vars(bel)) != snapshot.attributes:
        changed.append("attributes")
    arrays = _tracked_arrays(bel)
    for name, value in arrays.items():
        if id(value) != snapshot.identities[name] or not np.array_equal(
            value, snapshot.values[name]
        ):
            changed.append(name)
    current = (bel.X_pre_processing, bel.Y_pre_processing, bel.regression_model)
    if any(a is not b for a, b in zip(current, snapshot.objects, strict=True)):
        changed.append("processors")
    if bel.kde_functions is not snapshot.functions:
        changed.append("kde_functions")
    elif [id(item) for item in bel.kde_functions.ravel()] != snapshot.entries:
        changed.append("kde_function entries")
    if (bel.mode, bel.n_posts, bel.noise, bel.seed) != snapshot.scalars:
        changed.append("scalars")
    return changed


@cache
def _exported():
    """The single valid export, after the live baseline has populated its caches."""
    bel = _trained()
    _baseline()
    snapshot = _snapshot(bel)
    before = _component_fits()
    capsule = portable_kde.export_kde(bel, BANDWIDTHS)
    return SimpleNamespace(
        capsule=capsule,
        compiler_fits=_component_fits() - before,
        changes=_changes(snapshot, bel),
    )


@cache
def _capsule_results():
    capsule = _exported().capsule
    queries, uniforms = QUERIES.copy(), U.copy()
    rng_state = np.random.get_state()
    joint = capsule.sample(queries, uniforms)
    rng_unchanged = _same_rng(rng_state, np.random.get_state())
    inputs_unchanged = np.array_equal(queries, QUERIES) and np.array_equal(uniforms, U)
    first = capsule.sample(QUERIES, U, obs_n=0)
    last = capsule.sample(QUERIES, U, obs_n=-1)
    saved = joint.copy()
    owned = bool(joint.flags.owndata and joint.flags.writeable)
    joint[...] = np.nan
    again = capsule.sample(QUERIES, U)
    data = capsule.to_bytes()
    restored = KDEPredictionCapsule.from_bytes(data, expected_sha256=capsule.sha256())
    return SimpleNamespace(
        joint=saved,
        first=first,
        last=last,
        again=again,
        restored=restored.sample(QUERIES, U),
        restored_bytes=restored.to_bytes(),
        owned=owned,
        rng_unchanged=rng_unchanged,
        inputs_unchanged=inputs_unchanged,
    )


@cache
def _child_result():
    """Restore and sample in a nested fresh interpreter in which every fit raises."""
    capsule = _exported().capsule
    root = os.path.dirname(os.path.dirname(os.path.abspath(skbel.__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [root, env.get("PYTHONPATH")]))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = "1"
    command = [
        sys.executable,
        "-c",
        _CHILD,
        capsule.sha256(),
        json.dumps(QUERIES.tolist()),
        json.dumps(U.tolist()),
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


# Literal numeric bimodal + point-mass law (hand-built document, no production encoder).
def _bimodal_density():
    profile = (
        0.2
        + np.maximum(0.0, 1.0 - np.abs(AXIS + 1.0) / 0.3)
        + np.maximum(0.0, 1.0 - np.abs(AXIS - 1.0) / 0.3)
    )
    return np.repeat(profile[:, np.newaxis], 200, axis=1)  # rows: y, columns: x


def _pdf_record(density, x_axis=AXIS, y_axis=AXIS, bandwidth=0.5):
    return {
        "kind": "pdf",
        "bandwidth": bandwidth,
        "x_axis": np.asarray(x_axis).tolist(),
        "y_axis": np.asarray(y_axis).tolist(),
        "density": np.asarray(density).tolist(),
    }


def _pdf_state(density):
    return {
        "schema": portable_kde.SCHEMA,
        "convention": portable_kde.CONVENTION,
        "dimensions": {"predictor": 2, "canonical": 2, "target": 2},
        "A_x": [[1.0, 0.0], [0.0, 1.0]],
        "b_x": [0.0, 0.0],
        "A_y": copy.deepcopy(BIMODAL_A_Y),
        "b_y": list(BIMODAL_B_Y),
        "components": [
            _pdf_record(density),
            {"kind": "linear", "slope": BIMODAL_SLOPE, "intercept": BIMODAL_INTERCEPT},
        ],
    }


def _bimodal_state():
    return _pdf_state(_bimodal_density())


def _encode(payload):
    return json.dumps(payload).encode("utf-8")


@cache
def _bimodal_bytes():
    return _encode(_bimodal_state())


@cache
def _bimodal_results():
    capsule = KDEPredictionCapsule.from_bytes(_bimodal_bytes())
    return SimpleNamespace(
        full=capsule.sample(QUERIES, U),
        selected=capsule.sample(QUERIES, U, obs_n=1),
    )


@cache
def _bimodal_source_expectation():
    """Source helpers (not the module under test) plus explicit affine arithmetic."""
    density = _bimodal_density()
    canonical = np.empty((len(QUERIES), N_SAMPLES, 2))
    for i, (d0, d1) in enumerate(QUERIES):  # A_x = I, b_x = 0
        hp, sup = posterior_conditional(
            X_obs=d0, dens=density.copy(), support=(AXIS.copy(), AXIS.copy()), k=2**7 + 1
        )
        hp[np.abs(hp) < 1e-8] = 0
        pdf = interpolate.interp1d(sup, hp, kind="linear")
        cdf = it_sampling(
            pdf=pdf, lower_bd=pdf.x.min(), upper_bd=pdf.x.max(), k=2**7 + 1, return_cdf=True
        )
        canonical[i, :, 0] = np.interp(U[i, :, 0], cdf, pdf.x)  # it_sampling's draw rule
        canonical[i, :, 1] = BIMODAL_SLOPE * d1 + BIMODAL_INTERCEPT
    z0, z1 = canonical[..., 0], canonical[..., 1]
    samples = np.stack([2.0 * z0 - 1.0 * z1 + 0.5, 1.0 * z0 + 3.0 * z1 - 1.0], axis=-1)
    return canonical, samples


def _bimodal_canonical(samples):
    """Invert y = z A_y + b_y with the exact inverse (1/7) [[3, -1], [1, 2]]."""
    shifted = np.asarray(samples) - np.array(BIMODAL_B_Y)
    inverse = np.array([[3.0, -1.0], [1.0, 2.0]]) / 7.0
    return shifted @ inverse


# Independent piecewise-linear CDF inverse (exact rational arithmetic, no NumPy interp).
def _pwl_quantile(cdf, support, u):
    c = [Fraction(value) for value in cdf]
    t = [Fraction(value) for value in support]
    q = Fraction(u)
    if q >= c[-1]:
        return float(t[-1])
    j = max(index for index in range(len(c)) if c[index] <= q)  # rightmost knot not above u
    if c[j] == q:
        return float(t[j])
    return float(t[j] + (q - c[j]) * (t[j + 1] - t[j]) / (c[j + 1] - c[j]))


# Small two-point-mass document for structural wire tests.
LINEAR_DIMS = {"predictor": 2, "canonical": 2, "target": 3}
LINEAR_COMPONENTS = [
    {"kind": "linear", "slope": 2.0, "intercept": -0.75},
    {"kind": "linear", "slope": -0.5, "intercept": 0.125},
]


def _linear_state():
    return {
        "schema": portable_kde.SCHEMA,
        "convention": portable_kde.CONVENTION,
        "dimensions": dict(LINEAR_DIMS),
        "A_x": [[1.0, 0.5], [-0.5, 1.0]],
        "b_x": [0.25, -0.5],
        "A_y": [[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]],
        "b_y": [0.5, -0.25, 1.0],
        "components": copy.deepcopy(LINEAR_COMPONENTS),
    }


def _with(**overrides):
    payload = _linear_state()
    payload.update(overrides)
    return _encode(payload)


def _without(key):
    payload = _linear_state()
    del payload[key]
    return _encode(payload)


def _linear_component(index, drop=None, **changes):
    payload = _linear_state()
    component = payload["components"][index]
    component.update(changes)
    if drop is not None:
        del component[drop]
    return _encode(payload)


def _pdf_with(drop=None, **changes):
    state = _bimodal_state()
    state["components"][0].update(changes)
    if drop is not None:
        del state["components"][0][drop]
    return state


def _nested(depth, leaf):
    value = leaf
    for _ in range(depth):
        value = [value]
    return value


_WIRE_BYTES = _with()


def _count_numbers(value):
    if isinstance(value, list):
        return sum(_count_numbers(item) for item in value)
    if isinstance(value, dict):
        return sum(_count_numbers(item) for key, item in value.items() if key != "kind")
    return 1


# Profile mutators; each receives a shallow copy and only rebinds attributes.
def _use_subclass(bel):
    bel.__class__ = _SubBEL


def _use_mvn_mode(bel):
    bel.mode = "mvn"


def _use_tm_mode(bel):
    bel.mode = "tm"


def _use_nonlinear_pre(bel):
    bel.X_pre_processing = Pipeline([("power", PowerTransformer())])


def _use_whitened_pca(bel):
    bel.Y_pre_processing = Pipeline([("pca", PCA(whiten=True))])


def _use_scaled_post(bel):
    bel.Y_post_processing = Pipeline([("scale", StandardScaler())])


def _use_unfitted_cca(bel):
    bel.regression_model = CCA(n_components=2)


def _use_pls(bel):
    bel.regression_model = PLSCanonical(n_components=2)


def _use_observation_override(bel):
    bel.x_observation = np.zeros((1, 2))


def _use_nonfinite_state(bel):
    bel.X_f = bel.X_f.copy()
    bel.X_f[0, 0] = np.nan


def _use_too_many_rows(bel):
    bel.X_f = np.zeros((33, 2))
    bel.Y_f = np.zeros((33, 2))


def _use_unpaired_state(bel):
    bel.Y_f = bel.Y_f[:-1].copy()


def _no_component_fit():
    return (
        mock.patch.object(KernelDensity, "fit", side_effect=AssertionError("KDE fit")),
        mock.patch.object(LinearRegression, "fit", side_effect=AssertionError("linear fit")),
    )


class TestTrainedLiveParity(unittest.TestCase):
    def test_capsule_matches_live_same_profile_bel(self):
        baseline, results = _baseline(), _capsule_results()
        capsule = _exported().capsule
        self.assertEqual(capsule.kinds, baseline.kinds)
        self.assertIn("pdf", capsule.kinds)
        self.assertEqual(baseline.samples.shape, (2, N_SAMPLES, 2))
        self.assertEqual(results.joint.shape, (2, N_SAMPLES, 2))
        np.testing.assert_allclose(results.joint, baseline.samples, rtol=0, atol=ATOL)
        np.testing.assert_allclose(results.restored, baseline.samples, rtol=0, atol=ATOL)

    def test_baseline_consumed_exactly_the_frozen_uniforms_and_restored_global_state(self):
        baseline = _baseline()
        n_pdf = baseline.kinds.count("pdf")
        self.assertEqual(baseline.requests, [(0, 1, N_SAMPLES)] * (len(QUERIES) * n_pdf))
        self.assertEqual(baseline.leftover, 0)
        self.assertTrue(baseline.rng_restored)
        self.assertTrue(baseline.uniform_restored)

    def test_selected_rows_follow_the_full_batch_and_queries_differ(self):
        results = _capsule_results()
        self.assertEqual(results.first.shape, (1, N_SAMPLES, 2))
        self.assertEqual(results.last.shape, (1, N_SAMPLES, 2))
        np.testing.assert_allclose(results.first[0], results.joint[0], rtol=0, atol=ATOL)
        np.testing.assert_allclose(results.last[0], results.joint[1], rtol=0, atol=ATOL)
        self.assertFalse(np.allclose(results.joint[0], results.joint[1], rtol=0, atol=1e-6))

    def test_sampling_leaves_inputs_and_global_rng_alone_and_returns_owned_arrays(self):
        results = _capsule_results()
        self.assertTrue(results.rng_unchanged)
        self.assertTrue(results.inputs_unchanged)
        self.assertTrue(results.owned)
        np.testing.assert_array_equal(results.again, results.joint)
        self.assertEqual(results.joint.dtype, np.float64)


class TestFreshInterpreter(unittest.TestCase):
    def test_nested_interpreter_restores_and_samples_without_any_fit(self):
        child = _child_result()
        samples = np.asarray(_baseline().samples)
        np.testing.assert_allclose(child["full"], samples, rtol=0, atol=ATOL)
        np.testing.assert_allclose(child["selected"], samples[1:2], rtol=0, atol=ATOL)
        self.assertTrue(child["rng"])


class TestExport(unittest.TestCase):
    def test_export_preserves_every_source_array_processor_and_cache(self):
        exported, baseline = _exported(), _baseline()
        self.assertEqual(exported.changes, [])
        self.assertLessEqual(exported.compiler_fits, 2)
        self.assertLessEqual(baseline.reference_fits, 2)
        self.assertLessEqual(exported.compiler_fits + baseline.reference_fits, 4)

    def test_unsupported_profiles_fail_closed_without_fitting(self):
        cases = [
            ("subclass", _use_subclass),
            ("mvn mode", _use_mvn_mode),
            ("tm mode", _use_tm_mode),
            ("nonlinear pre-processing", _use_nonlinear_pre),
            ("whitened PCA", _use_whitened_pca),
            ("non-passthrough post-processing", _use_scaled_post),
            ("unfitted CCA", _use_unfitted_cca),
            ("PLSCanonical", _use_pls),
            ("cached observation override", _use_observation_override),
            ("non-finite paired state", _use_nonfinite_state),
            ("too many training rows", _use_too_many_rows),
            ("unpaired state", _use_unpaired_state),
        ]
        kde_guard, linear_guard = _no_component_fit()
        with kde_guard, linear_guard:
            for label, mutate in cases:
                with self.subTest(label):
                    broken = copy.copy(_trained())
                    mutate(broken)
                    with self.assertRaises(KDEPredictionError):
                        portable_kde.export_kde(broken, BANDWIDTHS)

    def test_invalid_bandwidths_fail_before_fitting(self):
        invalid = [
            [0.5],
            [0.5, 0.0],
            [0.5, -1.0],
            [0.5, np.nan],
            [0.5, np.inf],
            [True, 0.5],
            "0.5",
            [[0.5, 0.5]],
            None,
        ]
        kde_guard, linear_guard = _no_component_fit()
        with kde_guard, linear_guard:
            for bandwidths in invalid:
                with self.subTest(bandwidths=bandwidths), self.assertRaises(KDEPredictionError):
                    portable_kde.export_kde(_trained(), bandwidths)

    def test_errors_are_value_errors(self):
        self.assertTrue(issubclass(KDEPredictionError, ValueError))


class TestNumericLaws(unittest.TestCase):
    def test_bimodal_point_law_matches_source_conventions_and_affine_arithmetic(self):
        canonical, expected = _bimodal_source_expectation()
        results = _bimodal_results()
        self.assertEqual(results.full.shape, (2, N_SAMPLES, 2))
        np.testing.assert_allclose(results.full, expected, rtol=0, atol=ATOL)
        np.testing.assert_allclose(_bimodal_canonical(results.full), canonical, rtol=0, atol=ATOL)
        self.assertEqual(results.selected.shape, (1, N_SAMPLES, 2))
        np.testing.assert_allclose(results.selected[0], results.full[1], rtol=0, atol=ATOL)

    def test_point_mass_branch_is_exact_and_ignores_its_uniform_channel(self):
        recovered = _bimodal_canonical(_bimodal_results().full)
        for case, value in enumerate(BIMODAL_SLOPE * QUERIES[:, 1] + BIMODAL_INTERCEPT):
            np.testing.assert_allclose(recovered[case, :, 1], value, rtol=0, atol=ATOL)

    def test_bimodal_shape_is_kept_not_gaussianized(self):
        # Continuous law: CDF(-1) = 0.25 and CDF(0) = 0.5; mass in [-1.3, -0.7] is 0.3.
        # A moment-matched Gaussian would put the 0.25 quantile near -0.74 and spread the
        # 0.1875/0.3125 quantiles about 0.43 apart.
        z = _bimodal_canonical(_bimodal_results().full)[..., 0]
        self.assertLessEqual(abs(z[0, 2] - (-1.0)), SHAPE_ATOL)  # U = 0.25
        self.assertLessEqual(abs(z[0, 4] - 0.0), SHAPE_ATOL)  # U = 0.5
        self.assertTrue(-1.3 < z[1, 1] < z[1, 2] < -0.7)  # U = 0.1875, 0.3125
        self.assertLess(z[1, 2] - z[1, 1], 0.25)

    def test_inverse_cdf_matches_independent_piecewise_linear_oracle(self):
        support = np.arange(129) / 128.0
        cdf = (support + support**2 / 2.0) / 1.5
        self.assertEqual((cdf[0], cdf[-1]), (0.0, 1.0))
        uniforms = np.concatenate([U.reshape(-1), [0.0, 1.0, cdf[37], cdf[64], 0.999999]])
        got = portable_kde._inverse_cdf(cdf, support, uniforms)
        expected = [_pwl_quantile(cdf, support, u) for u in uniforms]
        np.testing.assert_allclose(got, expected, rtol=0, atol=ATOL)
        self.assertEqual((got[-5], got[-4]), (0.0, 1.0))

    def test_inverse_cdf_plateau_convention_is_rightmost_knot(self):
        got = portable_kde._inverse_cdf(
            [0.0, 0.0, 0.5, 0.5, 1.0], [0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 0.25, 0.5, 0.75, 1.0]
        )
        np.testing.assert_allclose(got, [1.0, 1.5, 3.0, 3.5, 4.0], rtol=0, atol=ATOL)

    def test_inverse_cdf_rejects_invalid_tables(self):
        support = [0.0, 1.0, 2.0]
        cases = {
            "decreasing": ([0.0, 0.6, 0.4], support, [0.5]),
            "above one": ([0.0, 1.2, 1.0], support, [0.5]),
            "not starting at zero": ([0.1, 0.5, 1.0], support, [0.5]),
            "not ending at one": ([0.0, 0.5, 0.9], support, [0.5]),
            "nan": ([0.0, np.nan, 1.0], support, [0.5]),
            "unsorted support": ([0.0, 0.5, 1.0], [0.0, 2.0, 1.0], [0.5]),
            "length mismatch": ([0.0, 1.0], support, [0.5]),
            "uniform above one": ([0.0, 0.5, 1.0], support, [1.5]),
            "uniform below zero": ([0.0, 0.5, 1.0], support, [-0.1]),
            "boolean uniforms": ([0.0, 0.5, 1.0], support, [True]),
            "2D uniforms": ([0.0, 0.5, 1.0], support, [[0.5]]),
        }
        for label, (cdf, axis, uniforms) in cases.items():
            with self.subTest(label), self.assertRaises(KDEPredictionError):
                portable_kde._inverse_cdf(cdf, axis, uniforms)


class TestQueryLaw(unittest.TestCase):
    def test_invalid_queries_are_rejected_before_sampling(self):
        capsule = _exported().capsule
        invalid = {
            "wrong width": (np.ones((2, 3)), U),
            "one dimension": (np.ones(2), U),
            "nan query": (np.array([[np.nan, 0.0], [0.0, 0.0]]), U),
            "boolean query": (np.ones((2, 2), dtype=bool), U),
            "U for one case": (QUERIES, U[:1]),
            "U for one component": (QUERIES, U[..., :1]),
            "U without samples": (QUERIES, U[:, :0]),
            "U above one": (QUERIES, U + 0.5),
            "U below zero": (QUERIES, U - 0.5),
            "U nan": (QUERIES, np.where(U == 0.5, np.nan, U)),
            "U boolean": (QUERIES, U > 0.5),
            "U 2D": (QUERIES, U[0]),
        }
        for label, (queries, uniforms) in invalid.items():
            with self.subTest(label), self.assertRaises(KDEPredictionError):
                capsule.sample(queries, uniforms)
        for obs_n in (True, np.bool_(False), 2, -3, 1.0, "0"):
            with self.subTest(obs_n=obs_n), self.assertRaises(KDEPredictionError):
                capsule.sample(QUERIES, U, obs_n=obs_n)

    def test_query_outside_the_recorded_support_fails_closed(self):
        with self.assertRaisesRegex(KDEPredictionError, "outside"):
            _exported().capsule.sample(np.array([[100.0, 100.0]]), U[:1])

    def test_zero_conditional_law_fails_instead_of_returning_zeros(self):
        density = np.zeros((200, 200))
        density[:, 100:] = 1.0
        capsule = KDEPredictionCapsule.from_state(_pdf_state(density))
        with self.assertRaisesRegex(KDEPredictionError, "zero"):
            capsule.sample(np.array([[-1.9, 0.0]]), U[:1])

    def test_negative_conditional_law_fails_instead_of_being_repaired(self):
        density = np.zeros((200, 200))
        density[100, :] = 1.0
        capsule = KDEPredictionCapsule.from_state(_pdf_state(density))
        with self.assertRaisesRegex(KDEPredictionError, "negative"):
            capsule.sample(np.array([[0.0, 0.0]]), U[:1])


class TestWireFormat(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(portable_kde.SCHEMA, "skbel.kde-prediction-capsule/v1")
        self.assertEqual(
            portable_kde.CONVENTION,
            "gaussian-euclidean-fixed-bw-grid200-cut1-count-pixels-cubic-constant0-"
            "prefilter-129-romberg-cutoffs-linear-cdf-inverse/v1",
        )
        self.assertEqual(portable_kde.MAX_BYTES, 4 * 1024 * 1024)

    def test_exported_bytes_are_deterministic_strict_and_data_only(self):
        capsule = _exported().capsule
        data = capsule.to_bytes()
        self.assertEqual(data, capsule.to_bytes())
        self.assertLess(len(data), portable_kde.MAX_BYTES)
        document = json.loads(data)
        self.assertEqual(
            set(document),
            {"schema", "convention", "dimensions", "A_x", "b_x", "A_y", "b_y", "components"},
        )
        self.assertEqual(document["dimensions"], {"predictor": 2, "canonical": 2, "target": 2})
        self.assertEqual(tuple(item["kind"] for item in document["components"]), capsule.kinds)
        expected = 4 + 2 + 4 + 2
        for item in document["components"]:
            if item["kind"] == "linear":
                self.assertEqual(set(item), {"kind", "slope", "intercept"})
                expected += 2
            else:
                self.assertEqual(set(item), {"kind", "bandwidth", "x_axis", "y_axis", "density"})
                self.assertEqual(item["bandwidth"], 0.5)
                expected += 1 + 200 + 200 + 200 * 200
        names = ("A_x", "b_x", "A_y", "b_y", "components")
        self.assertEqual(sum(_count_numbers(document[name]) for name in names), expected)
        stripped = data.replace(portable_kde.SCHEMA.encode(), b"").replace(
            portable_kde.CONVENTION.encode(), b""
        )
        for token in (b"sklearn", b"numpy", b"joblib", b"pickle", b"BEL", b"/", b"seed"):
            self.assertNotIn(token, stripped)

    def test_roundtrip_is_exact_and_digest_is_external_integrity_only(self):
        capsule = _exported().capsule
        data = capsule.to_bytes()
        digest = hashlib.sha256(data).hexdigest()
        self.assertEqual(capsule.sha256(), digest)
        self.assertEqual(_capsule_results().restored_bytes, data)
        self.assertEqual(KDEPredictionCapsule.from_state(capsule.state()).to_bytes(), data)
        KDEPredictionCapsule.from_bytes(data, expected_sha256=digest.upper())
        for bad in ("0" * 64, digest[:-1], digest + "0", "g" * 64, digest.encode(), 12):
            with self.subTest(bad=bad), self.assertRaises(KDEPredictionError):
                KDEPredictionCapsule.from_bytes(data, expected_sha256=bad)
        with self.assertRaisesRegex(KDEPredictionError, "SHA-256"):
            KDEPredictionCapsule.from_bytes(b"not json", expected_sha256=digest)
        # A rehashed, still valid alteration is accepted: SHA-256 is not authenticity.
        self.assertIn(b'"slope": 2.0', _WIRE_BYTES)
        altered = _WIRE_BYTES.replace(b'"slope": 2.0', b'"slope": 3.0', 1)
        original_digest = hashlib.sha256(_WIRE_BYTES).hexdigest()
        with self.assertRaises(KDEPredictionError):
            KDEPredictionCapsule.from_bytes(altered, expected_sha256=original_digest)
        rehashed = KDEPredictionCapsule.from_bytes(
            altered, expected_sha256=hashlib.sha256(altered).hexdigest()
        )
        self.assertEqual(rehashed.state()["components"][0]["slope"], 3.0)

    def test_rejects_hostile_and_malformed_documents(self):
        self.assertIn(b"0.25", _WIRE_BYTES)
        hook = "__import__('builtins').setattr(__import__('sys'), '_kde_executed', True)"
        cases = {
            "empty": b"",
            "not json": b"not json",
            "utf8 bom": b"\xef\xbb\xbf" + _WIRE_BYTES,
            "invalid utf8": b"\xff\xfe" + _WIRE_BYTES,
            "trailing data": _WIRE_BYTES + b"{}",
            "top-level list": b"[]",
            "top-level number": b"1",
            "duplicate key": _WIRE_BYTES.replace(b"{", b'{"schema": "x", ', 1),
            "missing key": _without("A_x"),
            "missing components": _without("components"),
            "unknown key": _with(producer="sklearn.neighbors.KernelDensity"),
            "unknown version": _with(schema="skbel.kde-prediction-capsule/v2"),
            "linear-mvn schema": _with(schema="skbel.linear-mvn-capsule/v1"),
            "schema not a string": _with(schema=1),
            "unknown convention": _with(convention="other"),
            "dimension missing": _with(dimensions={"predictor": 2, "canonical": 2}),
            "dimension extra": _with(dimensions={**LINEAR_DIMS, "extra": 1}),
            "dimension bool": _with(dimensions={**LINEAR_DIMS, "predictor": True}),
            "dimension float": _with(dimensions={**LINEAR_DIMS, "predictor": 2.0}),
            "dimension mismatch": _with(dimensions={**LINEAR_DIMS, "predictor": 3}),
            "canonical three": _with(dimensions={"predictor": 3, "canonical": 3, "target": 3}),
            "target five": _with(dimensions={**LINEAR_DIMS, "target": 5}),
            "components not a list": _with(components={"0": {"kind": "linear"}}),
            "one component": _with(components=LINEAR_COMPONENTS[:1]),
            "three components": _with(components=[*LINEAR_COMPONENTS, LINEAR_COMPONENTS[0]]),
            "component not an object": _with(components=[1.0, 2.0]),
            "unknown kind": _linear_component(0, kind="tm"),
            "kind not a string": _linear_component(0, kind=1),
            "linear extra key": _linear_component(0, bandwidth=0.5),
            "linear missing key": _linear_component(1, drop="intercept"),
            "linear bool slope": _linear_component(0, slope=True),
            "linear string slope": _linear_component(0, slope="2.0"),
            "linear null intercept": _linear_component(0, intercept=None),
            "linear list slope": _linear_component(0, slope=[2.0]),
            "bool entry": _with(A_x=[[True, 0.5], [-0.5, 1.0]]),
            "string entry": _with(b_x=["0.25", -0.5]),
            "null entry": _with(b_x=[None, -0.5]),
            "nested entry": _with(b_x=[[0.25], -0.5]),
            "object entry": _with(b_x=[{}, -0.5]),
            "executable string entry": _with(b_y=[hook, -0.25, 1.0]),
            "executable schema": _with(schema=hook),
            "huge integer": _with(b_x=[10**400, -0.5]),
            "enormous integer literal": _WIRE_BYTES.replace(b"0.25", b"9" * 5000, 1),
            "ragged matrix": _with(A_x=[[1.0, 0.5], [-0.5]]),
            "flat matrix": _with(A_x=[1.0, 0.5, -0.5, 1.0]),
            "short vector": _with(b_x=[0.25]),
            "scalar vector": _with(b_x=0.25),
            "too deep": _with(b_x=_nested(14, 0.25)),
            "NaN token": _WIRE_BYTES.replace(b"0.25", b"NaN", 1),
            "Infinity token": _WIRE_BYTES.replace(b"0.25", b"Infinity", 1),
            "negative Infinity token": _WIRE_BYTES.replace(b"0.25", b"-Infinity", 1),
            "overflowing float": _WIRE_BYTES.replace(b"0.25", b"1e999", 1),
            "bracket bomb": b"[" * 100_000 + b"]" * 100_000,
            "unterminated bracket bomb": b"[" * 100_000,
            "brace bomb": b'{"a":' * 1000 + b"1" + b"}" * 1000,
            "oversize blanks": b" " * (portable_kde.MAX_BYTES + 1),
            "oversize padded document": _WIRE_BYTES + b" " * portable_kde.MAX_BYTES,
        }
        for label, data in cases.items():
            with self.subTest(label), self.assertRaises(KDEPredictionError):
                KDEPredictionCapsule.from_bytes(data)
        self.assertFalse(hasattr(sys, "_kde_executed"))

    def test_inconsistent_density_records_are_rejected_by_construction_and_load(self):
        density = _bimodal_density()
        negative, below, short = density.copy(), density.copy(), density[:-1]
        negative[3, 4] = -0.1
        below[3, 4] = 1e-9
        uneven = AXIS.copy()
        uneven[100] += 1e-3
        cases = {
            "negative density": _pdf_with(density=negative.tolist()),
            "density below cutoff": _pdf_with(density=below.tolist()),
            "zero density": _pdf_with(density=np.zeros((200, 200)).tolist()),
            "short density": _pdf_with(density=short.tolist()),
            "reversed x axis": _pdf_with(x_axis=AXIS[::-1].tolist()),
            "uneven y axis": _pdf_with(y_axis=uneven.tolist()),
            "short axis": _pdf_with(x_axis=AXIS[:-1].tolist()),
            "zero bandwidth": _pdf_with(bandwidth=0.0),
            "negative bandwidth": _pdf_with(bandwidth=-0.5),
            "bool bandwidth": _pdf_with(bandwidth=True),
            "missing bandwidth": _pdf_with(drop="bandwidth"),
            "extra slope": _pdf_with(slope=1.0),
        }
        for label, state in cases.items():
            data = _encode(state)
            digest = hashlib.sha256(data).hexdigest()
            with self.subTest(label, route="bytes"), self.assertRaises(KDEPredictionError):
                KDEPredictionCapsule.from_bytes(data, expected_sha256=digest)
            with self.subTest(label, route="state"), self.assertRaises(KDEPredictionError):
                KDEPredictionCapsule.from_state(state)

    def test_rejects_non_bytes_inputs(self):
        for label, value in (
            ("str", _WIRE_BYTES.decode("utf-8")),
            ("bytearray", bytearray(_WIRE_BYTES)),
            ("memoryview", memoryview(_WIRE_BYTES)),
            ("none", None),
        ):
            with self.subTest(label), self.assertRaises(KDEPredictionError):
                KDEPredictionCapsule.from_bytes(value)

    def test_size_limit_is_inclusive(self):
        padded = _WIRE_BYTES + b" " * (portable_kde.MAX_BYTES - len(_WIRE_BYTES))
        self.assertEqual(len(padded), portable_kde.MAX_BYTES)
        restored = KDEPredictionCapsule.from_bytes(padded)
        dims = (restored.predictor_dim, restored.canonical_dim, restored.target_dim)
        self.assertEqual(dims, (2, 2, 3))
        self.assertEqual(restored.kinds, ("linear", "linear"))


class TestStateOwnership(unittest.TestCase):
    def test_construction_and_accessors_copy_state(self):
        state = _bimodal_state()
        arrays = {name: np.array(state[name]) for name in ("A_x", "b_x", "A_y", "b_y")}
        density = np.array(state["components"][0]["density"])
        state.update(arrays)
        state["components"][0]["density"] = density
        built = KDEPredictionCapsule.from_state(state)
        before = built.to_bytes()
        self.assertEqual(before, _encode_sorted(_bimodal_state()))
        for array in arrays.values():
            array[...] = 7.0
        density[...] = 7.0
        self.assertEqual(built.to_bytes(), before)
        copied = built.state()
        copied["A_x"][...] = 7.0
        copied["components"][0]["density"][...] = 7.0
        copied["components"][1]["slope"] = 7.0
        self.assertEqual(built.to_bytes(), before)

    def test_direct_construction_validates_like_restoration(self):
        def state_with(**overrides):
            state = _linear_state()
            state.update(overrides)
            return state

        bad = {
            "not a dict": [],
            "bool array": state_with(b_x=np.array([True, False])),
            "mixed boolean list": state_with(b_x=[True, 0.0]),
            "numpy boolean in list": state_with(b_x=[np.True_, 0.0]),
            "complex array": state_with(b_x=np.array([1.0 + 0j, 0.5])),
            "object array": state_with(b_x=np.array([0.25, -0.5], dtype=object)),
            "masked array": state_with(b_x=np.ma.masked_array([0.25, -0.5], mask=[False, True])),
            "nan array": state_with(b_x=np.array([np.nan, 0.5])),
            "wrong shape": state_with(A_x=np.ones((3, 2))),
            "numpy integer dimension": state_with(
                dimensions={"predictor": np.int64(2), "canonical": 2, "target": 3}
            ),
        }
        for label, state in bad.items():
            with self.subTest(label), self.assertRaises(KDEPredictionError):
                KDEPredictionCapsule.from_state(state)


def _encode_sorted(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class TestSharedFrozenState(unittest.TestCase):
    def test_emit_frozen_numeric_state_when_requested(self):
        target = os.environ.get(ARTIFACT_ENV)
        if not target:
            self.skipTest(f"set {ARTIFACT_ENV} to a directory to emit the frozen state")
        capsule, baseline = _exported().capsule, _baseline()
        fixture = {
            "queries": QUERIES.tolist(),
            "uniforms": U.tolist(),
            "bandwidths": BANDWIDTHS,
            "seed": SEED,
            "kinds": list(capsule.kinds),
            "trained_sha256": capsule.sha256(),
            "bimodal_sha256": hashlib.sha256(_bimodal_bytes()).hexdigest(),
            "baseline_samples": np.asarray(baseline.samples).tolist(),
            "capsule_samples": _capsule_results().joint.tolist(),
        }
        documents = {
            "trained_capsule.json": capsule.to_bytes(),
            "bimodal_capsule.json": _bimodal_bytes(),
            "frozen_fixture.json": _encode_sorted(fixture),
        }
        total = sum(len(data) for data in documents.values())
        self.assertLessEqual(total, portable_kde.MAX_BYTES)
        directory = Path(target)
        directory.mkdir(parents=True, exist_ok=True)
        for name, data in documents.items():
            (directory / name).write_bytes(data)


class TestZCallBudget(unittest.TestCase):
    def test_finite_call_counts_stay_within_test_limits(self):
        self.assertEqual(_CALLS["grid_search"], 0)
        self.assertEqual(_CALLS["transport_map"], 0)
        self.assertEqual(_CALLS["pca_fit"], 0)
        for key in ("bel_fit", "cca_fit", "scaler_fit", "predict", "random_sample", "public_map"):
            with self.subTest(key):
                self.assertLessEqual(_CALLS[key], TEST_LIMITS[key])
        self.assertLessEqual(_component_fits(), TEST_LIMITS["component_fit"])
        self.assertLessEqual(_CALLS["export"], TEST_LIMITS["export"])
        valid = _CALLS["sample"] - _CALLS["sample_invalid"] + CHILD_SAMPLE_CALLS
        self.assertLessEqual(valid, TEST_LIMITS["valid_sample"])


if __name__ == "__main__":
    unittest.main()
