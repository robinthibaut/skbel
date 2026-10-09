#  Copyright (c) 2026. Robin Thibaut, Ghent University

"""Tests for the optional neural posterior ensemble and quantile recalibrator.

The NumPy paths (recalibration, sampling of precomputed mixtures, estimator
parameters, persistence checks) run everywhere; training tests use tiny
networks and skip when PyTorch is not installed.
"""

import ast
import inspect
import json
import math
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone

from skbel.design import SimulationBank, fit_designs
from skbel.evaluation import decision_risks, event_probabilities, score_samples
from skbel.neural import posterior as npost

SRC = Path(npost.__file__).resolve()
EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "neural_posterior.py"
TINY = dict(
    n_components=3,
    hidden=(8, 8, 8),
    n_pca=4,
    batch_size=16,
    max_epochs=6,
    n_validation=32,
    patience=2,
)
SEEDS = {"pca": 11, "init": 12, "shuffle": 13}

BLOCK_TORCH = (
    "import sys\n"
    "class _Block:\n"
    "    def find_spec(self, name, path=None, target=None):\n"
    "        if name == 'torch' or name.startswith('torch.'):\n"
    "            raise ModuleNotFoundError('torch is blocked for this test')\n"
    "sys.meta_path.insert(0, _Block())\n"
)


def tiny(covariance):
    return npost.MixtureSettings(covariance=covariance, **TINY)


def toy_rows(n=96, d=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 10))
    z = x[:, :d] * 0.5 + 0.1 * rng.standard_normal((n, d))
    return x, z


def _torch_or_skip():
    return pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Module boundaries and the optional extra
# ---------------------------------------------------------------------------


def test_module_imports_only_stdlib_numpy_and_sklearn_base_at_top_level():
    tree = ast.parse(SRC.read_text())
    top = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "no relative imports in the generic module"
            top.append(node.module)
    allowed = {"__future__", "json", "math", "time", "dataclasses", "pathlib", "numpy"}
    assert set(top) <= allowed | {"sklearn.base"}


def test_core_and_numpy_paths_work_with_torch_blocked():
    code = BLOCK_TORCH + (
        "import numpy as np\n"
        "import skbel\n"
        "from skbel import design, evaluation, metrics\n"
        "from sklearn.base import clone\n"
        "from skbel import neural as m\n"
        "rng = np.random.default_rng(0)\n"
        "d = rng.standard_normal((150, 31, 2)); t = rng.standard_normal((150, 2))\n"
        "g = np.zeros((150, 2), dtype=int)\n"
        "r = m.QuantileRecalibrator().fit(d, t, g, seed=3)\n"
        "out = r.transform(d, g)\n"
        "mix = {'weights': np.full((2, 2), .5), 'means': np.zeros((2, 2, 2)), "
        "'scales': np.ones((2, 2, 2))}\n"
        "e = m.sample_ensemble_latent([mix, mix], [5, 6], 4, 1)\n"
        "evaluation.score_samples(out, t, levels=[0.5])\n"
        "est = clone(m.NeuralPosterior(n_members=2))\n"
        "try:\n"
        "    est.fit(np.zeros((40, 3)), np.zeros((40, 1)))\n"
        "except ImportError as exc:\n"
        "    assert \"pip install 'skbel[neural]'\" in str(exc), exc\n"
        "else:\n"
        "    raise AssertionError('fit without torch must raise ImportError')\n"
        "assert out.shape == d.shape and e['latent'].shape == (2, 4, 2)\n"
        "assert 'torch' not in sys.modules\n"
        "print('ok')\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "ok"


def test_settings_defaults_and_validation():
    s = npost.MixtureSettings()
    assert (s.covariance, s.n_components, s.hidden, s.n_pca, s.learning_rate, s.batch_size) == (
        "full",
        20,
        (128, 128, 128),
        64,
        1e-3,
        512,
    )
    assert (s.max_epochs, s.n_validation, s.patience, s.min_improvement) == (200, 1024, 20, 1e-4)
    assert s.log_scale_bounds == (-7.0, 3.0)
    assert npost.MixtureSettings.from_dict(json.loads(json.dumps(s.to_dict()))) == s
    for bad in (
        {"covariance": "banded"},
        {"n_components": 0},
        {"n_pca": True},
        {"hidden": ()},
        {"hidden": (8, 0)},
        {"log_scale_bounds": (3.0, -7.0)},
        {"learning_rate": math.nan},
        {"min_improvement": -1.0},
    ):
        with pytest.raises(ValueError):
            npost.MixtureSettings(**bad)
    with pytest.raises(ValueError):
        npost.MixtureSettings.from_dict({**s.to_dict(), "extra": 1})
    assert npost.n_outputs(5, npost.MixtureSettings(covariance="full")) == 20 * (1 + 5 + 5 + 10)
    assert npost.n_outputs(8, npost.MixtureSettings(covariance="diagonal")) == 20 * (1 + 16)


# ---------------------------------------------------------------------------
# Estimator parameters and seeds (no training)
# ---------------------------------------------------------------------------


def test_estimator_defaults_match_settings_and_clone_keeps_parameters():
    est = npost.NeuralPosterior()
    assert est.settings() == npost.MixtureSettings()
    assert est.n_members == 5 and est.random_state == 0 and est.member_seeds is None
    seeds = [{"pca": 1, "init": 2, "shuffle": 3}]
    est = npost.NeuralPosterior(1, covariance="diagonal", member_seeds=seeds, **TINY)
    twin = clone(est)
    assert twin is not est and twin.get_params() == est.get_params()
    assert twin.member_seeds == seeds and twin.member_seeds is not seeds
    assert not hasattr(twin, "ensemble_")


def test_member_seed_rule_is_local_prefix_stable_and_distinct():
    seeds = npost.member_seeds(5, 4)
    for m, s in enumerate(seeds):
        for i, name in enumerate(npost.FIT_SEEDS):
            seq = np.random.SeedSequence(5, spawn_key=(m, i))
            expect = (
                int(seq.generate_state(1, np.uint32)[0])
                if name == "pca"
                else int(seq.generate_state(1, np.uint64)[0] >> np.uint64(1))
            )
            assert s[name] == expect
        assert s["pca"] < 2**32 and s["init"] < 2**63 and s["shuffle"] < 2**63
    assert npost.member_seeds(5, 2) == seeds[:2]
    values = [v for s in seeds for v in s.values()]
    assert len(set(values)) == len(values)
    assert npost.member_seeds(6, 4) != seeds
    assert npost.NeuralPosterior(4, random_state=5)._seeds() == seeds


@pytest.mark.parametrize(
    "params",
    [
        {"n_members": 0},
        {"n_members": True},
        {"random_state": None},
        {"random_state": -1},
        {"n_members": 2, "member_seeds": [SEEDS]},
        {"n_members": 1, "member_seeds": [{"pca": 1, "init": 2}]},
        {"n_members": 1, "member_seeds": [{**SEEDS, "pca": 2**32}]},
        {"n_members": 1, "member_seeds": [{**SEEDS, "init": -1}]},
        {"covariance": "banded"},
        {"hidden": ()},
    ],
)
def test_estimator_rejects_bad_parameters_before_training(params):
    x, z = toy_rows()
    with pytest.raises(ValueError):
        npost.NeuralPosterior(**params).fit(x, z)


def test_unfitted_estimator_refuses_sampling_and_saving(tmp_path):
    est = npost.NeuralPosterior(1)
    with pytest.raises(RuntimeError):
        est.sample(np.zeros((2, 3)), 4, case_ids=[0, 1], seed=0)
    with pytest.raises(RuntimeError):
        est.save(tmp_path / "e")


# ---------------------------------------------------------------------------
# Recalibration math (NumPy only)
# ---------------------------------------------------------------------------


def test_pit_definition_ties_and_open_interval():
    draws = np.array([[[0.0], [1.0], [1.0], [2.0]], [[5.0], [5.0], [5.0], [5.0]]])
    truth = np.array([[1.0], [5.0]])
    u = npost.pit_uniforms([7, 8], 1, seed=4, n_draws=4)
    assert np.all((u > 0) & (u < 1))
    pit = npost.randomized_pit(draws, truth, seed=4, case_ids=[7, 8])
    assert pit[0, 0] == pytest.approx((1 + u[0, 0] * 3) / 5, rel=1e-15)
    assert pit[1, 0] == pytest.approx((0 + u[1, 0] * 5) / 5, rel=1e-15)
    assert np.all((pit > 0) & (pit < 1))
    # keyed by case ID, not position
    swapped = npost.randomized_pit(draws[::-1], truth[::-1], seed=4, case_ids=[8, 7])
    assert np.array_equal(swapped, pit[::-1])


def test_pit_grid_indices_follow_the_declared_stream():
    k = npost.pit_grid(31)
    j = npost.pit_grid_indices([3, 8], 2, 7, 31)
    for i, cid in enumerate([3, 8]):
        for t in range(2):
            rng = np.random.default_rng(np.random.SeedSequence(7, spawn_key=(cid, t)))
            assert j[i, t] == rng.integers(0, k)


@pytest.mark.parametrize("m", [1, 15, 1023, 4095])
def test_pit_exact_grid_endpoints_stay_strictly_open(m):
    k = npost.pit_grid(m)
    assert 2 * k * (m + 1) < 2**51
    for below, ties in ((0, 0), (m, 0), (0, m)):
        for j in (0, k - 1):
            u = npost.pit_from_counts(below, ties, j, m)
            assert 0.0 < float(u) < 1.0
    assert float(npost.pit_from_counts(m, 0, k - 1, m)) == (2 * k * (m + 1) - 1) / (2 * k * (m + 1))
    assert float(npost.pit_from_counts(0, 0, 0, m)) == 1 / (2 * k * (m + 1))
    with pytest.raises(ValueError):
        npost.pit_from_counts(m, 1, 0, m)
    with pytest.raises(ValueError):
        npost.pit_from_counts(0, 0, k, m)
    with pytest.raises(ValueError):
        npost.pit_from_counts(-1, 0, 0, m)
    draws = np.zeros((3, m, 1))
    truth = np.array([[-1.0], [1.0], [0.0]])
    pit = npost.randomized_pit(draws, truth, seed=11, case_ids=[1, 2, 3])
    assert np.all((pit > 0) & (pit < 1))


def test_pit_grid_rejects_bad_draw_counts():
    for bad in (0, -1, True, 2.0):
        with pytest.raises(ValueError):
            npost.pit_grid(bad)
    with pytest.raises(ValueError):
        npost.pit_grid(2**51)


def test_mid_cdf_knots_endpoints_ties_and_strict_monotonicity():
    x, y = npost.mid_cdf_knots([0.2, 0.2, 0.5, 0.9])
    assert x.tolist() == [0.0, 0.2, 0.5, 0.9, 1.0]
    assert y.tolist() == [0.0, 0.25, 0.625, 0.875, 1.0]
    for bad in ([0.0, 0.5], [0.5, 1.0], [], [math.nan]):
        with pytest.raises(ValueError):
            npost.mid_cdf_knots(bad)


def test_midrank_and_empirical_quantile_identity_with_ties():
    draws = np.array([[[3.0, 1.0], [1.0, 1.0], [2.0, 4.0], [1.0, 1.0], [5.0, 0.0]]])
    p = npost.midrank_probabilities(draws)
    assert p[0, :, 0].tolist() == [0.7, 0.2, 0.5, 0.2, 0.9]
    s = np.sort(draws[0].T, axis=1)
    back = npost.empirical_quantile(s, p[0].T)
    assert np.array_equal(back, draws[0].T)
    assert npost.empirical_quantile(s[:1], np.array([[0.0, 1.0]])).tolist() == [[1.0, 5.0]]
    with pytest.raises(ValueError):
        npost.empirical_quantile(s, np.full((2, 5), 1.5))


def identity_recalibrator(n_targets, labels=(0, 1, 2)):
    r = npost.QuantileRecalibrator()
    r.maps = {
        (t, g): {"n": 100, "complete": True, "x": np.array([0.0, 1.0]), "y": np.array([0.0, 1.0])}
        for t in range(n_targets)
        for g in labels
    }
    r.n_targets, r.labels, r.min_cases, r.seed = n_targets, list(labels), 100, 0
    return r


def test_identity_map_returns_draws_exactly_including_ties():
    rng = np.random.default_rng(1)
    draws = np.round(rng.standard_normal((20, 63, 3)), 1)
    groups = rng.integers(0, 3, size=(20, 3))
    out = identity_recalibrator(3).transform(draws, groups)
    assert np.array_equal(out, draws)


def test_recalibration_orientation_and_within_case_monotonicity():
    rng = np.random.default_rng(2)
    n, m = 400, 127
    mu = rng.standard_normal(n)
    truth = (mu + rng.standard_normal(n))[:, None]
    groups = np.zeros((n, 1), dtype=int)
    narrow = mu[:, None, None] + 0.4 * rng.standard_normal((n, m, 1))
    wide = mu[:, None, None] + 2.5 * rng.standard_normal((n, m, 1))
    for draws, wider in ((narrow, True), (wide, False)):
        r = npost.QuantileRecalibrator().fit(draws, truth, groups, seed=5)
        out = r.transform(draws, groups)
        ratio = out.std(axis=1).mean() / draws.std(axis=1).mean()
        assert (ratio > 1.0) if wider else (ratio < 1.0)
        order = np.argsort(draws[:, :, 0], axis=1, kind="stable")
        assert np.all(np.diff(np.take_along_axis(out[:, :, 0], order, 1), axis=1) >= 0)
        assert all(np.isin(out[i, :, 0], draws[i, :, 0]).all() for i in range(5))
        x, y = r.maps[(0, 0)]["x"], r.maps[(0, 0)]["y"]
        assert (x[0], y[0], x[-1], y[-1]) == (0.0, 0.0, 1.0, 1.0)


def test_joint_draw_indices_are_shared_across_targets():
    rng = np.random.default_rng(12)
    n, m = 200, 31
    draws = rng.standard_normal((n, m, 2))
    truth = rng.standard_normal((n, 2))
    groups = np.zeros((n, 2), dtype=int)
    out = npost.QuantileRecalibrator().fit(draws, truth, groups, seed=1).transform(draws, groups)
    for t in range(2):
        # each target is a monotone map of its own column, so draw j stays draw j
        order = np.argsort(draws[:, :, t], axis=1)
        assert np.all(np.diff(np.take_along_axis(out[:, :, t], order, 1), axis=1) >= 0)


def test_incomplete_groups_are_visible_and_never_pooled():
    rng = np.random.default_rng(3)
    draws = rng.standard_normal((150, 31, 2))
    truth = rng.standard_normal((150, 2))
    groups = np.zeros((150, 2), dtype=int)
    groups[:30, 0] = 2
    r = npost.QuantileRecalibrator().fit(draws, truth, groups, seed=1, labels=(0, 1, 2))
    assert not r.complete
    assert {"target": 0, "group": 2, "n": 30} in r.incomplete()
    assert {"target": 1, "group": 1, "n": 0} in r.incomplete()
    assert r.maps[(0, 0)]["n"] == 120 and r.maps[(1, 0)]["n"] == 150
    with pytest.raises(ValueError, match="INCOMPLETE"):
        r.transform(draws, groups)
    with pytest.raises(ValueError, match="INCOMPLETE"):
        r.map_values(0, 2, [0.5])
    assert r.transform(draws, np.zeros_like(groups)).shape == draws.shape


def test_recalibrator_rejects_malformed_inputs():
    rng = np.random.default_rng(7)
    draws = rng.standard_normal((120, 15, 2))
    truth = rng.standard_normal((120, 2))
    groups = np.zeros((120, 2), dtype=int)
    r = npost.QuantileRecalibrator()
    with pytest.raises(RuntimeError):
        r.transform(draws, groups)
    bad = draws.copy()
    bad[0, 0, 0] = math.nan
    bad_truth = truth.copy()
    bad_truth[0, 0] = math.inf
    for args in (
        (bad, truth, groups),
        (draws, bad_truth, groups),
        (draws, truth[:, :1], groups),
        (draws, truth, groups.astype(float)),
        (draws, truth, groups[:, :1]),
        (draws[:, :, 0], truth, groups),
    ):
        with pytest.raises(ValueError):
            npost.QuantileRecalibrator().fit(*args, seed=0)
    for kwargs in (
        {"seed": -1},
        {"seed": 0, "min_cases": 0},
        {"seed": 0, "labels": (1,)},
        {"seed": 0, "labels": (0, 0)},
        {"seed": 0, "case_ids": np.zeros(120, dtype=int)},
    ):
        with pytest.raises(ValueError):
            npost.QuantileRecalibrator().fit(draws, truth, groups, **kwargs)
    r.fit(draws, truth, groups, seed=0)
    with pytest.raises(ValueError, match="no map"):
        r.transform(draws, groups + 1)
    with pytest.raises(ValueError):
        r.transform(draws[:, :, :1], groups[:, :1])
    with pytest.raises(ValueError):
        r.transform(bad, groups)


def test_recalibrator_application_takes_no_truth_and_persists_safely(tmp_path):
    rng = np.random.default_rng(4)
    draws = rng.standard_normal((300, 31, 2))
    truth = rng.standard_normal((300, 2))
    groups = np.repeat([[0, 1], [1, 0]], 150, axis=0)
    r = npost.QuantileRecalibrator().fit(draws, truth, groups, seed=9, case_ids=np.arange(300) + 7)
    assert r.complete
    assert list(inspect.signature(npost.QuantileRecalibrator.transform).parameters) == [
        "self",
        "draws",
        "groups",
    ]
    new = rng.standard_normal((10, 31, 2))
    routes = rng.integers(0, 2, size=(10, 2))
    out = r.transform(new, routes)
    r.save(tmp_path / "maps.npz")
    with pytest.raises(FileExistsError):
        r.save(tmp_path / "maps.npz")
    back = npost.QuantileRecalibrator.load(tmp_path / "maps.npz")
    assert np.array_equal(back.transform(new, routes), out)
    assert np.array_equal(back.pit_, r.pit_)
    with np.load(tmp_path / "maps.npz", allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    assert str(arrays["format"]) == npost.RECALIBRATOR_FORMAT
    # a tampered, non-monotone map is refused on load
    arrays["map/0/0/y"] = arrays["map/0/0/y"][::-1].copy()
    np.savez(tmp_path / "tampered.npz", **arrays)
    with pytest.raises(ValueError, match="strictly increasing"):
        npost.QuantileRecalibrator.load(tmp_path / "tampered.npz")
    np.savez(tmp_path / "other.npz", format=np.array("something/1"))
    with pytest.raises(ValueError):
        npost.QuantileRecalibrator.load(tmp_path / "other.npz")


# ---------------------------------------------------------------------------
# Ensemble sampling of precomputed mixtures (NumPy only)
# ---------------------------------------------------------------------------


def point_mixture(offset, n=3, k=2, d=2, kind="chol"):
    w = np.tile(np.array([0.25, 0.75]), (n, 1))
    mu = offset + np.arange(k)[None, :, None] * 10.0 + np.zeros((n, k, d))
    if kind == "chol":
        shape = np.tile(1e-9 * np.eye(d), (n, k, 1, 1))
    else:
        shape = np.full((n, k, d), 1e-9)
    return {"weights": w, "means": mu, kind: shape}


@pytest.mark.parametrize("kind", ["chol", "scales"])
def test_ensemble_equal_weight_member_and_component_labels(kind):
    members = [point_mixture(100.0 * j, kind=kind) for j in range(5)]
    out = npost.sample_ensemble_latent(members, [3, 9, 4], 20000, seed=7)
    member, comp, z = out["member"], out["component"], out["latent"]
    freq = np.bincount(member.ravel(), minlength=5) / member.size
    assert np.allclose(freq, 0.2, atol=0.01)
    assert abs(np.mean(comp) - 0.75) < 0.01
    expect = 100.0 * member + 10.0 * comp
    assert np.allclose(z[..., 0], expect, atol=1e-6) and np.allclose(z[..., 1], expect, atol=1e-6)


def test_ensemble_draws_follow_the_declared_per_case_stream():
    rng = np.random.default_rng(8)
    members = []
    for _ in range(3):
        a = rng.standard_normal((2, 4, 3, 3))
        members.append(
            {
                "weights": rng.dirichlet(np.ones(4), size=2),
                "means": rng.standard_normal((2, 4, 3)),
                "chol": np.tril(a) + 2 * np.eye(3),
            }
        )
    out = npost.sample_ensemble_latent(members, [4, 9], 6, seed=21)
    for i, cid in enumerate([4, 9]):
        g = np.random.default_rng(np.random.SeedSequence(21, spawn_key=(cid,)))
        m = g.integers(0, 3, size=6)
        u = g.random(6)
        eps = g.standard_normal((6, 3))
        for j in range(6):
            w = members[m[j]]["weights"][i]
            c = int(np.searchsorted(np.cumsum(w), u[j] * np.cumsum(w)[-1], side="right"))
            vec = members[m[j]]["means"][i, c] + members[m[j]]["chol"][i, c] @ eps[j]
            assert out["member"][i, j] == m[j] and out["component"][i, j] == c
            assert np.allclose(out["latent"][i, j], vec, rtol=0, atol=1e-12)


def test_ensemble_case_order_subset_seed_and_global_rng():
    rng = np.random.default_rng(5)
    members = []
    for _ in range(5):
        a = rng.standard_normal((4, 3, 2, 2))
        members.append(
            {
                "weights": rng.dirichlet(np.ones(3), size=4),
                "means": rng.standard_normal((4, 3, 2)),
                "chol": np.tril(a) + 3 * np.eye(2),
            }
        )
    ids = np.array([10, 2, 33, 4])
    state = npost._global_rng_state()
    full = npost.sample_ensemble_latent(members, ids, 50, seed=1)
    assert npost._global_rng_state() == state
    perm = [2, 0, 3, 1]
    shuffled = npost.sample_ensemble_latent(
        [{k: v[perm] for k, v in m.items()} for m in members], ids[perm], 50, seed=1
    )
    for key in ("latent", "member", "component"):
        assert np.array_equal(shuffled[key], full[key][perm])
    part = npost.sample_ensemble_latent(
        [{k: v[:2] for k, v in m.items()} for m in members], ids[:2], 50, 1
    )
    assert np.array_equal(part["latent"], full["latent"][:2])
    other = npost.sample_ensemble_latent(members, ids, 50, seed=2)
    assert not np.array_equal(other["member"], full["member"])


def test_ensemble_sampler_rejects_malformed_inputs():
    good = point_mixture(0.0, n=4, k=2)
    ids = [1, 2, 3, 4]
    zero = {**good, "weights": np.zeros((4, 2))}
    negative = {**good, "weights": np.tile([-0.5, 1.5], (4, 1))}
    nan_means = {**good, "means": np.full((4, 2, 2), math.nan)}
    both = {**good, "scales": np.ones((4, 2, 2))}
    for members, case_ids, n_draws, seed in (
        ([], ids, 5, 1),
        ([good], [1, 1, 2, 3], 5, 1),
        ([good], [1, 2, 3], 5, 1),
        ([good], [-1, 2, 3, 4], 5, 1),
        ([good], np.array([1.0, 2.0, 3.0, 4.0]), 5, 1),
        ([good], ids, 0, 1),
        ([good], ids, 5, -1),
        ([good, point_mixture(0.0, n=4, k=2, kind="scales")], ids, 5, 1),
        ([zero], ids, 5, 1),
        ([negative], ids, 5, 1),
        ([nan_means], ids, 5, 1),
        ([both], ids, 5, 1),
    ):
        with pytest.raises(ValueError):
            npost.sample_ensemble_latent(members, case_ids, n_draws, seed)


def test_ensemble_directory_loading_refuses_unsafe_records(tmp_path):
    for i, record in enumerate(
        (
            {"format": "other/1", "members": ["member_0.npz"], "weights": "equal"},
            {"format": npost.ENSEMBLE_FORMAT, "members": ["member_0.npz"], "weights": [0.5]},
            {"format": npost.ENSEMBLE_FORMAT, "members": ["../member_0.npz"], "weights": "equal"},
            {"format": npost.ENSEMBLE_FORMAT, "members": ["/abs/member.npz"], "weights": "equal"},
            {"format": npost.ENSEMBLE_FORMAT, "members": ["member_0.pkl"], "weights": "equal"},
            {"format": npost.ENSEMBLE_FORMAT, "members": "member_0.npz", "weights": "equal"},
        )
    ):
        d = tmp_path / f"e{i}"
        d.mkdir()
        (d / "ensemble.json").write_text(json.dumps(record))
        with pytest.raises(ValueError):
            npost.EqualWeightEnsemble.load(d)


def test_ensemble_load_scalar_file_is_one_member_and_missing_file_is_truthful(tmp_path):
    torch = _torch_or_skip()
    x, z = toy_rows()
    est = npost.NeuralPosterior(1, random_state=2, **TINY).fit(x, z)
    names = est.save(tmp_path / "ens")
    member = tmp_path / "ens" / names[0]
    for source in (member, str(member)):
        back = npost.EqualWeightEnsemble.load(source)
        assert len(back.members) == 1
        assert npost.NeuralPosterior.load(source).n_members == 1
    # an ordered one-shot iterable of real saved members loads in order
    est2 = npost.NeuralPosterior(2, random_state=3, **TINY).fit(x, z)
    names2 = est2.save(tmp_path / "ens2")
    ordered = [tmp_path / "ens2" / f for f in names2[::-1]]
    from_iter = npost.EqualWeightEnsemble.load(p for p in ordered)
    from_list = npost.EqualWeightEnsemble.load(ordered)
    assert len(from_iter.members) == 2
    assert np.array_equal(from_iter.members[0].log_prob(x, z), from_list.members[0].log_prob(x, z))
    assert np.array_equal(from_iter.members[0].log_prob(x, z), est2.members_[1].log_prob(x, z))
    for missing in (tmp_path / "no_such_member.npz", str(tmp_path / "no_such_member.npz")):
        with pytest.raises(FileNotFoundError, match="no_such_member.npz"):
            npost.EqualWeightEnsemble.load(missing)
        with pytest.raises(FileNotFoundError, match="no_such_member.npz"):
            npost.NeuralPosterior.load(missing)
    assert torch is not None


def test_numpy_draws_feed_evaluation_metrics_and_decisions():
    rng = np.random.default_rng(9)
    n = 160
    members = [
        {
            "weights": rng.dirichlet(np.ones(2), size=n),
            "means": rng.standard_normal((n, 2, 2)),
            "scales": np.exp(rng.uniform(-1, 0, (n, 2, 2))),
        }
        for _ in range(3)
    ]
    draws = npost.sample_ensemble_latent(members, np.arange(n), 41, seed=4)["latent"]
    truth = rng.standard_normal((n, 2))
    groups = np.zeros((n, 2), dtype=int)
    recal = npost.QuantileRecalibrator().fit(draws[:120], truth[:120], groups[:120], seed=2)
    adjusted = recal.transform(draws[120:], groups[120:])
    scores = score_samples(adjusted, truth[120:], levels=[0.5, 0.9], joint=True)
    assert scores.crps.shape == (40, 2) and scores.energy.shape == (40,)
    p = event_probabilities(adjusted[:, :, :1] > 0.0)
    assert p.shape == (40, 1) and np.all((p >= 0) & (p <= 1))
    losses = np.stack([np.ones((40, 41)), 3.0 * (adjusted[:, :, 0] > 0.0)], axis=2)
    risks = decision_risks(losses)
    assert risks.expected_losses.shape == (40, 2) and len(risks.bayes_actions) == 40


# ---------------------------------------------------------------------------
# Training (tiny networks; PyTorch only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("covariance", ["diagonal", "full"])
def test_estimator_fit_and_sample_match_its_members(covariance):
    torch = _torch_or_skip()
    x, z = toy_rows()
    np_state, t_state = npost._global_rng_state(), torch.random.get_rng_state()
    est = npost.NeuralPosterior(2, covariance=covariance, random_state=4, **TINY).fit(x, z)
    assert npost._global_rng_state() == np_state
    assert torch.equal(torch.random.get_rng_state(), t_state)
    assert est.n_features_in_ == 10 and est.n_targets_ == 3
    assert est.member_seeds_ == npost.member_seeds(4, 2)
    ids = np.arange(x.shape[0])
    for member, seeds in zip(est.members_, est.member_seeds_, strict=True):
        alone = npost.MixturePosterior(tiny(covariance)).fit(x, z, ids, seeds)
        a, b = alone.net.state_dict(), member.net.state_dict()
        assert all(torch.equal(a[k], b[k]) for k in a)
        assert member.history["seeds"] == seeds
        assert member.history["chosen_epochs"] >= 1
    w0, w1 = (m.net.state_dict()["0.weight"] for m in est.members_)
    assert not torch.equal(w0, w1)

    draws = est.sample(x[:9], 31, case_ids=np.arange(9) + 40, seed=3)
    assert draws.shape == (9, 31, 3) and np.all(np.isfinite(draws))
    labelled = est.sample(x[:9], 31, case_ids=np.arange(9) + 40, seed=3, return_labels=True)
    mixtures = est.predict_mixtures(x[:9])
    direct = npost.sample_ensemble_latent(mixtures, np.arange(9) + 40, 31, seed=3)
    for key in ("latent", "member", "component"):
        assert np.array_equal(labelled[key], direct[key])
    assert np.array_equal(labelled["latent"], draws)
    key = "scales" if covariance == "diagonal" else "chol"
    assert set(mixtures[0]) == {"weights", "means", key}
    assert npost._global_rng_state() == np_state


def test_sampling_is_invariant_to_case_order_and_chunking():
    _torch_or_skip()
    x, z = toy_rows()
    est = npost.NeuralPosterior(2, random_state=1, **TINY).fit(x, z)
    ids = np.array([40, 7, 13, 2, 99, 5, 31, 8, 60])
    one = est.sample(x[:9], 25, case_ids=ids, seed=6, chunk=9, return_labels=True)
    chunked = est.sample(x[:9], 25, case_ids=ids, seed=6, chunk=2, return_labels=True)
    perm = np.array([3, 8, 0, 5, 1, 7, 2, 6, 4])
    permuted = est.sample(x[:9][perm], 25, case_ids=ids[perm], seed=6, return_labels=True)
    subset = est.sample(x[2:4], 25, case_ids=ids[2:4], seed=6, return_labels=True)
    for key in ("member", "component"):
        assert np.array_equal(chunked[key], one[key])
        assert np.array_equal(permuted[key], one[key][perm])
        assert np.array_equal(subset[key], one[key][2:4])
    # network batching changes only the rounding of the mixture parameters
    assert np.allclose(chunked["latent"], one["latent"], rtol=1e-12, atol=1e-14)
    assert np.allclose(permuted["latent"], one["latent"][perm], rtol=1e-12, atol=1e-14)
    other = est.sample(x[:9], 25, case_ids=ids, seed=7, return_labels=True)
    assert not np.array_equal(other["member"], one["member"])
    for bad in (
        {"case_ids": ids[:8]},
        {"case_ids": np.zeros(9, dtype=int)},
        {"case_ids": ids, "seed": -1},
    ):
        kwargs = {"case_ids": ids, "seed": 6, **bad}
        with pytest.raises(ValueError):
            est.sample(x[:9], 25, **kwargs)
    with pytest.raises(ValueError):
        est.sample(x[:9], 0, case_ids=ids, seed=6)
    with pytest.raises(ValueError):
        est.sample(x[:9, :4], 5, case_ids=ids, seed=6)
    with pytest.raises(ValueError):
        est.sample(np.full((9, 10), np.nan), 5, case_ids=ids, seed=6)


def test_full_log_density_matches_scipy_multivariate_normal():
    torch = _torch_or_skip()
    from scipy import stats

    rng = np.random.default_rng(6)
    k, d = 3, 4
    logits = torch.as_tensor(rng.standard_normal((1, k)))
    means = torch.as_tensor(rng.standard_normal((1, k, d)))
    a = rng.standard_normal((1, k, d, d))
    chol = torch.as_tensor(
        np.tril(a, -1) + np.eye(d) * np.exp(rng.uniform(-1, 1, (1, k, d)))[..., None]
    )
    z = rng.standard_normal((1, d))
    lp = npost.mixture_log_prob(
        torch.log_softmax(logits, 1), means, chol, torch.as_tensor(z), "full"
    )
    w = np.exp(logits.numpy()[0]) / np.exp(logits.numpy()[0]).sum()
    ref = np.log(
        sum(
            w[j]
            * stats.multivariate_normal(
                means.numpy()[0, j], chol.numpy()[0, j] @ chol.numpy()[0, j].T
            ).pdf(z[0])
            for j in range(k)
        )
    )
    assert float(lp[0]) == pytest.approx(ref, rel=1e-10)


def test_diagonal_log_density_matches_scipy_normal():
    torch = _torch_or_skip()
    from scipy import stats

    rng = np.random.default_rng(10)
    k, d = 3, 2
    logits = rng.standard_normal((1, k))
    means = rng.standard_normal((1, k, d))
    log_scales = rng.uniform(-1, 1, (1, k, d))
    z = rng.standard_normal((1, d))
    lp = npost.mixture_log_prob(
        torch.log_softmax(torch.as_tensor(logits), 1),
        torch.as_tensor(means),
        torch.as_tensor(log_scales),
        torch.as_tensor(z),
        "diagonal",
    )
    w = np.exp(logits[0]) / np.exp(logits[0]).sum()
    dens = [np.prod(stats.norm(means[0, j], np.exp(log_scales[0, j])).pdf(z[0])) for j in range(k)]
    assert float(lp[0]) == pytest.approx(np.log(np.dot(w, dens)), rel=1e-10)


def test_save_load_parity_and_member_file_interoperability(tmp_path):
    torch = _torch_or_skip()
    x, z = toy_rows()
    est = npost.NeuralPosterior(2, random_state=2, **TINY).fit(x, z)
    ids = np.arange(9) + 40
    out = est.sample(x[:9], 31, case_ids=ids, seed=3, return_labels=True)
    names = est.save(tmp_path / "ens")
    assert names == ["member_0.npz", "member_1.npz"]
    with pytest.raises(FileExistsError):
        est.save(tmp_path / "ens")
    meta = json.loads((tmp_path / "ens" / "ensemble.json").read_text())
    assert meta == {"format": npost.ENSEMBLE_FORMAT, "members": names, "weights": "equal"}
    for f in names:
        with np.load(tmp_path / "ens" / f, allow_pickle=False) as data:
            assert str(data["format"]) == npost.MODEL_FORMAT
    for source in (tmp_path / "ens", [tmp_path / "ens" / f for f in names]):
        back = npost.NeuralPosterior.load(source)
        again = back.sample(x[:9], 31, case_ids=ids, seed=3, return_labels=True)
        for key in ("latent", "member", "component"):
            assert np.array_equal(again[key], out[key])
        assert back.member_seeds == est.member_seeds_ and back.random_state is None
        assert back.settings() == est.settings()
    assert np.array_equal(back.members_[0].log_prob(x, z), est.members_[0].log_prob(x, z))
    reversed_ = npost.NeuralPosterior.load([tmp_path / "ens" / f for f in names[::-1]])
    assert reversed_.member_seeds == est.member_seeds_[::-1]
    # a clone of the loaded estimator refits the same members from the recorded seeds
    refit = clone(back).fit(x, z)
    for a, b in zip(refit.members_, est.members_, strict=True):
        sa, sb = a.net.state_dict(), b.net.state_dict()
        assert all(torch.equal(sa[k], sb[k]) for k in sa)


def test_clone_refit_isolation():
    torch = _torch_or_skip()
    x, z = toy_rows()
    template = npost.NeuralPosterior(2, random_state=3, **TINY)
    params = template.get_params()
    a = clone(template).fit(x, z)
    b = clone(template).fit(x, z)
    assert template.get_params() == params and not hasattr(template, "ensemble_")
    assert a.ensemble_ is not b.ensemble_
    for ma, mb in zip(a.members_, b.members_, strict=True):
        assert ma.net is not mb.net
        sa, sb = ma.net.state_dict(), mb.net.state_dict()
        assert all(torch.equal(sa[k], sb[k]) for k in sa)
    fresh = clone(a)
    assert not hasattr(fresh, "ensemble_") and fresh.get_params() == a.get_params()
    x2, z2 = toy_rows(seed=1)
    fresh.fit(x2, z2)
    w_a = a.members_[0].net.state_dict()["0.weight"]
    assert not torch.equal(fresh.members_[0].net.state_dict()["0.weight"], w_a)
    assert torch.equal(a.members_[0].net.state_dict()["0.weight"], w_a)


def test_fit_refuses_bad_inputs():
    _torch_or_skip()
    x, z = toy_rows()
    ids = np.arange(x.shape[0])
    post = npost.MixturePosterior(tiny("diagonal"))
    with pytest.raises(ValueError):
        post.fit(x, z, ids[::-1], SEEDS)
    bad = z.copy()
    bad[0, 0] = math.inf
    with pytest.raises(ValueError):
        post.fit(x, bad, ids, SEEDS)
    with pytest.raises(ValueError):
        post.fit(x, z, ids, {"pca": 1, "init": 2})
    with pytest.raises(ValueError):
        npost.MixturePosterior(npost.MixtureSettings(**{**TINY, "n_validation": 96})).fit(
            x, z, ids, SEEDS
        )
    est = npost.NeuralPosterior(1, **TINY)
    for args in ((x, z[:, 0]), (x, z[:50]), (x[:, 0], z), (x, bad)):
        with pytest.raises(ValueError):
            est.fit(*args)
    with pytest.raises(ValueError):
        est.fit(x, z, row_ids=ids[::-1])
    with pytest.raises(RuntimeError):
        npost.EqualWeightEnsemble([npost.MixturePosterior()])


def _design_bank(seed=0, rows=150, n_time=12, n_sensors=3):
    rng = np.random.default_rng(seed)
    targets = rng.normal(size=(rows, 2))
    t = np.arange(n_time)
    depth = np.arange(n_sensors)
    wave = np.sin(2 * np.pi * t / 6)[None, :, None] * np.exp(-depth / 2)[None, None, :]
    obs = (
        targets[:, 0, None, None] * wave
        + targets[:, 1, None, None] * (depth / 2)[None, None, :]
        + 0.1 * rng.normal(size=(rows, n_time, n_sensors))
    )
    return SimulationBank(obs, targets)


def test_design_refits_clone_the_template_and_feed_evaluation():
    _torch_or_skip()
    bank = _design_bank()
    designs = [
        bank.select([0, 2], start=0, stop=12, step=2),
        bank.select([1], start=0, stop=12, step=1),
    ]
    template = npost.NeuralPosterior(2, random_state=5, **TINY)
    params = template.get_params()
    train, held = list(range(120)), np.arange(120, 150)
    refits = fit_designs(template, bank, designs, train_rows=train)
    assert template.get_params() == params and not hasattr(template, "ensemble_")
    assert [r.model.n_features_in_ for r in refits] == [12, 12]
    assert refits[0].model is not refits[1].model
    # each design's fit equals a direct fit on that design's features
    x0, y0 = bank.features(designs[0], train)
    direct = clone(template).fit(x0, y0)
    feats = refits[0].features(bank.observations[held])
    a = refits[0].model.sample(feats, 21, case_ids=held, seed=1)
    assert np.array_equal(a, direct.sample(feats, 21, case_ids=held, seed=1))
    for refit in refits:
        draws = refit.model.sample(
            refit.features(bank.observations[held]), 21, case_ids=held, seed=1
        )
        scores = score_samples(draws, bank.targets[held], levels=[0.5, 0.9])
        assert scores.crps.shape == (30, 2) and np.all(np.isfinite(scores.crps))
        assert event_probabilities(draws[:, :, :1] > 0).shape == (30, 1)


def test_example_runs_end_to_end():
    _torch_or_skip()
    results = runpy.run_path(str(EXAMPLE))["main"](verbose=False)
    assert set(results) == {"dense", "sparse"}
    for r in results.values():
        assert r["crps"].shape == (2,) and np.all(np.isfinite(r["crps"]))
        assert r["coverage"].shape == (2, 2)
        assert 0.0 <= r["mean_event_probability"] <= 1.0
