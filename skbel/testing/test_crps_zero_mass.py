"""Finite regression tests: zero-mass support must not change empirical CRPS."""

import numpy as np
import pytest

from skbel.metrics.posterior import marginal_crps


def _oracle(x, w, y):
    w = np.asarray(w, float) / np.sum(w)
    term1 = np.sum(w * np.abs(x - y))
    term2 = 0.5 * np.sum(w[:, None] * w[None, :] * np.abs(x[:, None] - x[None, :]))
    return term1 - term2


def _score(x, y, w):
    return marginal_crps(
        np.asarray(x, float)[None, :, None], np.array([[y]], float), np.asarray(w, float)[None, :]
    )[0, 0]


@pytest.mark.parametrize("huge", [1e100, -1e100, 1e150])
def test_huge_zero_mass_support_invariant(huge):
    assert _score([0.0, 1.0], 0.0, [0.5, 0.5]) == pytest.approx(0.25, abs=1e-12)
    assert _score([0.0, 1.0, huge], 0.0, [0.5, 0.5, 0.0]) == pytest.approx(0.25, abs=1e-12)


def test_ordering_multiple_zeros_and_unnormalized_weights():
    x = [1e100, 0.0, -1e100, 1.0, 1e50]
    w = [0.0, 3.0, 0.0, 3.0, 0.0]
    assert _score(x, 0.0, w) == pytest.approx(0.25, abs=1e-12)
    assert _score(x[::-1], 0.0, w[::-1]) == pytest.approx(0.25, abs=1e-12)


def test_multi_case_multi_target_and_inputs_unchanged():
    base = np.array([[0.0, 1.0], [2.0, 5.0]])
    samples = np.zeros((2, 4, 2))
    samples[:, :2, 0] = base
    samples[:, :2, 1] = base + 1
    samples[0, 2:, :] = 1e100
    samples[1, 2:, :] = -1e100
    truth = np.array([[0.0, 1.0], [2.0, 3.0]])
    weights = np.array([[1.0, 1.0, 0.0, 0.0], [2.0, 6.0, 0.0, 0.0]])
    s0, w0, t0 = samples.copy(), weights.copy(), truth.copy()
    res = marginal_crps(samples, truth, weights)
    ref = marginal_crps(samples[:, :2], truth, weights[:, :2])
    np.testing.assert_allclose(res, ref, atol=1e-12)
    np.testing.assert_array_equal(samples, s0)
    np.testing.assert_array_equal(weights, w0)
    np.testing.assert_array_equal(truth, t0)


def test_moderate_weighted_matches_oracle():
    rng = np.random.default_rng(0)
    for _ in range(10):
        x = rng.normal(size=12)
        w = rng.random(12)
        w[::4] = 0.0
        y = rng.normal()
        assert _score(x, y, w) == pytest.approx(_oracle(x, w, y), abs=1e-12)


def test_singleton_mass_is_absolute_error():
    assert _score([3.0, 1e100], 5.0, [2.0, 0.0]) == pytest.approx(2.0, abs=1e-12)


@pytest.mark.parametrize(
    "w", [[0.0, 0.0], [-1.0, 1.0], [np.nan, 1.0], [np.inf, 1.0], [1.0, 1.0, 1.0]]
)
def test_malformed_weights_still_fail(w):
    with pytest.raises(ValueError):
        _score([0.0, 1.0], 0.0, w)
