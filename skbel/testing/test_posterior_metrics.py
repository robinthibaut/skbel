"""Unit tests for skbel.metrics.posterior.

Import strategy: normal tests require the complete package import. An import
failure intentionally fails test collection rather than silently bypassing
``skbel.__init__`` with a direct file import.
"""

from __future__ import annotations

import unittest

import numpy as np

from skbel.metrics import bayes_action_set, brier_score, expected_action_losses, marginal_crps

# --------------------------------------------------------------------------
# Independent oracle implementations (naive O(M^2)/O(M) double loops, NOT the
# production sorted formula) used to cross-check the module under arbitrary
# random inputs, not just the exact hand-worked tiny cases below.
# --------------------------------------------------------------------------


def _oracle_empirical_crps_1d(x: np.ndarray, w: np.ndarray, y: float) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    m = len(x)
    term1 = sum(w[i] * abs(x[i] - y) for i in range(m))
    term2 = sum(w[i] * w[j] * abs(x[i] - x[j]) for i in range(m) for j in range(m))
    return term1 - 0.5 * term2


def _oracle_unbiased_crps_1d(x: np.ndarray, y: float) -> float:
    x = np.asarray(x, dtype=float)
    m = len(x)
    term1 = sum(abs(xi - y) for xi in x) / m
    term2 = sum(abs(x[i] - x[j]) for i in range(m) for j in range(m) if i != j)
    return term1 - term2 / (2.0 * m * (m - 1))


def _oracle_brier(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    cases, classes = probs.shape
    out = np.zeros(cases)
    for c in range(cases):
        s = 0.0
        for k in range(classes):
            target = 1.0 if k == labels[c] else 0.0
            s += (probs[c, k] - target) ** 2
        out[c] = s
    return out


def _oracle_expected_action_losses(losses: np.ndarray, w: np.ndarray) -> np.ndarray:
    draws, actions = losses.shape
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    out = np.zeros(actions)
    for a in range(actions):
        out[a] = sum(w[i] * losses[i, a] for i in range(draws))
    return out


# --------------------------------------------------------------------------
# marginal_crps
# --------------------------------------------------------------------------


class TestMarginalCrpsExactTinyCases(unittest.TestCase):
    """Hand-worked exact cases with analytically determined values."""

    def test_pointmass_pair_empirical(self):
        # samples = [0, 2], truth = 1 -> empirical CRPS = 0.5
        samples = np.array([[[0.0], [2.0]]])  # (cases=1, draws=2, targets=1)
        truth = np.array([[1.0]])
        result = marginal_crps(samples, truth, estimator="empirical")
        self.assertEqual(result.shape, (1, 1))
        self.assertAlmostEqual(result[0, 0], 0.5, places=12)

    def test_pointmass_pair_unbiased_is_zero(self):
        samples = np.array([[[0.0], [2.0]]])
        truth = np.array([[1.0]])
        result = marginal_crps(samples, truth, estimator="unbiased")
        self.assertAlmostEqual(result[0, 0], 0.0, places=12)

    def test_weighted_pair_masses_quarter_three_quarter(self):
        # samples = [0, 1], weights = [.25, .75], truth = 0 -> 0.5625
        samples = np.array([[[0.0], [1.0]]])
        truth = np.array([[0.0]])
        weights = np.array([[0.25, 0.75]])
        result = marginal_crps(samples, truth, weights=weights, estimator="empirical")
        self.assertAlmostEqual(result[0, 0], 0.5625, places=12)


class TestMarginalCrpsOracleCrossCheck(unittest.TestCase):
    """Cross-check against the independent naive oracle on random data."""

    def test_random_cases_targets_against_oracle_empirical_unweighted(self):
        rng = np.random.default_rng(20260910)
        cases, draws, targets = 4, 7, 3
        samples = rng.normal(size=(cases, draws, targets))
        truth = rng.normal(size=(cases, targets))
        result = marginal_crps(samples, truth, estimator="empirical")
        for c in range(cases):
            for t in range(targets):
                expected = _oracle_empirical_crps_1d(
                    samples[c, :, t], np.full(draws, 1.0 / draws), truth[c, t]
                )
                self.assertAlmostEqual(result[c, t], expected, places=9)

    def test_random_cases_targets_against_oracle_empirical_weighted(self):
        rng = np.random.default_rng(7)
        cases, draws, targets = 3, 9, 2
        samples = rng.normal(size=(cases, draws, targets))
        truth = rng.normal(size=(cases, targets))
        raw_weights = rng.uniform(0.01, 1.0, size=(cases, draws))
        result = marginal_crps(samples, truth, weights=raw_weights, estimator="empirical")
        for c in range(cases):
            for t in range(targets):
                expected = _oracle_empirical_crps_1d(samples[c, :, t], raw_weights[c], truth[c, t])
                self.assertAlmostEqual(result[c, t], expected, places=9)

    def test_random_against_oracle_unbiased(self):
        rng = np.random.default_rng(99)
        cases, draws, targets = 3, 12, 2
        samples = rng.normal(size=(cases, draws, targets))
        truth = rng.normal(size=(cases, targets))
        result = marginal_crps(samples, truth, estimator="unbiased")
        for c in range(cases):
            for t in range(targets):
                expected = _oracle_unbiased_crps_1d(samples[c, :, t], truth[c, t])
                self.assertAlmostEqual(result[c, t], expected, places=9)

    def test_weights_need_not_be_prenormalized(self):
        # Same relative masses (1:3) scaled up -> identical result to .25/.75
        samples = np.array([[[0.0], [1.0]]])
        truth = np.array([[0.0]])
        weights = np.array([[10.0, 30.0]])  # sums to 40, ratio 1:3
        result = marginal_crps(samples, truth, weights=weights, estimator="empirical")
        self.assertAlmostEqual(result[0, 0], 0.5625, places=12)

    def test_huge_weights_are_rescaled_before_sum(self):
        # Direct summation of these finite weights overflows to inf and used
        # to produce an all-zero normalized vector and an incorrect CRPS.
        samples = np.array([[[0.0], [2.0]]])
        truth = np.array([[1.0]])
        result = marginal_crps(samples, truth, weights=np.array([[1e308, 1e308]]))
        self.assertAlmostEqual(result[0, 0], 0.5, places=12)

    def test_huge_and_tiny_weight_rescalings_preserve_relative_mass(self):
        samples = np.array([[[0.0], [1.0]]])
        truth = np.array([[0.0]])
        expected = marginal_crps(samples, truth, weights=np.array([[1.0, 3.0]]))
        for weights in (np.array([[1e307, 3e307]]), np.array([[1e-320, 3e-320]])):
            np.testing.assert_allclose(
                marginal_crps(samples, truth, weights=weights), expected, atol=1e-12
            )


class TestMarginalCrpsShapesAndPermutations(unittest.TestCase):
    def test_no_silent_broadcasting_truth_shape_mismatch(self):
        samples = np.zeros((2, 5, 3))
        bad_truth = np.zeros((2, 4))  # wrong targets count
        with self.assertRaises(ValueError):
            marginal_crps(samples, bad_truth)

    def test_samples_must_be_3d(self):
        with self.assertRaises(ValueError):
            marginal_crps(np.zeros((2, 5)), np.zeros((2,)))

    def test_multi_case_multi_target_matches_per_case_per_target_loop(self):
        rng = np.random.default_rng(11)
        cases, draws, targets = 3, 6, 4
        samples = rng.normal(size=(cases, draws, targets))
        truth = rng.normal(size=(cases, targets))
        batched = marginal_crps(samples, truth, estimator="empirical")
        for c in range(cases):
            for t in range(targets):
                single = marginal_crps(
                    samples[c : c + 1, :, t : t + 1],
                    truth[c : c + 1, t : t + 1],
                    estimator="empirical",
                )
                self.assertAlmostEqual(batched[c, t], single[0, 0], places=9)

    def test_permutation_invariance_unweighted(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=8)
        perm = rng.permutation(8)
        samples_a = x.reshape(1, 8, 1)
        samples_b = x[perm].reshape(1, 8, 1)
        truth = np.array([[0.3]])
        a = marginal_crps(samples_a, truth, estimator="empirical")
        b = marginal_crps(samples_b, truth, estimator="empirical")
        self.assertAlmostEqual(a[0, 0], b[0, 0], places=12)

    def test_permutation_invariance_weighted_moves_with_draws(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=6)
        w = rng.uniform(0.1, 1.0, size=6)
        perm = rng.permutation(6)
        samples_a = x.reshape(1, 6, 1)
        samples_b = x[perm].reshape(1, 6, 1)
        w_a = w.reshape(1, 6)
        w_b = w[perm].reshape(1, 6)
        truth = np.array([[0.0]])
        a = marginal_crps(samples_a, truth, weights=w_a, estimator="empirical")
        b = marginal_crps(samples_b, truth, weights=w_b, estimator="empirical")
        self.assertAlmostEqual(a[0, 0], b[0, 0], places=9)


class TestMarginalCrpsEquivariance(unittest.TestCase):
    def test_shift_equivariance(self):
        rng = np.random.default_rng(5)
        samples = rng.normal(size=(2, 5, 2))
        truth = rng.normal(size=(2, 2))
        shift = 137.0
        base = marginal_crps(samples, truth, estimator="empirical")
        shifted = marginal_crps(samples + shift, truth + shift, estimator="empirical")
        np.testing.assert_allclose(base, shifted, atol=1e-9)

    def test_positive_scale_equivariance(self):
        rng = np.random.default_rng(6)
        samples = rng.normal(size=(2, 5, 2))
        truth = rng.normal(size=(2, 2))
        a = 3.5
        base = marginal_crps(samples, truth, estimator="empirical")
        scaled = marginal_crps(samples * a, truth * a, estimator="empirical")
        np.testing.assert_allclose(scaled, base * a, atol=1e-9)


class TestMarginalCrpsEdgeCasesAndValidation(unittest.TestCase):
    def test_duplicate_draws_no_crash_and_correct(self):
        samples = np.array([[[1.0], [1.0], [1.0], [1.0]]])  # all identical draws
        truth = np.array([[2.0]])
        result = marginal_crps(samples, truth, estimator="empirical")
        # a point mass at 1 scored against truth 2: CRPS = |1-2| = 1
        self.assertAlmostEqual(result[0, 0], 1.0, places=12)

    def test_one_draw_valid_for_empirical(self):
        samples = np.array([[[5.0]]])
        truth = np.array([[2.0]])
        result = marginal_crps(samples, truth, estimator="empirical")
        self.assertAlmostEqual(result[0, 0], 3.0, places=12)

    def test_one_draw_invalid_for_unbiased(self):
        samples = np.array([[[5.0]]])
        truth = np.array([[2.0]])
        with self.assertRaises(ValueError):
            marginal_crps(samples, truth, estimator="unbiased")

    def test_unbiased_is_provably_nonnegative_not_possibly_negative(self):
        # For a fixed truth value y, the triangle inequality ensures that the
        # finite-sample iid U-estimator is non-negative in exact arithmetic.
        # Include a tight cluster plus one distant point as a stress case.
        samples = np.array([[[0.0], [0.0], [100.0]]])
        truth = np.array([[0.0]])
        result = marginal_crps(samples, truth, estimator="unbiased")
        self.assertGreaterEqual(result[0, 0], 0.0)

        rng = np.random.default_rng(2026)
        for _ in range(200):
            m = int(rng.integers(2, 9))
            x = rng.normal(scale=rng.uniform(0.1, 5.0), size=m)
            y = rng.normal(scale=rng.uniform(0.1, 10.0))
            value = marginal_crps(x.reshape(1, m, 1), np.array([[y]]), estimator="unbiased")[0, 0]
            self.assertGreaterEqual(
                value, -1e-9, msg=f"unbiased CRPS went negative for x={x!r}, y={y!r}"
            )

    def test_unbiased_rejects_weights(self):
        samples = np.zeros((1, 3, 1))
        truth = np.zeros((1, 1))
        weights = np.full((1, 3), 1.0 / 3.0)
        with self.assertRaises(ValueError):
            marginal_crps(samples, truth, weights=weights, estimator="unbiased")

    def test_all_zero_weights_row_rejected(self):
        samples = np.zeros((1, 3, 1))
        truth = np.zeros((1, 1))
        weights = np.zeros((1, 3))
        with self.assertRaises(ValueError):
            marginal_crps(samples, truth, weights=weights)

    def test_negative_weights_rejected(self):
        samples = np.zeros((1, 3, 1))
        truth = np.zeros((1, 1))
        weights = np.array([[0.5, -0.1, 0.6]])
        with self.assertRaises(ValueError):
            marginal_crps(samples, truth, weights=weights)

    def test_nonfinite_weights_rejected(self):
        samples = np.zeros((1, 3, 1))
        truth = np.zeros((1, 1))
        weights = np.array([[0.5, np.nan, 0.6]])
        with self.assertRaises(ValueError):
            marginal_crps(samples, truth, weights=weights)

    def test_nonfinite_samples_rejected(self):
        samples = np.array([[[0.0], [np.inf], [1.0]]])
        truth = np.zeros((1, 1))
        with self.assertRaises(ValueError):
            marginal_crps(samples, truth)

    def test_empty_axes_rejected(self):
        with self.assertRaises(ValueError):
            marginal_crps(np.zeros((0, 3, 1)), np.zeros((0, 1)))
        with self.assertRaises(ValueError):
            marginal_crps(np.zeros((2, 0, 1)), np.zeros((2, 1)))
        with self.assertRaises(ValueError):
            marginal_crps(np.zeros((2, 3, 0)), np.zeros((2, 0)))

    def test_invalid_estimator_name_rejected(self):
        samples = np.zeros((1, 3, 1))
        truth = np.zeros((1, 1))
        with self.assertRaises(ValueError):
            marginal_crps(samples, truth, estimator="not-a-real-estimator")


# --------------------------------------------------------------------------
# brier_score
# --------------------------------------------------------------------------


class TestBrierScore(unittest.TestCase):
    def test_exact_binary_case(self):
        probs = np.array([[0.7, 0.3], [0.2, 0.8]])
        labels = np.array([0, 1])
        result = brier_score(probs, labels)
        # case 0: (0.7-1)^2 + (0.3-0)^2 = 0.09+0.09=0.18
        # case 1: (0.2-0)^2 + (0.8-1)^2 = 0.04+0.04=0.08
        np.testing.assert_allclose(result, [0.18, 0.08], atol=1e-12)

    def test_binary_convention_is_twice_single_probability_convention(self):
        p_positive = 0.63
        probs = np.array([[1 - p_positive, p_positive]])
        labels = np.array([1])
        result = brier_score(probs, labels)
        single_convention = (p_positive - 1.0) ** 2
        self.assertAlmostEqual(result[0], 2.0 * single_convention, places=12)

    def test_random_against_oracle(self):
        rng = np.random.default_rng(42)
        cases, classes = 6, 4
        raw = rng.uniform(0.01, 1.0, size=(cases, classes))
        probs = raw / raw.sum(axis=1, keepdims=True)
        labels = rng.integers(0, classes, size=cases)
        result = brier_score(probs, labels)
        expected = _oracle_brier(probs, labels)
        np.testing.assert_allclose(result, expected, atol=1e-9)

    def test_rows_not_summing_to_one_rejected(self):
        probs = np.array([[0.5, 0.4]])  # sums to 0.9
        labels = np.array([0])
        with self.assertRaises(ValueError):
            brier_score(probs, labels)

    def test_negative_probability_rejected(self):
        probs = np.array([[1.2, -0.2]])
        labels = np.array([0])
        with self.assertRaises(ValueError):
            brier_score(probs, labels)

    def test_non_integer_labels_rejected(self):
        probs = np.array([[0.5, 0.5]])
        labels = np.array([0.0])  # float, not int
        with self.assertRaises(TypeError):
            brier_score(probs, labels)

    def test_out_of_range_label_rejected(self):
        probs = np.array([[0.5, 0.5]])
        labels = np.array([2])  # only classes 0,1 exist
        with self.assertRaises(ValueError):
            brier_score(probs, labels)

    def test_shape_mismatch_rejected(self):
        probs = np.array([[0.5, 0.5], [0.3, 0.7]])
        labels = np.array([0])  # only one label for two cases
        with self.assertRaises(ValueError):
            brier_score(probs, labels)

    def test_empty_axes_rejected(self):
        with self.assertRaises(ValueError):
            brier_score(np.zeros((0, 2)), np.zeros((0,), dtype=int))


# --------------------------------------------------------------------------
# expected_action_losses / bayes_action_set
# --------------------------------------------------------------------------


class TestExpectedActionLossesAndBayesActionSet(unittest.TestCase):
    def test_uniform_weights_matches_simple_mean(self):
        losses = np.array([[1.0, 4.0], [3.0, 2.0], [5.0, 0.0]])
        result = expected_action_losses(losses)
        expected = losses.mean(axis=0)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_nonuniform_weights_against_oracle(self):
        rng = np.random.default_rng(123)
        draws, actions = 7, 3
        losses = rng.normal(size=(draws, actions))
        weights = rng.uniform(0.1, 2.0, size=draws)
        result = expected_action_losses(losses, weights=weights)
        expected = _oracle_expected_action_losses(losses, weights)
        np.testing.assert_allclose(result, expected, atol=1e-9)

    def test_single_best_action(self):
        losses = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        min_risk, minimizers = bayes_action_set(losses)
        self.assertAlmostEqual(min_risk, 1.0, places=12)
        np.testing.assert_array_equal(minimizers, np.array([0]))

    def test_exact_tie_returns_all_minimizers(self):
        # actions 0 and 2 tie exactly at expected loss 2.0
        losses = np.array([[2.0, 5.0, 2.0], [2.0, 5.0, 2.0], [2.0, 5.0, 2.0]])
        min_risk, minimizers = bayes_action_set(losses)
        self.assertAlmostEqual(min_risk, 2.0, places=12)
        np.testing.assert_array_equal(np.sort(minimizers), np.array([0, 2]))

    def test_exact_tie_with_nonuniform_weights(self):
        # Constructed so actions 0 and 1 have identical expected loss under
        # nonuniform weights, action 2 is strictly worse.
        weights = np.array([0.2, 0.8])
        losses = np.array(
            [
                [10.0, 0.0, 20.0],
                [0.0, 2.5, 20.0],
            ]
        )
        # action0: 0.2*10+0.8*0=2.0 ; action1: 0.2*0+0.8*2.5=2.0 ; action2: 20
        min_risk, minimizers = bayes_action_set(losses, weights=weights)
        self.assertAlmostEqual(min_risk, 2.0, places=12)
        np.testing.assert_array_equal(np.sort(minimizers), np.array([0, 1]))

    def test_losses_wrong_ndim_rejected(self):
        with self.assertRaises(ValueError):
            expected_action_losses(np.zeros(5))

    def test_empty_axes_rejected(self):
        with self.assertRaises(ValueError):
            expected_action_losses(np.zeros((0, 3)))
        with self.assertRaises(ValueError):
            expected_action_losses(np.zeros((3, 0)))

    def test_all_zero_weights_rejected(self):
        losses = np.zeros((3, 2))
        with self.assertRaises(ValueError):
            expected_action_losses(losses, weights=np.zeros(3))

    def test_huge_weights_are_rescaled_for_action_losses_and_ties(self):
        losses = np.array([[1.0, 3.0], [1.0, 3.0]])
        weights = np.array([1e308, 1e308])
        np.testing.assert_allclose(expected_action_losses(losses, weights), [1.0, 3.0])
        min_risk, minimizers = bayes_action_set(losses, weights)
        self.assertEqual(min_risk, 1.0)
        np.testing.assert_array_equal(minimizers, np.array([0]))

    def test_negative_weights_rejected(self):
        losses = np.zeros((3, 2))
        with self.assertRaises(ValueError):
            expected_action_losses(losses, weights=np.array([0.5, -0.1, 0.6]))

    def test_nonfinite_losses_rejected(self):
        losses = np.array([[1.0, np.nan], [2.0, 3.0]])
        with self.assertRaises(ValueError):
            expected_action_losses(losses)

    def test_weight_shape_mismatch_rejected(self):
        losses = np.zeros((3, 2))
        with self.assertRaises(ValueError):
            expected_action_losses(losses, weights=np.zeros(4))


if __name__ == "__main__":
    unittest.main()
