"""Unit tests for skbel.metrics.joint.energy_score.

All fixtures are deterministic integer-formula constructions (no PRNG), at most
128 draws / 4 targets / 4 cases. The oracle below is an independent explicit
double loop using ``math.fsum``/``math.sqrt``; it shares no helper with the
production module.
"""

from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np

import skbel.metrics.joint as joint_module
from skbel.metrics import energy_score, marginal_crps


def _fixture(cases: int, draws: int, targets: int):
    samples = np.empty((cases, draws, targets))
    truth = np.empty((cases, targets))
    for c in range(cases):
        for t in range(targets):
            truth[c, t] = ((7 * c + 3 * t + 2) % 9 - 4) / 3.0
            for i in range(draws):
                samples[c, i, t] = (
                    ((c + 1) * (i + 2) * (t + 3) + 5 * i + 3 * t + c) % 13 - 6
                ) / 4.0
    return samples, truth


def _weights(cases: int, draws: int):
    return np.array([[((3 * i + c) % 4) / 2.0 for i in range(draws)] for c in range(cases)])


def _dist(a, b) -> float:
    return math.sqrt(math.fsum((float(p) - float(q)) ** 2 for p, q in zip(a, b, strict=True)))


def _oracle_case(x, w, y) -> float:
    total = math.fsum(float(v) for v in w)
    wn = [float(v) / total for v in w]
    n = len(wn)
    term1 = math.fsum(wn[i] * _dist(x[i], y) for i in range(n))
    term2 = math.fsum(wn[i] * wn[j] * _dist(x[i], x[j]) for i in range(n) for j in range(n))
    return term1 - 0.5 * term2


def _oracle(samples, truth, weights=None):
    cases, draws, _ = samples.shape
    out = []
    for c in range(cases):
        w = [1.0] * draws if weights is None else list(weights[c])
        out.append(_oracle_case(samples[c], w, truth[c]))
    return np.array(out)


class TestHandDependenceControl(unittest.TestCase):
    Q_PLUS = np.array([[0.0, 0.0], [1.0, 1.0]])
    Q_MINUS = np.array([[0.0, 1.0], [1.0, 0.0]])

    def test_hand_values_at_both_diagonal_truths(self):
        s2 = math.sqrt(2.0)
        for truth in ([0.0, 0.0], [1.0, 1.0]):
            samples = np.stack([self.Q_PLUS, self.Q_MINUS])
            y = np.array([truth, truth])
            result = energy_score(samples, y)
            self.assertAlmostEqual(result[0], s2 / 4.0, places=14)
            self.assertAlmostEqual(result[1], 1.0 - s2 / 4.0, places=14)
            self.assertAlmostEqual(result[1] - result[0], 1.0 - s2 / 2.0, places=14)

    def test_marginals_identical_and_quarter_but_joint_differs(self):
        samples = np.stack([self.Q_PLUS, self.Q_MINUS])
        truth = np.zeros((2, 2))
        crps = marginal_crps(samples, truth)
        np.testing.assert_allclose(crps, 0.25, atol=1e-14)
        es = energy_score(samples, truth)
        self.assertGreater(abs(es[0] - es[1]), 0.1)

    def test_loop_oracle_agrees_with_hand_control(self):
        self.assertAlmostEqual(
            _oracle_case(self.Q_PLUS, [1, 1], [0, 0]), math.sqrt(2.0) / 4.0, places=14
        )
        self.assertAlmostEqual(
            _oracle_case(self.Q_MINUS, [1, 1], [0, 0]), 1.0 - math.sqrt(2.0) / 4.0, places=14
        )


class TestOracleAndChunking(unittest.TestCase):
    def test_unweighted_matches_loops(self):
        samples, truth = _fixture(4, 12, 4)
        np.testing.assert_allclose(
            energy_score(samples, truth), _oracle(samples, truth), rtol=1e-12, atol=1e-12
        )

    def test_weighted_with_zero_weights_matches_loops(self):
        samples, truth = _fixture(4, 16, 3)
        w = _weights(4, 16)
        self.assertTrue(np.any(w == 0))
        np.testing.assert_allclose(
            energy_score(samples, truth, w), _oracle(samples, truth, w), rtol=1e-12, atol=1e-12
        )

    def test_max_fixture_size_matches_loops_and_chunking_is_invariant(self):
        samples, truth = _fixture(1, 128, 4)
        expected = _oracle(samples, truth)
        np.testing.assert_allclose(energy_score(samples, truth), expected, rtol=1e-11, atol=1e-11)
        with mock.patch.object(joint_module, "_PAIR_CHUNK_ELEMENTS", 1):
            np.testing.assert_allclose(
                energy_score(samples, truth), expected, rtol=1e-11, atol=1e-11
            )

    def test_output_is_float64_per_case_and_integer_input_accepted(self):
        out = energy_score(np.array([[[0, 0], [3, 4]]]), np.array([[0, 0]]))
        self.assertEqual(out.dtype, np.float64)
        self.assertEqual(out.shape, (1,))
        self.assertAlmostEqual(out[0], 0.5 * 5.0 - 0.5 * (0.5 * 5.0), places=12)


class TestInvariances(unittest.TestCase):
    def setUp(self):
        self.samples, self.truth = _fixture(3, 10, 2)
        self.w = _weights(3, 10)
        self.base = energy_score(self.samples, self.truth, self.w)

    def test_joint_atom_permutation(self):
        perm = np.array([3, 7, 0, 9, 1, 5, 2, 8, 4, 6])
        out = energy_score(self.samples[:, perm, :], self.truth, self.w[:, perm])
        np.testing.assert_allclose(out, self.base, atol=1e-12)

    def test_weighted_duplicates_equal_merged_atom(self):
        a, b = [0.0, 0.0], [2.0, 1.0]
        merged = energy_score(np.array([[a, b]]), np.array([[1.0, 1.0]]), np.array([[0.25, 0.75]]))
        split = energy_score(
            np.array([[a, a, b]]),
            np.array([[1.0, 1.0]]),
            np.array([[0.1, 0.15, 0.75]]),
        )
        np.testing.assert_allclose(split, merged, atol=1e-14)

    def test_translation(self):
        shift = np.array([5.0, -3.0])
        out = energy_score(self.samples + shift, self.truth + shift, self.w)
        np.testing.assert_allclose(out, self.base, atol=1e-12)

    def test_orthogonal_rotation_and_coordinate_swap(self):
        rot = np.array([[0.0, -1.0], [1.0, 0.0]])
        out = energy_score(self.samples @ rot.T, self.truth @ rot.T, self.w)
        np.testing.assert_allclose(out, self.base, atol=1e-12)
        out = energy_score(self.samples[:, :, ::-1], self.truth[:, ::-1], self.w)
        np.testing.assert_allclose(out, self.base, atol=1e-12)

    def test_positive_scale(self):
        out = energy_score(self.samples * 4.0, self.truth * 4.0, self.w)
        np.testing.assert_allclose(out, self.base * 4.0, atol=1e-12)

    def test_weights_need_not_be_prenormalized(self):
        out = energy_score(self.samples, self.truth, self.w * 1000.0)
        np.testing.assert_allclose(out, self.base, atol=1e-12)


class TestOneTargetAndPointMass(unittest.TestCase):
    def test_one_target_matches_empirical_crps(self):
        samples, truth = _fixture(4, 9, 1)
        w = _weights(4, 9)
        np.testing.assert_allclose(
            energy_score(samples, truth, w), marginal_crps(samples, truth, w)[:, 0], atol=1e-12
        )

    def test_point_mass_is_distance_to_truth(self):
        samples = np.tile(np.array([3.0, 4.0]), (1, 5, 1))
        out = energy_score(samples, np.array([[0.0, 0.0]]))
        self.assertAlmostEqual(out[0], 5.0, places=12)

    def test_truth_at_point_mass_is_zero(self):
        samples = np.tile(np.array([1.5, -2.0]), (1, 4, 1))
        out = energy_score(samples, np.array([[1.5, -2.0]]))
        self.assertEqual(out[0], 0.0)

    def test_single_draw_is_euclidean_distance(self):
        out = energy_score(np.array([[[1.0, 2.0, 2.0]]]), np.zeros((1, 3)))
        self.assertAlmostEqual(out[0], 3.0, places=14)


class TestRepresentability(unittest.TestCase):
    def test_representable_3_4_5_does_not_square_overflow(self):
        out = energy_score(np.array([[[3e200, 4e200]]]), np.zeros((1, 2)))
        np.testing.assert_allclose(out, [5e200], rtol=1e-14)

    def test_difference_overflow_raises(self):
        samples = np.array([[[1.7e308, 0.0]]])
        with self.assertRaises(ValueError):
            energy_score(samples, np.array([[-1.7e308, 0.0]]))

    def test_norm_overflow_raises(self):
        samples = np.array([[[1.7e308, 1.7e308]]])
        with self.assertRaises(ValueError):
            energy_score(samples, np.zeros((1, 2)))

    def test_pairwise_overflow_raises(self):
        samples = np.array([[[1.7e308, 0.0], [-1.7e308, 0.0]]])
        with self.assertRaises(ValueError):
            energy_score(samples, np.zeros((1, 2)))

    def test_zero_weight_draw_is_excluded_before_distance(self):
        samples = np.array([[[0.0, 0.0], [1.0, 1.0], [1.7e308, 1.7e308]]])
        truth = np.array([[0.0, 0.0]])
        out = energy_score(samples, truth, np.array([[1.0, 1.0, 0.0]]))
        ref = energy_score(samples[:, :2, :], truth)
        np.testing.assert_allclose(out, ref, atol=1e-14)

    def test_lost_positive_mass_raises(self):
        samples = np.zeros((1, 2, 1))
        with self.assertRaises(ValueError):
            energy_score(samples, np.zeros((1, 1)), np.array([[1e308, 1e-320]]))

    def test_huge_weights_are_rescaled(self):
        samples = np.array([[[0.0], [2.0]]])
        out = energy_score(samples, np.array([[1.0]]), np.array([[1e308, 1e308]]))
        self.assertAlmostEqual(out[0], 0.5, places=12)


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.s = np.zeros((2, 3, 2))
        self.t = np.zeros((2, 2))

    def test_malformed_shapes(self):
        with self.assertRaises(ValueError):
            energy_score(np.zeros((3, 2)), np.zeros((3,)))
        with self.assertRaises(ValueError):
            energy_score(self.s, np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            energy_score(self.s, np.zeros((1, 2)))  # no broadcast
        with self.assertRaises(ValueError):
            energy_score(self.s, np.zeros(2))

    def test_empty_axes(self):
        for shape in ((0, 3, 2), (2, 0, 2), (2, 3, 0)):
            with self.assertRaises(ValueError):
                energy_score(np.zeros(shape), np.zeros((shape[0], shape[2])))

    def test_nonnumeric_dtypes(self):
        for bad in (self.s.astype(bool), self.s.astype(complex), self.s.astype(str)):
            with self.assertRaises(TypeError):
                energy_score(bad, self.t)
        with self.assertRaises(TypeError):
            energy_score(self.s, self.t.astype(bool))

    def test_nonfinite_inputs(self):
        for bad in (np.nan, np.inf, -np.inf):
            s = self.s.copy()
            s[1, 2, 1] = bad
            with self.assertRaises(ValueError):
                energy_score(s, self.t)
            t = self.t.copy()
            t[0, 0] = bad
            with self.assertRaises(ValueError):
                energy_score(self.s, t)

    def test_nonfinite_sample_with_zero_weight_still_rejected(self):
        s = self.s.copy()
        s[0, 0, 0] = np.nan
        w = np.ones((2, 3))
        w[0, 0] = 0.0
        with self.assertRaises(ValueError):
            energy_score(s, self.t, w)

    def test_bad_weights(self):
        bad_weights = [
            np.ones((3,)),  # broadcast candidate
            np.ones((2, 1)),
            np.ones((1, 3)),
            np.ones((2, 4)),
            np.full((2, 3), -0.1),
            np.array([[1.0, np.nan, 1.0]] * 2),
            np.array([[1.0, np.inf, 1.0]] * 2),
            np.zeros((2, 3)),
            np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),  # one all-zero row
        ]
        for w in bad_weights:
            with self.assertRaises(ValueError, msg=f"weights {w!r}"):
                energy_score(self.s, self.t, w)

    def test_inputs_not_mutated(self):
        samples, truth = _fixture(2, 6, 3)
        w = _weights(2, 6)
        s0, t0, w0 = samples.copy(), truth.copy(), w.copy()
        energy_score(samples, truth, w)
        np.testing.assert_array_equal(samples, s0)
        np.testing.assert_array_equal(truth, t0)
        np.testing.assert_array_equal(w, w0)

    def test_deterministic_repeat(self):
        samples, truth = _fixture(2, 8, 3)
        np.testing.assert_array_equal(energy_score(samples, truth), energy_score(samples, truth))


if __name__ == "__main__":
    unittest.main()
