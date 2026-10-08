"""Tests for ``skbel.metrics.calibration``: SBC ranks, PIT, central intervals and streams.

Hand-computed cases fix the conventions; seeded finite-sample checks with generous
binomial bounds cover the exchangeable null and a too-narrow posterior.
"""

import re
import unittest
from pathlib import Path

import numpy as np

from skbel.metrics import (
    CoverageSummary,
    IntervalCoverage,
    case_rng,
    empirical_pit,
    interval_coverage,
    sbc_rank_histogram,
    sbc_ranks,
    summarize_coverage,
)

_DOC = Path(__file__).resolve().parents[2] / "docs" / "calibration.md"


def _state():
    return np.random.get_state()


def _same_state(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and tuple(a[2:]) == tuple(b[2:])


def _tied(cases=400, targets=2):
    """Draws [1, 2, 2, 2, 3] and truth 2 for every case and target."""
    draws = np.array([1.0, 2.0, 2.0, 2.0, 3.0])
    samples = np.broadcast_to(draws[None, :, None], (cases, 5, targets)).copy()
    return samples, np.full((cases, targets), 2.0)


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self.samples = np.arange(12.0).reshape(2, 3, 2)
        self.truth = np.zeros((2, 2))
        self.functions = (
            lambda s, t: sbc_ranks(s, t, seed=0),
            lambda s, t: empirical_pit(s, t, seed=0),
            lambda s, t: interval_coverage(s, t, 0.5),
        )

    def test_shapes_types_and_finiteness(self):
        bad = [
            (np.zeros((2, 3)), self.truth, ValueError),
            (self.samples, np.zeros(2), ValueError),
            (self.samples, np.zeros((2, 3)), ValueError),
            (self.samples, np.zeros((1, 2)), ValueError),
            (np.zeros((0, 3, 2)), np.zeros((0, 2)), ValueError),
            (np.zeros((2, 0, 2)), self.truth, ValueError),
            (np.where(self.samples > 5, np.nan, self.samples), self.truth, ValueError),
            (self.samples, np.full((2, 2), np.inf), ValueError),
            (self.samples > 3, self.truth, TypeError),
            (self.samples.astype(str), self.truth, TypeError),
        ]
        for function in self.functions:
            for samples, truth, error in bad:
                with self.subTest(shape=np.shape(samples)), self.assertRaises(error):
                    function(samples, truth)

    def test_options_seeds_and_ids(self):
        s, t = self.samples, self.truth
        with self.assertRaises(ValueError):
            sbc_ranks(s, t, ties="midpoint")
        with self.assertRaises(ValueError):
            empirical_pit(s, t, ties="up")
        for function in (sbc_ranks, empirical_pit):
            with self.assertRaisesRegex(ValueError, "requires an integer seed"):
                function(s, t)
            for seed, error in ((True, TypeError), (1.0, TypeError), (-1, ValueError)):
                with self.subTest(seed=seed), self.assertRaises(error):
                    function(s, t, seed=seed)
            bad_ids = [
                ([0], ValueError),
                ([0, 0], ValueError),
                ([0, -1], ValueError),
                ([0, 1.5], TypeError),
                ([0, True], TypeError),
                (np.array([[0, 1]]), ValueError),
                ("ab", TypeError),
                (7, TypeError),
            ]
            for ids, error in bad_ids:
                with self.subTest(ids=ids):
                    with self.assertRaises(error):
                        function(s, t, seed=0, case_ids=ids)
                    with self.assertRaises(error):
                        function(s, t, seed=0, target_ids=ids)

    def test_weights_and_levels(self):
        s, t = self.samples, self.truth
        for weights in (np.ones((2, 2)), -np.ones((2, 3)), np.zeros((2, 3))):
            with self.assertRaises(ValueError):
                empirical_pit(s, t, weights=weights, ties="midpoint")
            with self.assertRaises(ValueError):
                interval_coverage(s, t, 0.5, weights=weights)
        for levels in (0.0, 1.0, [0.5, np.nan], [], [[0.5]], -0.2):
            with self.subTest(levels=levels), self.assertRaises(ValueError):
                interval_coverage(s, t, levels)
        with self.assertRaises(TypeError):
            sbc_ranks(s, t, seed=0, weights=np.ones((2, 3)))
        with self.assertRaises(TypeError):
            summarize_coverage(np.ones((2, 1, 2), dtype=bool))

    def test_inputs_and_global_state_are_not_modified(self):
        samples, truth = _tied()
        copies = samples.copy(), truth.copy()
        before = _state()
        sbc_ranks(samples, truth, seed=1)
        empirical_pit(samples, truth, seed=1)
        empirical_pit(samples, truth, weights=np.ones(samples.shape[:2]), seed=1)
        summarize_coverage(interval_coverage(samples, truth, [0.5, 0.9]))
        case_rng(1, "a", "x").random(3)
        self.assertTrue(_same_state(before, _state()))
        np.testing.assert_array_equal(samples, copies[0])
        np.testing.assert_array_equal(truth, copies[1])


class RankTest(unittest.TestCase):
    def test_hand_computed_ranks_without_ties(self):
        samples = np.array([[[1.0], [2.0], [3.0]]] * 3)
        truth = np.array([[2.5], [0.0], [4.0]])
        expected = np.array([[2], [0], [3]])
        np.testing.assert_array_equal(sbc_ranks(samples, truth, ties="error"), expected)
        np.testing.assert_array_equal(sbc_ranks(samples, truth, seed=9), expected)
        self.assertEqual(sbc_ranks(samples, truth, ties="error").dtype, np.int64)
        with self.assertRaisesRegex(ValueError, "equal the truth"):
            sbc_ranks(samples, np.array([[2.0], [0.0], [4.0]]), ties="error")

    def test_randomized_ties_are_uniform_over_tied_positions(self):
        samples, truth = _tied(cases=4000, targets=1)
        ranks = sbc_ranks(samples, truth, seed=3)
        self.assertTrue(set(np.unique(ranks)) <= {1, 2, 3, 4})
        counts = np.bincount(ranks[:, 0], minlength=5)[1:]
        sd = np.sqrt(4000 * 0.25 * 0.75)
        self.assertTrue(np.all(np.abs(counts - 1000) < 5 * sd), counts)

    def test_point_mass_posterior_gives_uniform_ranks(self):
        """Deterministic posterior equal to the truth: every position is a tie."""
        cases, draws = 3000, 4
        samples = np.full((cases, draws, 1), 0.5)
        ranks = sbc_ranks(samples, np.full((cases, 1), 0.5), seed=0)
        counts = sbc_rank_histogram(ranks, draws)
        self.assertEqual(counts.shape, (draws + 1, 1))
        sd = np.sqrt(cases * 0.2 * 0.8)
        self.assertTrue(np.all(np.abs(counts[:, 0] - cases / 5) < 5 * sd), counts)

    def test_streams_follow_explicit_ids_under_reorder_and_subset(self):
        samples, truth = _tied(cases=40, targets=2)
        ids = [f"case-{i}" for i in range(40)]
        full = sbc_ranks(samples, truth, seed=5, case_ids=ids)
        np.testing.assert_array_equal(full, sbc_ranks(samples, truth, seed=5, case_ids=ids))
        order = np.random.default_rng(1).permutation(40)
        reordered = sbc_ranks(
            samples[order], truth[order], seed=5, case_ids=[ids[i] for i in order]
        )
        np.testing.assert_array_equal(reordered, full[order])
        subset = sbc_ranks(samples[7:9], truth[7:9], seed=5, case_ids=ids[7:9])
        np.testing.assert_array_equal(subset, full[7:9])
        # Swapping target ids swaps the target columns.
        swapped = sbc_ranks(samples, truth, seed=5, case_ids=ids, target_ids=[1, 0])
        np.testing.assert_array_equal(swapped, full[:, ::-1])

    def test_streams_distinguish_targets_seeds_and_id_types(self):
        samples, truth = _tied(cases=60, targets=2)
        ranks = sbc_ranks(samples, truth, seed=5)
        self.assertFalse(np.array_equal(ranks[:, 0], ranks[:, 1]))
        self.assertFalse(np.array_equal(ranks, sbc_ranks(samples, truth, seed=6)))
        as_strings = sbc_ranks(samples, truth, seed=5, case_ids=[str(i) for i in range(60)])
        self.assertFalse(np.array_equal(ranks, as_strings))
        # Default positional ids equal explicit integer ids.
        np.testing.assert_array_equal(
            ranks, sbc_ranks(samples, truth, seed=5, case_ids=np.arange(60))
        )

    def test_histogram_bins_and_validation(self):
        ranks = np.array([[0, 5], [1, 4], [2, 3], [5, 0]])
        counts = sbc_rank_histogram(ranks, n_draws=5, n_bins=3)
        np.testing.assert_array_equal(counts, [[2, 1], [1, 1], [1, 2]])
        self.assertEqual(counts.sum(), ranks.size)
        for kwargs in ({"n_draws": 5, "n_bins": 4}, {"n_draws": 4}, {"n_draws": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                sbc_rank_histogram(ranks, **kwargs)
        with self.assertRaises(TypeError):
            sbc_rank_histogram(ranks.astype(float), n_draws=5)
        with self.assertRaises(ValueError):
            sbc_rank_histogram(ranks[0], n_draws=5)


class PITTest(unittest.TestCase):
    def test_hand_computed_unweighted_conventions(self):
        samples = np.array([[[1.0], [2.0], [2.0], [4.0]]])
        np.testing.assert_allclose(empirical_pit(samples, [[3.0]], ties="error"), [[0.75]])
        np.testing.assert_allclose(empirical_pit(samples, [[2.0]], ties="midpoint"), [[0.5]])
        pit = empirical_pit(samples, [[2.0]], seed=4)
        self.assertTrue(0.25 <= pit[0, 0] < 0.75)
        with self.assertRaises(ValueError):
            empirical_pit(samples, [[2.0]], ties="error")
        np.testing.assert_allclose(empirical_pit(samples, [[0.0]], ties="error"), [[0.0]])
        np.testing.assert_allclose(empirical_pit(samples, [[9.0]], ties="error"), [[1.0]])

    def test_hand_computed_weighted(self):
        samples = np.array([[[0.0], [1.0], [5.0]]])
        weights = np.array([[1.0, 3.0, 0.0]])
        np.testing.assert_allclose(
            empirical_pit(samples, [[0.5]], weights=weights, ties="error"), [[0.25]]
        )
        np.testing.assert_allclose(
            empirical_pit(samples, [[1.0]], weights=weights, ties="midpoint"), [[0.625]]
        )
        # A zero-weight draw equal to the truth is not an atom.
        np.testing.assert_allclose(
            empirical_pit(samples, [[5.0]], weights=weights, ties="error"), [[1.0]]
        )
        # Uniform weights reproduce the unweighted PIT, randomized ties included.
        s, t = _tied(cases=30)
        np.testing.assert_allclose(
            empirical_pit(s, t, weights=np.full((30, 5), 2.0), seed=8),
            empirical_pit(s, t, seed=8),
            rtol=0,
            atol=1e-15,
        )

    def test_randomized_pit_and_rank_share_the_tie_position(self):
        samples, truth = _tied(cases=200)
        ranks = sbc_ranks(samples, truth, seed=12)
        pit = empirical_pit(samples, truth, seed=12)
        # rank = 1 + floor(v * 4) and pit = (1 + 3 v) / 5 for the same v.
        v = (pit * 5 - 1) / 3
        np.testing.assert_array_equal(ranks, 1 + np.minimum(np.floor(v * 4 + 1e-12), 3))


class IntervalTest(unittest.TestCase):
    def test_hand_computed_unweighted_endpoints(self):
        samples = np.arange(1.0, 11.0)[::-1].reshape(1, 10, 1)  # draws 10..1, unsorted
        result = interval_coverage(samples, [[9.0]], [0.8, 0.5])
        self.assertIsInstance(result, IntervalCoverage)
        self.assertEqual(result.lower.shape, (1, 2, 1))
        np.testing.assert_array_equal(result.lower[0, :, 0], [1.0, 3.0])
        np.testing.assert_array_equal(result.upper[0, :, 0], [9.0, 8.0])
        np.testing.assert_array_equal(result.width[0, :, 0], [8.0, 5.0])
        # Closed interval: a truth on the endpoint is covered.
        np.testing.assert_array_equal(result.covered[0, :, 0], [True, False])
        np.testing.assert_allclose(result.exchangeable_coverage, [8 / 11, 5 / 11])

    def test_scalar_level_and_degenerate_posterior(self):
        samples = np.full((2, 6, 1), 3.0)
        result = interval_coverage(samples, [[3.0], [3.5]], 0.9)
        self.assertEqual(result.levels.shape, (1,))
        np.testing.assert_array_equal(result.width, 0.0)
        np.testing.assert_array_equal(result.covered[:, 0, 0], [True, False])

    def test_weighted_endpoints_skip_zero_mass(self):
        samples = np.array([[[0.0], [1.0], [2.0], [3.0]]])
        result = interval_coverage(samples, [[1.5]], 0.5, weights=[[0.0, 1.0, 1.0, 0.0]])
        np.testing.assert_array_equal(result.lower[0, 0], [1.0])
        np.testing.assert_array_equal(result.upper[0, 0], [2.0])
        self.assertIsNone(result.exchangeable_coverage)
        # Uniform weights reproduce the unweighted endpoints.
        rng = np.random.default_rng(2)
        s, t = rng.normal(size=(20, 50, 3)), rng.normal(size=(20, 3))
        levels = [0.5, 0.8, 0.9]
        plain = interval_coverage(s, t, levels)
        weighted = interval_coverage(s, t, levels, weights=np.full((20, 50), 0.3))
        np.testing.assert_array_equal(plain.lower, weighted.lower)
        np.testing.assert_array_equal(plain.upper, weighted.upper)

    def test_interval_holds_at_least_the_nominal_empirical_mass(self):
        rng = np.random.default_rng(3)
        s, t = rng.normal(size=(30, 37, 2)), np.zeros((30, 2))
        w = rng.uniform(0.1, 2.0, size=(30, 37))
        for weights in (None, w):
            result = interval_coverage(s, t, [0.5, 0.9], weights=weights)
            mass_w = np.full((30, 37), 1 / 37) if weights is None else w / w.sum(1, keepdims=True)
            inside = (s[:, :, None, :] >= result.lower[:, None]) & (
                s[:, :, None, :] <= result.upper[:, None]
            )
            mass = np.einsum("cd,cdlt->clt", mass_w, inside)
            self.assertTrue(np.all(mass >= np.array([0.5, 0.9])[None, :, None] - 1e-12))

    def test_summary_shapes_and_values(self):
        samples = np.arange(1.0, 11.0).reshape(1, 10, 1).repeat(4, axis=0)
        truth = np.array([[0.0], [5.0], [9.5], [5.5]])
        summary = summarize_coverage(interval_coverage(samples, truth, [0.8]))
        self.assertIsInstance(summary, CoverageSummary)
        self.assertEqual(summary.n_cases, 4)
        np.testing.assert_allclose(summary.coverage, [[0.5]])
        np.testing.assert_allclose(summary.nominal_se, [np.sqrt(0.8 * 0.2 / 4)])
        np.testing.assert_allclose(summary.mean_width, [[8.0]])

    def test_unrepresentable_width_is_rejected(self):
        # Finite endpoints whose difference overflows to inf.
        samples = np.array([[[-1e308], [1e308]]])
        with np.errstate(over="raise"), self.assertRaisesRegex(ValueError, "interval width"):
            interval_coverage(samples, [[0.0]], 0.5)

    def test_mean_of_finite_widths_near_float_limit_stays_finite(self):
        # Each width is finite, but their plain sum overflows to inf.
        samples = np.array([[[0.0], [1e308]], [[0.0], [1e308]], [[0.0], [5e307]]])
        truth = np.zeros((3, 1))
        with np.errstate(over="raise"):
            summary = summarize_coverage(interval_coverage(samples, truth, 0.5))
        np.testing.assert_allclose(summary.mean_width, [[1e308 * (2.5 / 3)]], rtol=1e-12)
        self.assertTrue(np.all(np.isfinite(summary.mean_width)))
        # A hand-built result with a non-finite width is rejected, not averaged.
        result = interval_coverage(samples, truth, 0.5)
        bad = result._replace(width=np.full_like(result.width, np.inf))
        with self.assertRaisesRegex(ValueError, "interval width"):
            summarize_coverage(bad)


class FiniteSampleControlTest(unittest.TestCase):
    """Seeded exchangeable null and a too-narrow posterior, with generous bounds."""

    CASES, DRAWS = 3000, 19

    def _draws(self, spread):
        rng = np.random.default_rng(21)
        mean = rng.normal(size=(self.CASES, 1, 2))
        truth = mean[:, 0, :] + rng.normal(size=(self.CASES, 2))
        samples = mean + spread * rng.normal(size=(self.CASES, self.DRAWS, 2))
        return samples, truth

    def test_exchangeable_null(self):
        samples, truth = self._draws(spread=1.0)
        ranks = sbc_ranks(samples, truth, seed=0)
        counts = sbc_rank_histogram(ranks, self.DRAWS, n_bins=5)
        expected, p = self.CASES / 5, 0.2
        self.assertTrue(np.all(np.abs(counts - expected) < 5 * np.sqrt(self.CASES * p * (1 - p))))
        summary = summarize_coverage(interval_coverage(samples, truth, [0.5, 0.9]))
        ref = summary.exchangeable_coverage[:, None]
        bound = 5 * np.sqrt(ref * (1 - ref) / self.CASES)
        self.assertTrue(np.all(np.abs(summary.coverage - ref) < bound), summary.coverage)
        pit = empirical_pit(samples, truth, ties="error")
        self.assertLess(abs(pit.mean() - 0.5), 0.02)

    def test_too_narrow_posterior_is_flagged(self):
        samples, truth = self._draws(spread=0.5)
        counts = sbc_rank_histogram(sbc_ranks(samples, truth, seed=0), self.DRAWS, n_bins=5)
        # Truth falls in the tails too often: edge bins overfull, middle bin underfull.
        self.assertTrue(np.all(counts[0] > 1.5 * self.CASES / 5))
        self.assertTrue(np.all(counts[-1] > 1.5 * self.CASES / 5))
        self.assertTrue(np.all(counts[2] < 0.75 * self.CASES / 5))
        summary = summarize_coverage(interval_coverage(samples, truth, 0.9))
        self.assertTrue(np.all(summary.coverage < summary.exchangeable_coverage[0] - 0.2))


class CaseRngTest(unittest.TestCase):
    def test_streams_are_labeled_and_fixed_width(self):
        draw = lambda *args, **kw: case_rng(*args, **kw).random(4)  # noqa: E731
        np.testing.assert_array_equal(draw(1, 0, "a"), draw(np.int64(1), np.uint8(0), "a"))
        distinct = [
            draw(1, 0, "a"),
            draw(2, 0, "a"),
            draw(1, 1, "a"),
            draw(1, "0", "a"),
            draw(1, 0, "b"),
            draw(1, 0, "a", target_id=0),
            draw(1, 0, "a", target_id="0"),
            draw(1, 2**40, "a"),
        ]
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                self.assertFalse(np.array_equal(distinct[i], distinct[j]), (i, j))

    def test_invalid_arguments(self):
        for args, error in (
            ((None, 0, "a"), TypeError),
            ((-1, 0, "a"), ValueError),
            ((1, -1, "a"), ValueError),
            ((1, 0.0, "a"), TypeError),
            ((1, 0, 3), TypeError),
        ):
            with self.subTest(args=args), self.assertRaises(error):
                case_rng(*args)
        with self.assertRaises(TypeError):
            case_rng(1, 0, "a", target_id=False)


class DocumentationExampleTest(unittest.TestCase):
    def test_python_blocks_in_calibration_page_run(self):
        blocks = re.findall(r"```python\n(.*?)```", _DOC.read_text(encoding="utf-8"), re.S)
        self.assertTrue(blocks)
        namespace = {}
        before = _state()
        for block in blocks:
            exec(compile(block, str(_DOC), "exec"), namespace)  # noqa: S102
        self.assertTrue(_same_state(before, _state()))


if __name__ == "__main__":
    unittest.main()
