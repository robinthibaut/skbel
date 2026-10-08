"""Array-level checks of ``skbel.evaluation``: scores, events, Brier scores and decision risks.

Every expected value is either worked out by hand or recomputed with plain Python loops
from the definitions, on small arrays whose entries are dyadic rationals so that sums are
exact in float64 and ties are real ties.
"""

import inspect
import unittest

import numpy as np

from skbel import evaluation
from skbel.evaluation import (
    DecisionRisks,
    SampleScores,
    decision_risks,
    event_brier_scores,
    event_probabilities,
    score_samples,
)
from skbel.metrics import (
    IntervalCoverage,
    energy_score,
    interval_coverage,
    marginal_crps,
    summarize_coverage,
)

# (cases=2, draws=4, targets=2)
SAMPLES = np.array(
    [
        [[1.0, 0.0], [2.0, 0.0], [3.0, 2.0], [4.0, 2.0]],
        [[5.0, -1.0], [5.0, 0.0], [6.0, 1.0], [8.0, 2.0]],
    ]
)
TRUTH = np.array([[2.5, 1.0], [6.0, 0.5]])
# Row sums are 8 and 16; the second case has zero-weight draws.
WEIGHTS = np.array([[1.0, 1.0, 2.0, 4.0], [8.0, 0.0, 0.0, 8.0]])
LEVELS = [0.5, 0.9]


def _normalize(w):
    w = np.asarray(w, dtype=float)
    return w / w.sum()


def _crps(x, y, w):
    """CRPS of the weighted empirical law: E|X - y| - 0.5 E|X - X'|."""
    w = _normalize(w)
    first = sum(w[i] * abs(x[i] - y) for i in range(len(x)))
    second = sum(w[i] * w[j] * abs(x[i] - x[j]) for i in range(len(x)) for j in range(len(x)))
    return first - 0.5 * second


def _quantile(x, w, p):
    """Inverse-CDF quantile: the smallest draw whose cumulative weight reaches ``p``."""
    w = _normalize(w)
    order = sorted(range(len(x)), key=lambda i: x[i])
    cumulative = 0.0
    for i in order:
        cumulative += w[i]
        if cumulative >= p:
            return x[i]
    return x[order[-1]]


def _energy(x, y, w):
    w = _normalize(w)
    first = sum(w[i] * np.linalg.norm(x[i] - y) for i in range(len(x)))
    second = sum(
        w[i] * w[j] * np.linalg.norm(x[i] - x[j]) for i in range(len(x)) for j in range(len(x))
    )
    return first - 0.5 * second


def _frozen(*arrays):
    return [np.array(a, copy=True) for a in arrays]


class PublicApiTest(unittest.TestCase):
    def test_exported_names(self):
        self.assertEqual(
            sorted(evaluation.__all__),
            [
                "DecisionRisks",
                "SampleScores",
                "decision_risks",
                "event_brier_scores",
                "event_probabilities",
                "sample_bel_posterior",
                "score_samples",
            ],
        )
        for name in evaluation.__all__:
            self.assertTrue(hasattr(evaluation, name), name)

    def test_required_choices_are_keyword_only(self):
        levels = inspect.signature(score_samples).parameters["levels"]
        self.assertEqual(levels.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(levels.default, inspect.Parameter.empty)
        convention = inspect.signature(event_brier_scores).parameters["convention"]
        self.assertEqual(convention.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(convention.default, inspect.Parameter.empty)
        for function in (event_probabilities, decision_risks):
            weights = inspect.signature(function).parameters["weights"]
            self.assertEqual(weights.kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertIsNone(weights.default)

    def test_decision_risks_accepts_no_truth_or_outcome(self):
        self.assertEqual(list(inspect.signature(decision_risks).parameters), ["losses", "weights"])

    def test_result_types_are_named_tuples(self):
        self.assertEqual(SampleScores._fields, ("crps", "coverage", "energy"))
        self.assertEqual(DecisionRisks._fields, ("expected_losses", "min_risk", "bayes_actions"))


class ScoreSamplesTest(unittest.TestCase):
    def test_hand_computed_crps(self):
        samples = np.array([[[0.0], [1.0]]])
        for truth, expected in ((0.0, 0.25), (0.5, 0.25), (1.0, 0.25), (2.0, 1.25), (-1.0, 1.25)):
            with self.subTest(truth=truth):
                scores = score_samples(samples, np.array([[truth]]), levels=0.5)
                self.assertEqual(scores.crps.shape, (1, 1))
                self.assertEqual(scores.crps[0, 0], expected)

    def test_crps_matches_independent_sums(self):
        for weights in (None, WEIGHTS):
            with self.subTest(weighted=weights is not None):
                scores = score_samples(SAMPLES, TRUTH, levels=LEVELS, weights=weights)
                for c in range(2):
                    w = np.ones(4) if weights is None else weights[c]
                    for t in range(2):
                        np.testing.assert_allclose(
                            scores.crps[c, t],
                            _crps(SAMPLES[c, :, t], TRUTH[c, t], w),
                            rtol=0,
                            atol=1e-15,
                        )

    def test_coverage_matches_independent_quantiles(self):
        for weights in (None, WEIGHTS):
            with self.subTest(weighted=weights is not None):
                scores = score_samples(SAMPLES, TRUTH, levels=LEVELS, weights=weights)
                cov = scores.coverage
                self.assertIsInstance(cov, IntervalCoverage)
                self.assertEqual(cov.lower.shape, (2, 2, 2))
                np.testing.assert_array_equal(cov.levels, LEVELS)
                for c in range(2):
                    w = np.ones(4) if weights is None else weights[c]
                    for k, level in enumerate(LEVELS):
                        for t in range(2):
                            x = SAMPLES[c, :, t]
                            lower = _quantile(x, w, (1.0 - level) / 2.0)
                            upper = _quantile(x, w, (1.0 + level) / 2.0)
                            self.assertEqual(cov.lower[c, k, t], lower)
                            self.assertEqual(cov.upper[c, k, t], upper)
                            self.assertEqual(cov.width[c, k, t], upper - lower)
                            self.assertEqual(
                                bool(cov.covered[c, k, t]), lower <= TRUTH[c, t] <= upper
                            )

    def test_hand_computed_interval(self):
        samples = np.array([[[1.0], [2.0], [3.0], [4.0]]])
        inside = score_samples(samples, np.array([[2.5]]), levels=0.5).coverage
        self.assertEqual((inside.lower[0, 0, 0], inside.upper[0, 0, 0]), (1.0, 3.0))
        self.assertTrue(inside.covered[0, 0, 0])
        outside = score_samples(samples, np.array([[3.5]]), levels=0.5).coverage
        self.assertFalse(outside.covered[0, 0, 0])
        edge = score_samples(samples, np.array([[3.0]]), levels=0.5).coverage
        self.assertTrue(edge.covered[0, 0, 0], "the interval is closed")

    def test_energy_matches_independent_sums_and_is_optional(self):
        self.assertIsNone(score_samples(SAMPLES, TRUTH, levels=LEVELS).energy)
        for weights in (None, WEIGHTS):
            with self.subTest(weighted=weights is not None):
                scores = score_samples(SAMPLES, TRUTH, levels=LEVELS, weights=weights, joint=True)
                self.assertEqual(scores.energy.shape, (2,))
                for c in range(2):
                    w = np.ones(4) if weights is None else weights[c]
                    np.testing.assert_allclose(
                        scores.energy[c], _energy(SAMPLES[c], TRUTH[c], w), rtol=0, atol=1e-14
                    )

    def test_energy_of_one_target_is_the_crps(self):
        samples = SAMPLES[:, :, :1]
        truth = TRUTH[:, :1]
        for weights in (None, WEIGHTS):
            scores = score_samples(samples, truth, levels=0.5, weights=weights, joint=True)
            np.testing.assert_allclose(scores.energy, scores.crps[:, 0], rtol=0, atol=1e-14)

    def test_functions_agree_with_the_metrics_they_wrap(self):
        for weights in (None, WEIGHTS):
            scores = score_samples(SAMPLES, TRUTH, levels=LEVELS, weights=weights, joint=True)
            np.testing.assert_array_equal(scores.crps, marginal_crps(SAMPLES, TRUTH, weights))
            np.testing.assert_array_equal(scores.energy, energy_score(SAMPLES, TRUTH, weights))
            reference = interval_coverage(SAMPLES, TRUTH, LEVELS, weights=weights)
            for field in ("levels", "lower", "upper", "width", "covered"):
                np.testing.assert_array_equal(
                    getattr(scores.coverage, field), getattr(reference, field)
                )

    def test_coverage_summary_is_a_separate_explicit_step(self):
        scores = score_samples(SAMPLES, TRUTH, levels=LEVELS)
        summary = summarize_coverage(scores.coverage)
        self.assertEqual(summary.n_cases, 2)
        self.assertEqual(summary.coverage.shape, (2, 2))

    def test_unbiased_crps_matches_the_pair_formula(self):
        scores = score_samples(SAMPLES, TRUTH, levels=0.5, crps_estimator="unbiased")
        m = SAMPLES.shape[1]
        for c in range(2):
            for t in range(2):
                x = SAMPLES[c, :, t]
                mean_abs = sum(abs(v - TRUTH[c, t]) for v in x) / m
                pairs = sum(abs(x[i] - x[j]) for i in range(m) for j in range(m) if i != j)
                np.testing.assert_allclose(
                    scores.crps[c, t], mean_abs - pairs / (2 * m * (m - 1)), rtol=0, atol=1e-15
                )
        pair = score_samples(
            np.array([[[0.0], [1.0]]]), np.array([[0.0]]), levels=0.5, crps_estimator="unbiased"
        )
        self.assertEqual(pair.crps[0, 0], 0.0)

    def test_unbiased_requires_unweighted_draws_and_two_of_them(self):
        with self.assertRaises(ValueError):
            score_samples(SAMPLES, TRUTH, levels=0.5, weights=WEIGHTS, crps_estimator="unbiased")
        with self.assertRaises(ValueError):
            score_samples(SAMPLES[:, :1], TRUTH, levels=0.5, crps_estimator="unbiased")

    def test_estimator_must_be_known(self):
        for bad in ("fair", "", None, 1):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                score_samples(SAMPLES, TRUTH, levels=0.5, crps_estimator=bad)

    def test_results_are_per_case_and_do_not_depend_on_other_cases(self):
        full = score_samples(SAMPLES, TRUTH, levels=LEVELS, weights=WEIGHTS, joint=True)
        for c in range(2):
            alone = score_samples(
                SAMPLES[c : c + 1],
                TRUTH[c : c + 1],
                levels=LEVELS,
                weights=WEIGHTS[c : c + 1],
                joint=True,
            )
            np.testing.assert_array_equal(alone.crps[0], full.crps[c])
            np.testing.assert_array_equal(alone.energy[0], full.energy[c])
            np.testing.assert_array_equal(alone.coverage.lower[0], full.coverage.lower[c])
            np.testing.assert_array_equal(alone.coverage.covered[0], full.coverage.covered[c])

    def test_weights_are_renormalized_per_case(self):
        base = score_samples(SAMPLES, TRUTH, levels=LEVELS, weights=WEIGHTS, joint=True)
        scale = np.array([[3.0], [0.25]])
        scaled = score_samples(SAMPLES, TRUTH, levels=LEVELS, weights=WEIGHTS * scale, joint=True)
        np.testing.assert_allclose(scaled.crps, base.crps, rtol=0, atol=1e-14)
        np.testing.assert_allclose(scaled.energy, base.energy, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(scaled.coverage.lower, base.coverage.lower)
        np.testing.assert_array_equal(scaled.coverage.upper, base.coverage.upper)

    def test_uniform_weights_equal_no_weights(self):
        uniform = np.full((2, 4), 3.0)
        a = score_samples(SAMPLES, TRUTH, levels=LEVELS, joint=True)
        b = score_samples(SAMPLES, TRUTH, levels=LEVELS, weights=uniform, joint=True)
        np.testing.assert_allclose(a.crps, b.crps, rtol=0, atol=1e-14)
        np.testing.assert_allclose(a.energy, b.energy, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(a.coverage.lower, b.coverage.lower)
        np.testing.assert_array_equal(a.coverage.upper, b.coverage.upper)

    def test_zero_weight_draws_carry_no_mass(self):
        # Case 1 has two zero-weight draws; replace their values by far-away finite ones.
        moved = SAMPLES.copy()
        moved[1, 1] = [1.0e6, -1.0e6]
        moved[1, 2] = [-1.0e6, 1.0e6]
        a = score_samples(SAMPLES, TRUTH, levels=LEVELS, weights=WEIGHTS, joint=True)
        b = score_samples(moved, TRUTH, levels=LEVELS, weights=WEIGHTS, joint=True)
        np.testing.assert_allclose(b.crps, a.crps, rtol=0, atol=1e-9)
        np.testing.assert_allclose(b.energy, a.energy, rtol=0, atol=1e-9)
        np.testing.assert_array_equal(b.coverage.lower, a.coverage.lower)
        np.testing.assert_array_equal(b.coverage.upper, a.coverage.upper)
        # Dropping the zero-weight draws altogether gives the same case-1 scores.
        keep = [0, 3]
        dropped = score_samples(
            SAMPLES[1:, keep], TRUTH[1:], levels=LEVELS, weights=WEIGHTS[1:, keep], joint=True
        )
        np.testing.assert_allclose(dropped.crps[0], a.crps[1], rtol=0, atol=1e-14)
        np.testing.assert_allclose(dropped.energy[0], a.energy[1], rtol=0, atol=1e-14)
        np.testing.assert_array_equal(dropped.coverage.lower[0], a.coverage.lower[1])

    def test_malformed_values_raise_even_at_zero_weight(self):
        for bad in (np.nan, np.inf, -np.inf):
            for joint in (False, True):
                with self.subTest(bad=bad, joint=joint):
                    samples = SAMPLES.copy()
                    samples[1, 1, 0] = bad  # zero-weight draw of case 1
                    with self.assertRaises(ValueError):
                        score_samples(samples, TRUTH, levels=LEVELS, weights=WEIGHTS, joint=joint)

    def test_nonfinite_samples_truth_and_levels_raise(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(bad=bad):
                samples = SAMPLES.copy()
                samples[0, 0, 0] = bad
                with self.assertRaises(ValueError):
                    score_samples(samples, TRUTH, levels=LEVELS)
                truth = TRUTH.copy()
                truth[0, 0] = bad
                with self.assertRaises(ValueError):
                    score_samples(SAMPLES, truth, levels=LEVELS)
                with self.assertRaises(ValueError):
                    score_samples(SAMPLES, TRUTH, levels=[0.5, bad])

    def test_shapes_are_exact_with_no_broadcasting(self):
        cases = (
            ("truth one-dimensional", SAMPLES, TRUTH[:, 0]),
            ("truth one target", SAMPLES, TRUTH[:, :1]),
            ("truth one case", SAMPLES, TRUTH[:1]),
            ("truth extra case", SAMPLES, np.vstack([TRUTH, TRUTH[:1]])),
            ("truth three-dimensional", SAMPLES, TRUTH[None]),
            ("samples two-dimensional", SAMPLES[:, :, 0], TRUTH),
            ("samples four-dimensional", SAMPLES[None], TRUTH),
            ("no cases", SAMPLES[:0], TRUTH[:0]),
            ("no draws", SAMPLES[:, :0], TRUTH),
            ("no targets", SAMPLES[:, :, :0], TRUTH[:, :0]),
        )
        for label, samples, truth in cases:
            for joint in (False, True):
                with self.subTest(label=label, joint=joint), self.assertRaises(ValueError):
                    score_samples(samples, truth, levels=LEVELS, joint=joint)

    def test_non_numeric_arrays_are_rejected(self):
        bad_samples = (
            SAMPLES > 0,
            SAMPLES.astype(complex),
            SAMPLES.astype(object),
            SAMPLES.astype(str),
        )
        for bad in bad_samples:
            with self.subTest(dtype=bad.dtype), self.assertRaises((TypeError, ValueError)):
                score_samples(bad, TRUTH, levels=LEVELS)
            with self.subTest(truth_dtype=bad.dtype), self.assertRaises((TypeError, ValueError)):
                score_samples(SAMPLES, bad[:, 0], levels=LEVELS)

    def test_integer_arrays_are_accepted(self):
        samples = np.arange(24).reshape(2, 4, 3)
        truth = np.array([[1, 2, 3], [20, 21, 22]])
        scores = score_samples(samples, truth, levels=0.5, joint=True)
        np.testing.assert_allclose(
            scores.crps, marginal_crps(samples.astype(float), truth.astype(float)), atol=1e-15
        )

    def test_weights_are_validated(self):
        zero_total = WEIGHTS.copy()
        zero_total[0] = 0.0
        negative = WEIGHTS.copy()
        negative[0, 0] = -1.0
        bad_values = {
            "one-dimensional": WEIGHTS[0],
            "single row": WEIGHTS[:1],
            "single column": WEIGHTS[:, :1],
            "extra draw": np.hstack([WEIGHTS, WEIGHTS[:, :1]]),
            "three-dimensional": WEIGHTS[:, :, None],
            "negative": negative,
            "zero total in one case": zero_total,
            "all zero": np.zeros_like(WEIGHTS),
            "nan": np.where(WEIGHTS == 4.0, np.nan, WEIGHTS),
            "inf": np.where(WEIGHTS == 4.0, np.inf, WEIGHTS),
            "boolean": WEIGHTS > 0,
        }
        for label, weights in bad_values.items():
            for joint in (False, True):
                with self.subTest(label=label, joint=joint):
                    with self.assertRaises((ValueError, TypeError)):
                        score_samples(SAMPLES, TRUTH, levels=LEVELS, weights=weights, joint=joint)

    def test_malformed_weight_at_a_zero_weight_draw_raises(self):
        weights = WEIGHTS.copy()
        weights[1, 1] = np.nan  # a draw that would carry no mass
        with self.assertRaises(ValueError):
            score_samples(SAMPLES, TRUTH, levels=LEVELS, weights=weights)

    def test_levels_are_required_and_validated(self):
        with self.assertRaises(TypeError):
            score_samples(SAMPLES, TRUTH)
        with self.assertRaises(TypeError):
            score_samples(SAMPLES, TRUTH, LEVELS)
        with self.assertRaises(ValueError):
            score_samples(SAMPLES, TRUTH, levels=None)
        for bad in (0.0, 1.0, -0.1, 1.5, [], [0.5, 1.0], [[0.5]], "0.5"):
            with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                score_samples(SAMPLES, TRUTH, levels=bad)

    def test_levels_accept_scalar_list_and_array(self):
        scalar = score_samples(SAMPLES, TRUTH, levels=0.5).coverage
        listed = score_samples(SAMPLES, TRUTH, levels=[0.5]).coverage
        array = score_samples(SAMPLES, TRUTH, levels=np.array([0.5])).coverage
        self.assertEqual(scalar.lower.shape, (2, 1, 2))
        np.testing.assert_array_equal(scalar.lower, listed.lower)
        np.testing.assert_array_equal(scalar.lower, array.lower)

    def test_joint_must_be_a_bool(self):
        for bad in (1, 0, "yes", None, 1.0):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                score_samples(SAMPLES, TRUTH, levels=0.5, joint=bad)
        score_samples(SAMPLES, TRUTH, levels=0.5, joint=np.bool_(True))

    def test_inputs_are_not_modified(self):
        before = _frozen(SAMPLES, TRUTH, WEIGHTS)
        levels = np.array(LEVELS)
        levels_before = levels.copy()
        score_samples(SAMPLES, TRUTH, levels=levels, weights=WEIGHTS, joint=True)
        for original, kept in zip((SAMPLES, TRUTH, WEIGHTS), before, strict=True):
            np.testing.assert_array_equal(original, kept)
        np.testing.assert_array_equal(levels, levels_before)

    def test_outputs_are_float64_and_finite(self):
        scores = score_samples(
            SAMPLES.astype(np.float32), TRUTH.astype(np.float32), levels=LEVELS, joint=True
        )
        self.assertEqual(scores.crps.dtype, np.float64)
        self.assertEqual(scores.energy.dtype, np.float64)
        self.assertEqual(scores.coverage.lower.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(scores.crps)))


class EventProbabilitiesTest(unittest.TestCase):
    def test_unweighted_is_the_fraction_of_draws(self):
        indicators = np.array(
            [
                [[True, False], [True, True], [False, False], [True, False]],
                [[False, True], [False, True], [False, True], [False, True]],
            ]
        )
        probs = event_probabilities(indicators)
        np.testing.assert_array_equal(probs, [[0.75, 0.25], [0.0, 1.0]])
        self.assertEqual(probs.dtype, np.float64)
        self.assertEqual(probs.shape, (2, 2))

    def test_weighted_is_event_weight_over_total(self):
        indicators = np.array([[[True], [False], [True], [False]]])
        probs = event_probabilities(indicators, weights=np.array([[1.0, 2.0, 4.0, 1.0]]))
        self.assertEqual(probs[0, 0], 0.625)

    def test_weights_are_renormalized_per_case(self):
        indicators = np.random.default_rng(3).random((3, 6, 2)) > 0.5
        weights = np.random.default_rng(4).integers(1, 9, size=(3, 6)).astype(float)
        a = event_probabilities(indicators, weights=weights)
        b = event_probabilities(indicators, weights=weights * np.array([[5.0], [0.125], [64.0]]))
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-15)
        # Independent sums.
        for c in range(3):
            for e in range(2):
                inside = sum(weights[c, d] for d in range(6) if indicators[c, d, e])
                np.testing.assert_allclose(a[c, e], inside / weights[c].sum(), rtol=0, atol=1e-15)

    def test_uniform_weights_equal_the_unweighted_fraction(self):
        indicators = np.random.default_rng(5).random((4, 8, 3)) > 0.4
        a = event_probabilities(indicators)
        b = event_probabilities(indicators, weights=np.full((4, 8), 7.0))
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-15)

    def test_zero_weight_draws_carry_no_mass(self):
        indicators = np.array([[[True], [True], [False], [False]]])
        probs = event_probabilities(indicators, weights=np.array([[0.0, 1.0, 1.0, 0.0]]))
        self.assertEqual(probs[0, 0], 0.5)

    def test_probabilities_stay_in_unit_interval_without_clipping(self):
        rng = np.random.default_rng(6)
        indicators = rng.random((50, 10, 3)) > 0.5
        weights = rng.random((50, 10)) ** 8
        probs = event_probabilities(indicators, weights=weights)
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))
        everything = np.ones((2, 5, 1), dtype=bool)
        nothing = np.zeros((2, 5, 1), dtype=bool)
        w = rng.random((2, 5)) + 0.1
        np.testing.assert_array_equal(event_probabilities(everything, weights=w), [[1.0], [1.0]])
        np.testing.assert_array_equal(event_probabilities(nothing, weights=w), [[0.0], [0.0]])

    def test_event_and_complement_sum_to_one(self):
        rng = np.random.default_rng(7)
        event = rng.random((5, 9, 1)) > 0.3
        w = rng.random((5, 9)) + 0.05
        both = np.concatenate([event, ~event], axis=-1)
        for weights in (None, w):
            probs = event_probabilities(both, weights=weights)
            np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=0, atol=1e-14)

    def test_a_draw_exactly_at_the_threshold_counts_as_the_caller_wrote(self):
        draws = np.array([[[-1.0], [0.0], [0.0], [1.0]]])
        above = event_probabilities(draws > 0)
        at_or_above = event_probabilities(draws >= 0)
        below = event_probabilities(draws < 0)
        at_or_below = event_probabilities(draws <= 0)
        self.assertEqual(above[0, 0], 0.25)
        self.assertEqual(at_or_above[0, 0], 0.75)
        self.assertEqual(below[0, 0], 0.25)
        self.assertEqual(at_or_below[0, 0], 0.75)

    def test_exclusive_exhaustive_events_sum_to_one(self):
        draws = np.array([[[-2.0], [-1.0], [0.0], [0.0], [1.0], [3.0], [3.0], [4.0]]])
        events = np.concatenate([draws < 0, draws == 0, draws > 0], axis=-1)
        self.assertTrue(np.all(events.sum(axis=-1) == 1))
        probs = event_probabilities(events)
        np.testing.assert_array_equal(probs, [[0.25, 0.25, 0.5]])
        self.assertEqual(probs.sum(), 1.0)

    def test_only_booleans_are_events(self):
        base = np.ones((1, 3, 2), dtype=bool)
        for bad in (
            base.astype(int),
            base.astype(np.uint8),
            base.astype(float),
            base.astype(object),
            base.astype(complex),
        ):
            with self.subTest(dtype=bad.dtype), self.assertRaises(TypeError):
                event_probabilities(bad)
        event_probabilities(base)
        event_probabilities(np.array([[[True], [False]]], dtype=np.bool_))
        event_probabilities([[[True, False], [False, False]]])

    def test_shapes_are_exact(self):
        ok = np.ones((2, 3, 2), dtype=bool)
        for label, bad in (
            ("two-dimensional", ok[:, :, 0]),
            ("four-dimensional", ok[None]),
            ("no cases", ok[:0]),
            ("no draws", ok[:, :0]),
            ("no events", ok[:, :, :0]),
        ):
            with self.subTest(label=label), self.assertRaises(ValueError):
                event_probabilities(bad)
        w = np.ones((2, 3))
        for label, bad in (
            ("one-dimensional", w[0]),
            ("one case", w[:1]),
            ("one draw", w[:, :1]),
            ("extra draw", np.ones((2, 4))),
            ("extra case", np.ones((3, 3))),
            ("three-dimensional", w[:, :, None]),
        ):
            with self.subTest(label=label), self.assertRaises(ValueError):
                event_probabilities(ok, weights=bad)

    def test_weights_are_validated(self):
        ok = np.ones((2, 3, 1), dtype=bool)
        good = np.array([[1.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
        for label, bad in (
            ("negative", np.where(good == 2.0, -1.0, good)),
            ("nan", np.where(good == 2.0, np.nan, good)),
            ("inf", np.where(good == 2.0, np.inf, good)),
            ("zero total in a case", np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])),
            ("all zero", np.zeros((2, 3))),
        ):
            with self.subTest(label=label), self.assertRaises(ValueError):
                event_probabilities(ok, weights=bad)
        for label, bad in (("boolean", good > 0), ("complex", good.astype(complex))):
            with self.subTest(label=label), self.assertRaises((TypeError, ValueError)):
                event_probabilities(ok, weights=bad)

    def test_inputs_are_not_modified(self):
        indicators = np.random.default_rng(8).random((2, 5, 2)) > 0.5
        weights = np.random.default_rng(9).random((2, 5)) + 0.1
        before = _frozen(indicators, weights)
        event_probabilities(indicators, weights=weights)
        np.testing.assert_array_equal(indicators, before[0])
        np.testing.assert_array_equal(weights, before[1])


class EventBrierScoresTest(unittest.TestCase):
    def test_binary_is_the_single_probability_squared_error(self):
        probs = np.array([[0.25, 0.5], [1.0, 0.0]])
        outcomes = np.array([[True, False], [False, False]])
        scores = event_brier_scores(probs, outcomes, convention="binary")
        np.testing.assert_array_equal(scores, [[0.5625, 0.25], [1.0, 0.0]])
        self.assertEqual(scores.shape, (2, 2))
        self.assertEqual(scores.dtype, np.float64)

    def test_binary_scores_each_event_on_its_own(self):
        # Events need not be exclusive or exhaustive, and probabilities need not sum to one.
        probs = np.array([[0.5, 0.5, 0.5]])
        outcomes = np.array([[True, True, True]])
        scores = event_brier_scores(probs, outcomes, convention="binary")
        np.testing.assert_array_equal(scores, [[0.25, 0.25, 0.25]])

    def test_multiclass_sums_over_every_class(self):
        probs = np.array([[0.25, 0.75], [0.5, 0.5]])
        outcomes = np.array([[True, False], [False, True]])
        scores = event_brier_scores(probs, outcomes, convention="multiclass")
        self.assertEqual(scores.shape, (2,))
        np.testing.assert_array_equal(scores, [1.125, 0.5])

    def test_multiclass_is_twice_binary_for_two_classes(self):
        rng = np.random.default_rng(10)
        p = rng.random(12)
        probs = np.column_stack([p, 1.0 - p])
        outcome = rng.random(12) > 0.5
        outcomes = np.column_stack([outcome, ~outcome])
        multi = event_brier_scores(probs, outcomes, convention="multiclass")
        binary = event_brier_scores(probs[:, :1], outcomes[:, :1], convention="binary")
        np.testing.assert_allclose(multi, 2.0 * binary[:, 0], rtol=0, atol=1e-15)

    def test_multiclass_three_classes_against_a_loop(self):
        probs = np.array([[0.25, 0.25, 0.5], [0.125, 0.75, 0.125], [1.0, 0.0, 0.0]])
        outcomes = np.array([[False, False, True], [False, True, False], [False, False, True]])
        scores = event_brier_scores(probs, outcomes, convention="multiclass")
        expected = [
            sum((probs[c, k] - float(outcomes[c, k])) ** 2 for k in range(3)) for c in range(3)
        ]
        np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-15)
        self.assertEqual(scores[2], 2.0)

    def test_multiclass_requires_probability_rows_to_sum_to_one(self):
        outcomes = np.array([[True, False]])
        event_brier_scores(np.array([[0.5 + 5e-9, 0.5]]), outcomes, convention="multiclass")
        for bad in ([[0.5, 0.4]], [[0.5, 0.6]], [[0.5 + 1e-7, 0.5]], [[0.0, 0.0]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                event_brier_scores(np.array(bad), outcomes, convention="multiclass")
        # The same probabilities are fine under the binary convention.
        event_brier_scores(np.array([[0.5, 0.4]]), outcomes, convention="binary")

    def test_multiclass_requires_exactly_one_true_per_case(self):
        probs = np.array([[0.5, 0.5], [0.5, 0.5]])
        good = np.array([True, False])
        for label, bad in (
            ("none", np.array([False, False])),
            ("both", np.array([True, True])),
        ):
            for row in (0, 1):
                outcomes = np.vstack([good, good])
                outcomes[row] = bad
                with self.subTest(label=label, row=row), self.assertRaises(ValueError):
                    event_brier_scores(probs, outcomes, convention="multiclass")
        # Several true outcomes are fine for the binary convention.
        event_brier_scores(probs, np.array([[True, True], [False, False]]), convention="binary")

    def test_single_class_that_occurred_scores_zero(self):
        scores = event_brier_scores(np.array([[1.0]]), np.array([[True]]), convention="multiclass")
        np.testing.assert_array_equal(scores, [0.0])

    def test_convention_is_required_and_known(self):
        probs = np.array([[0.5, 0.5]])
        outcomes = np.array([[True, False]])
        with self.assertRaises(TypeError):
            event_brier_scores(probs, outcomes)
        with self.assertRaises(TypeError):
            event_brier_scores(probs, outcomes, "binary")
        for bad in (None, "", "Binary", "brier", "sum", 2, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                event_brier_scores(probs, outcomes, convention=bad)

    def test_probabilities_must_be_finite_numbers_in_the_unit_interval(self):
        outcomes = np.array([[True, False]])
        for convention in ("binary", "multiclass"):
            for bad in (
                [[-0.1, 1.1]],
                [[1.1, -0.1]],
                [[np.nan, 1.0]],
                [[np.inf, 0.0]],
                [[-np.inf, 0.0]],
            ):
                with self.subTest(convention=convention, bad=bad), self.assertRaises(ValueError):
                    event_brier_scores(np.array(bad), outcomes, convention=convention)
            for bad in (np.array([[True, False]]), np.array([[0.5, 0.5]], dtype=complex)):
                with (
                    self.subTest(convention=convention, dtype=bad.dtype),
                    self.assertRaises((TypeError, ValueError)),
                ):
                    event_brier_scores(bad, outcomes, convention=convention)
        # The boundary values themselves are valid, and integer probabilities are accepted.
        scores = event_brier_scores(np.array([[1, 0]]), outcomes, convention="binary")
        np.testing.assert_array_equal(scores, [[0.0, 0.0]])

    def test_outcomes_must_be_boolean_and_match_exactly(self):
        probs = np.array([[0.25, 0.75], [0.5, 0.5], [0.0, 1.0]])
        outcomes = np.array([[True, False], [False, True], [False, True]])
        for convention in ("binary", "multiclass"):
            for label, bad in (
                ("integer", outcomes.astype(int)),
                ("float", outcomes.astype(float)),
                ("object", outcomes.astype(object)),
            ):
                with self.subTest(convention=convention, label=label), self.assertRaises(TypeError):
                    event_brier_scores(probs, bad, convention=convention)
            for label, bad in (
                ("one-dimensional", outcomes[:, 0]),
                ("one column", outcomes[:, :1]),
                ("one case", outcomes[:1]),
                ("extra case", np.vstack([outcomes, outcomes[:1]])),
                ("extra event", np.hstack([outcomes, outcomes[:, :1]])),
                ("three-dimensional", outcomes[None]),
                ("no cases", outcomes[:0]),
            ):
                with (
                    self.subTest(convention=convention, label=label),
                    self.assertRaises(ValueError),
                ):
                    event_brier_scores(probs, bad, convention=convention)

    def test_probability_shapes_are_exact(self):
        outcomes = np.array([[True, False]])
        for convention in ("binary", "multiclass"):
            for label, bad in (
                ("one-dimensional", np.array([0.5, 0.5])),
                ("three-dimensional", np.array([[[0.5, 0.5]]])),
                ("no cases", np.empty((0, 2))),
                ("no events", np.empty((1, 0))),
            ):
                with (
                    self.subTest(convention=convention, label=label),
                    self.assertRaises(ValueError),
                ):
                    event_brier_scores(bad, outcomes, convention=convention)

    def test_inputs_are_not_modified(self):
        probs = np.array([[0.25, 0.75], [0.5, 0.5]])
        outcomes = np.array([[True, False], [False, True]])
        before = _frozen(probs, outcomes)
        for convention in ("binary", "multiclass"):
            event_brier_scores(probs, outcomes, convention=convention)
        np.testing.assert_array_equal(probs, before[0])
        np.testing.assert_array_equal(outcomes, before[1])

    def test_events_to_multiclass_brier_chain(self):
        rng = np.random.default_rng(11)
        draws = rng.normal(size=(6, 16, 1))
        events = np.concatenate([draws < -0.5, (draws >= -0.5) & (draws < 0.5), draws >= 0.5], -1)
        self.assertTrue(np.all(events.sum(axis=-1) == 1))
        probs = event_probabilities(events)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=0, atol=1e-14)
        truth = rng.normal(size=(6, 1))
        outcomes = np.concatenate([truth < -0.5, (truth >= -0.5) & (truth < 0.5), truth >= 0.5], -1)
        scores = event_brier_scores(probs, outcomes, convention="multiclass")
        expected = [
            sum((probs[c, k] - float(outcomes[c, k])) ** 2 for k in range(3)) for c in range(6)
        ]
        np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-15)


class DecisionRisksTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        # (cases=3, draws=4, actions=3); small integers keep every sum exact.
        self.losses = rng.integers(0, 8, size=(3, 4, 3)).astype(float)
        self.weights = np.array([[1.0, 2.0, 1.0, 4.0], [1.0, 1.0, 1.0, 1.0], [0.0, 4.0, 0.0, 4.0]])

    @staticmethod
    def _loops(losses, weights):
        cases, draws, actions = losses.shape
        expected = np.zeros((cases, actions))
        for c in range(cases):
            w = np.full(draws, 1.0) if weights is None else np.asarray(weights[c], dtype=float)
            w = w / w.sum()
            for a in range(actions):
                expected[c, a] = sum(w[d] * losses[c, d, a] for d in range(draws))
        bayes = [np.array([a for a in range(actions) if row[a] == row.min()]) for row in expected]
        return expected, expected.min(axis=1), bayes

    def test_table_against_independent_loops(self):
        for weights in (None, self.weights):
            with self.subTest(weighted=weights is not None):
                expected, minimum, bayes = self._loops(self.losses, weights)
                risks = decision_risks(self.losses, weights=weights)
                self.assertIsInstance(risks, DecisionRisks)
                np.testing.assert_array_equal(risks.expected_losses, expected)
                np.testing.assert_array_equal(risks.min_risk, minimum)
                self.assertEqual(risks.expected_losses.shape, (3, 3))
                self.assertEqual(risks.min_risk.shape, (3,))
                self.assertEqual(len(risks.bayes_actions), 3)
                for got, want in zip(risks.bayes_actions, bayes, strict=True):
                    np.testing.assert_array_equal(got, want)
                    self.assertEqual(got.dtype, np.int64)

    def test_random_float_losses_match_loops_closely(self):
        rng = np.random.default_rng(13)
        losses = rng.normal(size=(5, 30, 4)) * 10.0
        weights = rng.random((5, 30)) + 0.01
        expected, minimum, bayes = self._loops(losses, weights)
        risks = decision_risks(losses, weights=weights)
        np.testing.assert_allclose(risks.expected_losses, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(risks.min_risk, minimum, rtol=1e-12, atol=1e-12)
        for got, want in zip(risks.bayes_actions, bayes, strict=True):
            np.testing.assert_array_equal(got, want)

    def test_min_risk_is_the_smallest_expected_loss(self):
        risks = decision_risks(self.losses, weights=self.weights)
        np.testing.assert_array_equal(risks.min_risk, risks.expected_losses.min(axis=1))

    def test_hand_computed_two_action_example(self):
        # act: loss 1 in the first two of four draws, wait: loss 0.5 always.
        losses = np.array([[[1.0, 0.5], [1.0, 0.5], [0.0, 0.5], [0.0, 0.5]]])
        risks = decision_risks(losses)
        np.testing.assert_array_equal(risks.expected_losses, [[0.5, 0.5]])
        np.testing.assert_array_equal(risks.bayes_actions[0], [0, 1])
        weighted = decision_risks(losses, weights=np.array([[1.0, 1.0, 1.0, 5.0]]))
        np.testing.assert_array_equal(weighted.expected_losses, [[0.25, 0.5]])
        np.testing.assert_array_equal(weighted.bayes_actions[0], [0])

    def test_exact_ties_keep_every_tied_action(self):
        losses = np.zeros((4, 4, 3))
        # case 0: all three actions tie.
        losses[0] = 2.0
        # case 1: actions 0 and 2 tie (different columns, same mean), action 1 is worse.
        losses[1, :, 0] = [0.0, 2.0, 2.0, 0.0]
        losses[1, :, 1] = 3.0
        losses[1, :, 2] = 1.0
        # case 2: one clear minimizer.
        losses[2, :, 0] = 5.0
        losses[2, :, 1] = 1.0
        losses[2, :, 2] = 4.0
        # case 3: the tied pair holds the largest index.
        losses[3, :, 0] = 7.0
        losses[3, :, 1] = 0.0
        losses[3, :, 2] = 0.0
        risks = decision_risks(losses)
        wanted = ([0, 1, 2], [0, 2], [1], [1, 2])
        for got, want in zip(risks.bayes_actions, wanted, strict=True):
            np.testing.assert_array_equal(got, want)
            self.assertEqual(got.dtype, np.int64)
        np.testing.assert_array_equal(risks.min_risk, [2.0, 1.0, 1.0, 0.0])

    def test_all_action_tie_under_weights(self):
        losses = np.full((2, 4, 5), 3.0)
        risks = decision_risks(
            losses, weights=np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 0.0, 0.0, 1.0]])
        )
        for got in risks.bayes_actions:
            np.testing.assert_array_equal(got, np.arange(5))
        np.testing.assert_allclose(risks.min_risk, 3.0, rtol=0, atol=1e-15)

    def test_single_action_is_always_the_bayes_action(self):
        risks = decision_risks(self.losses[:, :, :1])
        for got in risks.bayes_actions:
            np.testing.assert_array_equal(got, [0])

    def test_a_tie_is_not_broken_toward_the_first_action(self):
        losses = np.array([[[1.0, 0.0, 0.0, 1.0]]])
        np.testing.assert_array_equal(decision_risks(losses).bayes_actions[0], [1, 2])

    def test_near_ties_are_not_ties(self):
        # One draw, so the expected loss is the loss itself.
        base = 0.1 + 0.2  # 0.30000000000000004
        losses = np.array([[[base, 0.3]]])
        risks = decision_risks(losses)
        self.assertNotEqual(risks.expected_losses[0, 0], risks.expected_losses[0, 1])
        np.testing.assert_array_equal(risks.bayes_actions[0], [1])
        nudged = np.array([[[1.0, np.nextafter(1.0, 2.0)]]])
        np.testing.assert_array_equal(decision_risks(nudged).bayes_actions[0], [0])
        np.testing.assert_array_equal(decision_risks(nudged[:, :, ::-1]).bayes_actions[0], [1])
        far = np.array([[[1.0, 1.0 + 1e-12]]])
        np.testing.assert_array_equal(decision_risks(far).bayes_actions[0], [0])

    def test_zero_weight_draws_carry_no_mass(self):
        losses = self.losses.copy()
        moved = losses.copy()
        moved[2, 0] = 1e9
        moved[2, 2] = -1e9
        a = decision_risks(losses, weights=self.weights)
        b = decision_risks(moved, weights=self.weights)
        np.testing.assert_allclose(b.expected_losses[2], a.expected_losses[2], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(b.bayes_actions[2], a.bayes_actions[2])

    def test_weights_are_renormalized_per_case(self):
        a = decision_risks(self.losses, weights=self.weights)
        b = decision_risks(self.losses, weights=self.weights * np.array([[8.0], [0.5], [3.0]]))
        np.testing.assert_allclose(a.expected_losses, b.expected_losses, rtol=0, atol=1e-14)

    def test_uniform_weights_equal_no_weights(self):
        a = decision_risks(self.losses)
        b = decision_risks(self.losses, weights=np.full((3, 4), 5.0))
        np.testing.assert_allclose(a.expected_losses, b.expected_losses, rtol=0, atol=1e-14)

    def test_cases_do_not_influence_one_another(self):
        full = decision_risks(self.losses, weights=self.weights)
        for c in range(3):
            alone = decision_risks(self.losses[c : c + 1], weights=self.weights[c : c + 1])
            np.testing.assert_array_equal(alone.expected_losses[0], full.expected_losses[c])
            np.testing.assert_array_equal(alone.bayes_actions[0], full.bayes_actions[c])

    def test_nonfinite_losses_raise_even_at_zero_weight(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(bad=bad, weights="none"):
                losses = self.losses.copy()
                losses[0, 0, 0] = bad
                with self.assertRaises(ValueError):
                    decision_risks(losses)
            with self.subTest(bad=bad, weights="zero-weight draw"):
                losses = self.losses.copy()
                losses[2, 0, 1] = bad  # draw 0 of case 2 has weight 0
                with self.assertRaises(ValueError):
                    decision_risks(losses, weights=self.weights)

    def test_weights_are_validated(self):
        zero_total_last = self.weights.copy()
        zero_total_last[2] = 0.0
        negative = self.weights.copy()
        negative[1, 1] = -1.0
        nan_at_zero_weight = self.weights.copy()
        nan_at_zero_weight[2, 0] = np.nan
        bad_values = {
            "one-dimensional": self.weights[0],
            "single row": self.weights[:1],
            "single column": self.weights[:, :1],
            "extra draw": np.hstack([self.weights, self.weights[:, :1]]),
            "extra case": np.vstack([self.weights, self.weights[:1]]),
            "three-dimensional": self.weights[:, :, None],
            "negative": negative,
            "zero total in the last case": zero_total_last,
            "all zero": np.zeros_like(self.weights),
            "nan": np.where(self.weights == 4.0, np.nan, self.weights),
            "inf": np.where(self.weights == 4.0, np.inf, self.weights),
            "nan at a zero-weight draw": nan_at_zero_weight,
            "boolean": self.weights > 0,
        }
        for label, weights in bad_values.items():
            with self.subTest(label=label), self.assertRaises((ValueError, TypeError)):
                decision_risks(self.losses, weights=weights)

    def test_loss_shapes_are_exact(self):
        for label, bad in (
            ("two-dimensional", self.losses[0]),
            ("one-dimensional", self.losses[0, 0]),
            ("four-dimensional", self.losses[None]),
            ("no cases", self.losses[:0]),
            ("no draws", self.losses[:, :0]),
            ("no actions", self.losses[:, :, :0]),
        ):
            with self.subTest(label=label), self.assertRaises(ValueError):
                decision_risks(bad)

    def test_non_numeric_losses_are_rejected(self):
        for bad in (
            self.losses > 3,
            self.losses.astype(complex),
            self.losses.astype(object),
            self.losses.astype(str),
        ):
            with self.subTest(dtype=bad.dtype), self.assertRaises((TypeError, ValueError)):
                decision_risks(bad)

    def test_integer_losses_are_accepted_and_reported_in_float64(self):
        risks = decision_risks(self.losses.astype(int))
        self.assertEqual(risks.expected_losses.dtype, np.float64)
        self.assertEqual(risks.min_risk.dtype, np.float64)
        expected, minimum, _ = self._loops(self.losses, None)
        np.testing.assert_array_equal(risks.expected_losses, expected)
        np.testing.assert_array_equal(risks.min_risk, minimum)

    def test_inputs_are_preserved_and_outputs_do_not_alias_them(self):
        before = _frozen(self.losses, self.weights)
        risks = decision_risks(self.losses, weights=self.weights)
        np.testing.assert_array_equal(self.losses, before[0])
        np.testing.assert_array_equal(self.weights, before[1])
        risks.expected_losses[...] = -1.0
        np.testing.assert_array_equal(self.losses, before[0])
        again = decision_risks(self.losses, weights=self.weights)
        self.assertTrue(np.all(again.expected_losses >= 0.0))

    def test_large_finite_losses_stay_finite(self):
        losses = np.full((1, 3, 2), 1.0e300)
        losses[0, :, 1] = 2.0e300
        risks = decision_risks(losses)
        self.assertTrue(np.all(np.isfinite(risks.expected_losses)))
        np.testing.assert_array_equal(risks.bayes_actions[0], [0])


class TruthNeverChangesADecisionTest(unittest.TestCase):
    """Retrospective scores take the truth; decision risks cannot."""

    def setUp(self):
        rng = np.random.default_rng(14)
        self.samples = rng.normal(size=(5, 40, 1))
        # The caller's decision table is built from the draws alone.
        act = np.where(self.samples[:, :, 0] < 0.0, 1.0, 0.0)
        wait = np.full(self.samples.shape[:2], 0.25)
        self.losses = np.stack([act, wait], axis=-1)

    def test_changing_the_truth_changes_scores_but_not_decisions(self):
        reference = decision_risks(self.losses)
        events = self.samples < 0.0
        probs = event_probabilities(events)
        truths = (np.full((5, 1), -4.0), np.full((5, 1), 4.0), np.linspace(-1, 1, 5)[:, None])
        brier = []
        crps = []
        for truth in truths:
            brier.append(event_brier_scores(probs, truth < 0.0, convention="binary"))
            crps.append(score_samples(self.samples, truth, levels=0.5).crps)
            risks = decision_risks(self.losses)
            np.testing.assert_array_equal(risks.expected_losses, reference.expected_losses)
            np.testing.assert_array_equal(risks.min_risk, reference.min_risk)
            for got, want in zip(risks.bayes_actions, reference.bayes_actions, strict=True):
                np.testing.assert_array_equal(got, want)
        # The retrospective scores really did react to the truth.
        self.assertFalse(np.array_equal(brier[0], brier[1]))
        self.assertFalse(np.allclose(crps[0], crps[1]))

    def test_scoring_does_not_alter_the_arrays_the_decision_uses(self):
        before = _frozen(self.samples, self.losses)
        probs = event_probabilities(self.samples < 0.0)
        event_brier_scores(probs, np.ones((5, 1), dtype=bool), convention="binary")
        score_samples(self.samples, np.zeros((5, 1)), levels=[0.5, 0.9], joint=True)
        np.testing.assert_array_equal(self.samples, before[0])
        np.testing.assert_array_equal(self.losses, before[1])

    def test_a_deterministic_loss_table_gives_the_same_action_set_on_every_call(self):
        first = decision_risks(self.losses)
        for _ in range(3):
            again = decision_risks(self.losses)
            np.testing.assert_array_equal(again.expected_losses, first.expected_losses)
            for a, b in zip(again.bayes_actions, first.bayes_actions, strict=True):
                np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
