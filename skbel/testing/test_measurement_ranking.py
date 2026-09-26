"""Unit tests for skbel.metrics.ranking.

Import strategy: normal tests require the complete package import. An import
failure intentionally fails test collection rather than silently bypassing
``skbel.__init__`` with a direct file import.
"""

from __future__ import annotations

import unittest

import numpy as np

from skbel.metrics import ProspectiveRanking, rank_prospective_measurements


class TestRankingRiskCriterion(unittest.TestCase):
    def test_strict_order_lowest_risk_first(self):
        result = rank_prospective_measurements(["a", "b", "c"], [3.0, 1.0, 2.0])
        self.assertEqual(result.order, ("b", "c", "a"))
        self.assertEqual(result.best, ("b",))
        self.assertEqual(result.ranks, {"a": 3, "b": 1, "c": 2})

    def test_default_criterion_is_risk(self):
        explicit = rank_prospective_measurements(["a", "b"], [1.0, 2.0], criterion="risk")
        implicit = rank_prospective_measurements(["a", "b"], [1.0, 2.0])
        self.assertEqual(explicit, implicit)


class TestRankingUtilityCriterion(unittest.TestCase):
    def test_strict_order_highest_utility_first(self):
        result = rank_prospective_measurements(
            ["a", "b", "c"], [3.0, 1.0, 2.0], criterion="utility"
        )
        self.assertEqual(result.order, ("a", "c", "b"))
        self.assertEqual(result.best, ("a",))
        self.assertEqual(result.ranks, {"a": 1, "b": 3, "c": 2})


class TestRankingTies(unittest.TestCase):
    def test_exact_tie_for_best_reported_in_full(self):
        result = rank_prospective_measurements(["a", "b", "c"], [0.5, 0.5, 0.9])
        self.assertEqual(result.best, ("a", "b"))
        self.assertEqual(result.ranks, {"a": 1, "b": 1, "c": 3})

    def test_ties_preserve_original_candidate_order_deterministically(self):
        # b and a tie for best; input order is [b, a, c] so b precedes a.
        result_1 = rank_prospective_measurements(["b", "a", "c"], [0.5, 0.5, 0.9])
        self.assertEqual(result_1.order, ("b", "a", "c"))
        # Swapping input order swaps the deterministic tie-break too.
        result_2 = rank_prospective_measurements(["a", "b", "c"], [0.5, 0.5, 0.9])
        self.assertEqual(result_2.order, ("a", "b", "c"))

    def test_all_tied_gives_standard_competition_ranking(self):
        result = rank_prospective_measurements(["a", "b", "c"], [1.0, 1.0, 1.0])
        self.assertEqual(result.best, ("a", "b", "c"))
        self.assertEqual(result.ranks, {"a": 1, "b": 1, "c": 1})

    def test_middle_tie_skips_rank_correctly(self):
        # Sorted risks: 1 (rank1), 2,2 (rank2 each), 4 (rank4, not rank3).
        result = rank_prospective_measurements(["a", "b", "c", "d"], [1.0, 2.0, 2.0, 4.0])
        self.assertEqual(result.ranks, {"a": 1, "b": 2, "c": 2, "d": 4})

    def test_repeated_ranking_call_is_deterministic(self):
        candidates = ["x", "y", "z", "w"]
        scores = [0.3, 0.1, 0.3, 0.2]
        first = rank_prospective_measurements(candidates, scores)
        for _ in range(5):
            self.assertEqual(rank_prospective_measurements(candidates, scores), first)


class TestRankingIntegerPrecision(unittest.TestCase):
    def test_large_adjacent_int64_scores_do_not_collapse_via_float64(self):
        # 2**53 and 2**53 + 1 are adjacent int64 values that collapse into a
        # false tie if cast through float64 (53-bit mantissa).
        scores = np.array([2**53 + 1, 2**53], dtype=np.int64)
        result = rank_prospective_measurements(["higher", "lower"], scores)
        self.assertEqual(result.best, ("lower",))
        self.assertEqual(result.order, ("lower", "higher"))

    def test_large_int64_scores_still_report_exact_ties(self):
        scores = np.array([2**53, 2**53], dtype=np.int64)
        result = rank_prospective_measurements(["a", "b"], scores)
        self.assertEqual(result.best, ("a", "b"))
        self.assertEqual(result.ranks, {"a": 1, "b": 1})

    def test_signed_int_minimum_utility_ordering_does_not_overflow(self):
        # Negating the minimum representable int64 value overflows/wraps in
        # fixed-width arithmetic; utility ordering must not rely on that.
        int64_min = -(2**63)
        scores = np.array([int64_min, int64_min + 1], dtype=np.int64)
        result = rank_prospective_measurements(["min", "next"], scores, criterion="utility")
        self.assertEqual(result.best, ("next",))
        self.assertEqual(result.order, ("next", "min"))

    def test_signed_int_minimum_risk_ordering_does_not_overflow(self):
        int64_min = -(2**63)
        scores = np.array([int64_min, int64_min + 1], dtype=np.int64)
        result = rank_prospective_measurements(["min", "next"], scores, criterion="risk")
        self.assertEqual(result.best, ("min",))
        self.assertEqual(result.order, ("min", "next"))


class TestRankingSingleCandidate(unittest.TestCase):
    def test_single_candidate(self):
        result = rank_prospective_measurements(["only"], [7.0])
        self.assertEqual(result.order, ("only",))
        self.assertEqual(result.best, ("only",))
        self.assertEqual(result.ranks, {"only": 1})


class TestRankingCandidateTypes(unittest.TestCase):
    def test_non_string_hashable_candidates(self):
        candidates = [(0, 0), (1, 1), (2, 2)]
        result = rank_prospective_measurements(candidates, [2.0, 1.0, 3.0])
        self.assertEqual(result.order, ((1, 1), (0, 0), (2, 2)))


class TestRankingValidation(unittest.TestCase):
    def test_empty_candidates_rejected(self):
        with self.assertRaises(ValueError):
            rank_prospective_measurements([], [])

    def test_duplicate_candidates_rejected(self):
        with self.assertRaises(ValueError):
            rank_prospective_measurements(["a", "a"], [1.0, 2.0])

    def test_unhashable_candidates_rejected(self):
        with self.assertRaises(TypeError):
            rank_prospective_measurements([["a"], ["b"]], [1.0, 2.0])

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            rank_prospective_measurements(["a", "b", "c"], [1.0, 2.0])

    def test_scores_must_be_1d(self):
        with self.assertRaises(ValueError):
            rank_prospective_measurements(["a", "b"], [[1.0], [2.0]])

    def test_nonfinite_scores_rejected(self):
        with self.assertRaises(ValueError):
            rank_prospective_measurements(["a", "b"], [1.0, np.nan])
        with self.assertRaises(ValueError):
            rank_prospective_measurements(["a", "b"], [1.0, np.inf])

    def test_non_numeric_scores_rejected(self):
        with self.assertRaises(TypeError):
            rank_prospective_measurements(["a", "b"], ["low", "high"])

    def test_invalid_criterion_rejected(self):
        with self.assertRaises(ValueError):
            rank_prospective_measurements(["a", "b"], [1.0, 2.0], criterion="lowest")


class TestRankingResultType(unittest.TestCase):
    def test_result_is_prospective_ranking_namedtuple(self):
        result = rank_prospective_measurements(["a", "b"], [1.0, 2.0])
        self.assertIsInstance(result, ProspectiveRanking)
        order, best, ranks = result
        self.assertEqual(order, result.order)
        self.assertEqual(best, result.best)
        self.assertEqual(ranks, result.ranks)


if __name__ == "__main__":
    unittest.main()
