"""Unit tests for examples.prospective_risk_ranking.

These oracle values are derived independently by hand (see comments), not by
re-running the example's own helper functions on themselves, so a bug in the
example's averaging/minimization order, outcome weighting, normalization, or
tie handling would be caught here even if the example were internally
self-consistent but wrong.

Hand derivation for the informative candidates (prior = [0.5, 0.3, 0.2],
likelihood rows p(y0|s), p(y1|s) = [0.9, 0.1], [0.5, 0.5], [0.1, 0.9],
loss table [[10, 0], [4, 6], [1, 20]]):

    p(y0) = 0.5*0.9 + 0.3*0.5 + 0.2*0.1 = 0.62
    p(y1) = 0.5*0.1 + 0.3*0.5 + 0.2*0.9 = 0.38  (0.62 + 0.38 == 1)

    posterior(y0) = [0.45, 0.15, 0.02] / 0.62 = [45/62, 15/62, 2/62]
    posterior(y1) = [0.05, 0.15, 0.18] / 0.38 = [5/38, 15/38, 18/38]

    E[treat | y0]   = (45*10 + 15*4 + 2*1) / 62  = 512/62  = 8.258064516129032
    E[monitor | y0] = (45*0  + 15*6 + 2*20) / 62 = 130/62  = 2.096774193548387 <- min
    E[treat | y1]   = (5*10 + 15*4 + 18*1) / 38  = 128/38  = 3.368421052631579 <- min
    E[monitor | y1] = (5*0  + 15*6 + 18*20) / 38 = 450/38  = 11.842105263157894

    expected future risk = 0.62 * (130/62) + 0.38 * (128/38) = 1.3 + 1.28 = 2.58

    E[treat] under prior   = 0.5*10 + 0.3*4 + 0.2*1 = 6.4
    E[monitor] under prior = 0.5*0  + 0.3*6 + 0.2*20 = 5.8 <- current Bayes risk

    utility = 5.8 - 2.58 = 3.22

For ``no_measurement`` the single outcome has likelihood 1 for every state,
so its posterior is exactly the prior: expected future risk == current risk
== 5.8, utility == 0.0.
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

import numpy as np

from examples.prospective_risk_ranking import (
    CANDIDATE_LIKELIHOODS,
    LOSS_TABLE,
    PRIOR,
    compute_rankings,
    expected_future_bayes_risk,
    main,
)

CURRENT_RISK = 5.8
INFORMATIVE_EFR = 2.58
INFORMATIVE_UTILITY = 3.22
NO_MEASUREMENT_EFR = 5.8
NO_MEASUREMENT_UTILITY = 0.0


class TestExampleOracleValues(unittest.TestCase):
    def test_current_risk_and_no_measurement_candidate(self):
        current_risk, details, _, _ = compute_rankings()
        self.assertAlmostEqual(current_risk, CURRENT_RISK, places=12)
        self.assertAlmostEqual(
            details["no_measurement"]["expected_future_risk"], NO_MEASUREMENT_EFR, places=12
        )
        self.assertAlmostEqual(
            details["no_measurement"]["utility"], NO_MEASUREMENT_UTILITY, places=12
        )

    def test_informative_candidate_expected_future_risk_and_utility(self):
        _, details, _, _ = compute_rankings()
        for name in ("informative_probe", "informative_probe_duplicate"):
            self.assertAlmostEqual(
                details[name]["expected_future_risk"], INFORMATIVE_EFR, places=12
            )
            self.assertAlmostEqual(details[name]["utility"], INFORMATIVE_UTILITY, places=12)

    def test_predictive_outcome_probabilities_are_nonuniform_and_normalized(self):
        _, details, _, _ = compute_rankings()
        predictive = details["informative_probe"]["predictive_outcome_probabilities"]
        np.testing.assert_allclose(predictive, [0.62, 0.38], atol=1e-12)
        self.assertNotAlmostEqual(predictive[0], predictive[1], places=6)
        self.assertAlmostEqual(float(np.sum(predictive)), 1.0, places=12)

    def test_per_outcome_min_risk_matches_hand_derivation(self):
        _, details, _, _ = compute_rankings()
        per_outcome = details["informative_probe"]["per_outcome_min_risk"]
        np.testing.assert_allclose(per_outcome, [130.0 / 62.0, 128.0 / 38.0], atol=1e-12)


class TestOrderMustBeAverageAfterMinimize(unittest.TestCase):
    """Regression tests for requirement: minimize per outcome, THEN average."""

    def test_true_expected_future_risk_differs_from_committing_to_one_action_upfront(self):
        # Minimizing BEFORE conditioning on the outcome (i.e. picking one
        # action in advance and only then averaging its loss over outcomes)
        # collapses back to the prior's Bayes risk, by the law of total
        # expectation -- it destroys the value of the measurement. The
        # correct expected future risk must differ from this wrong quantity.
        efr, predictive, per_outcome_min_risk = expected_future_bayes_risk(
            LOSS_TABLE, PRIOR, CANDIDATE_LIKELIHOODS["informative_probe"]
        )
        posterior_y0 = PRIOR * CANDIDATE_LIKELIHOODS["informative_probe"][:, 0] / predictive[0]
        posterior_y1 = PRIOR * CANDIDATE_LIKELIHOODS["informative_probe"][:, 1] / predictive[1]
        action_losses_y0 = posterior_y0 @ LOSS_TABLE
        action_losses_y1 = posterior_y1 @ LOSS_TABLE
        wrong_average_then_minimize = float(
            np.min(predictive[0] * action_losses_y0 + predictive[1] * action_losses_y1)
        )
        self.assertAlmostEqual(wrong_average_then_minimize, CURRENT_RISK, places=12)
        self.assertNotAlmostEqual(efr, wrong_average_then_minimize, places=6)
        self.assertAlmostEqual(efr, INFORMATIVE_EFR, places=12)

    def test_true_expected_future_risk_differs_from_uniform_outcome_weights(self):
        # Using uniform weights over outcomes instead of the true, nonuniform
        # predictive probabilities [0.62, 0.38] must give a different (and
        # therefore wrong) expected future risk.
        efr, _, per_outcome_min_risk = expected_future_bayes_risk(
            LOSS_TABLE, PRIOR, CANDIDATE_LIKELIHOODS["informative_probe"]
        )
        uniform_wrong = float(np.mean(per_outcome_min_risk))
        self.assertAlmostEqual(uniform_wrong, 2.732597623089983, places=12)
        self.assertNotAlmostEqual(efr, uniform_wrong, places=6)

    def test_true_expected_future_risk_differs_from_single_favorable_outcome(self):
        # Conditioning on only the more favorable (lower-risk) outcome
        # instead of enumerating and weighting every outcome must give a
        # different (and therefore wrong, overly optimistic) value.
        efr, _, per_outcome_min_risk = expected_future_bayes_risk(
            LOSS_TABLE, PRIOR, CANDIDATE_LIKELIHOODS["informative_probe"]
        )
        favorable_only = float(np.min(per_outcome_min_risk))
        self.assertAlmostEqual(favorable_only, 130.0 / 62.0, places=12)
        self.assertNotAlmostEqual(efr, favorable_only, places=6)
        self.assertLess(favorable_only, efr)


class TestNormalization(unittest.TestCase):
    def test_prior_sums_to_one(self):
        self.assertAlmostEqual(float(np.sum(PRIOR)), 1.0, places=12)

    def test_every_likelihood_row_sums_to_one(self):
        for name, likelihood in CANDIDATE_LIKELIHOODS.items():
            row_sums = likelihood.sum(axis=1)
            np.testing.assert_allclose(
                row_sums, np.ones_like(row_sums), atol=1e-12, err_msg=f"candidate {name}"
            )

    def test_predictive_and_posteriors_are_normalized_for_every_candidate(self):
        for likelihood in CANDIDATE_LIKELIHOODS.values():
            _, predictive, _ = expected_future_bayes_risk(LOSS_TABLE, PRIOR, likelihood)
            self.assertAlmostEqual(float(np.sum(predictive)), 1.0, places=12)
            for y in range(likelihood.shape[1]):
                posterior = PRIOR * likelihood[:, y] / predictive[y]
                self.assertAlmostEqual(float(np.sum(posterior)), 1.0, places=12)


class TestExactTieHandlingInRankings(unittest.TestCase):
    def test_duplicate_informative_candidates_tie_exactly(self):
        _, details, risk_ranking, utility_ranking = compute_rankings()
        self.assertEqual(
            details["informative_probe"]["expected_future_risk"],
            details["informative_probe_duplicate"]["expected_future_risk"],
        )
        self.assertEqual(
            details["informative_probe"]["utility"],
            details["informative_probe_duplicate"]["utility"],
        )

        self.assertEqual(
            set(risk_ranking.best), {"informative_probe", "informative_probe_duplicate"}
        )
        self.assertEqual(
            set(utility_ranking.best), {"informative_probe", "informative_probe_duplicate"}
        )

        # Standard competition ranking: both tied candidates share rank 1,
        # and no_measurement's rank skips ahead to 3 (not 2).
        self.assertEqual(risk_ranking.ranks["informative_probe"], 1)
        self.assertEqual(risk_ranking.ranks["informative_probe_duplicate"], 1)
        self.assertEqual(risk_ranking.ranks["no_measurement"], 3)

        self.assertEqual(utility_ranking.ranks["informative_probe"], 1)
        self.assertEqual(utility_ranking.ranks["informative_probe_duplicate"], 1)
        self.assertEqual(utility_ranking.ranks["no_measurement"], 3)

        # Input order ("informative_probe" before "informative_probe_duplicate")
        # is preserved deterministically among exact ties.
        self.assertEqual(
            risk_ranking.order[:2], ("informative_probe", "informative_probe_duplicate")
        )
        self.assertEqual(
            utility_ranking.order[:2], ("informative_probe", "informative_probe_duplicate")
        )

    def test_no_measurement_is_strictly_worst_on_both_criteria(self):
        _, _, risk_ranking, utility_ranking = compute_rankings()
        self.assertEqual(risk_ranking.order[-1], "no_measurement")
        self.assertEqual(utility_ranking.order[-1], "no_measurement")
        self.assertNotIn("no_measurement", risk_ranking.best)
        self.assertNotIn("no_measurement", utility_ranking.best)


class TestExampleRuns(unittest.TestCase):
    def test_main_runs_and_prints_named_quantities(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            current_risk, details, risk_ranking, utility_ranking = main()
        output = buffer.getvalue()

        self.assertAlmostEqual(current_risk, CURRENT_RISK, places=12)
        self.assertIn("Current Bayes risk", output)
        for name in details:
            self.assertIn(name, output)
        self.assertIn("expected future risk", output)
        self.assertIn("utility", output)
        self.assertIn("Risk ranking", output)
        self.assertIn("Utility ranking", output)
        self.assertEqual(risk_ranking.order[-1], "no_measurement")
        self.assertEqual(utility_ranking.order[-1], "no_measurement")


if __name__ == "__main__":
    unittest.main()
