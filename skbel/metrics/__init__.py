#  Copyright (c) 2026. Robin Thibaut, Ghent University

"""General-purpose posterior-evaluation, calibration and finite-action decision-risk metrics.

NumPy-only. No TensorFlow/scikit-learn dependency.
"""

from .calibration import (
    CoverageSummary,
    IntervalCoverage,
    case_rng,
    empirical_pit,
    interval_coverage,
    sbc_rank_histogram,
    sbc_ranks,
    summarize_coverage,
)
from .joint import energy_score
from .posterior import (
    bayes_action_set,
    brier_score,
    expected_action_losses,
    marginal_crps,
)
from .ranking import ProspectiveRanking, rank_prospective_measurements
from .sequential import (
    AcquisitionBranch,
    AcquisitionNode,
    CandidateEvidence,
    finite_acquisition_policy,
)

__all__ = [
    "AcquisitionBranch",
    "AcquisitionNode",
    "CandidateEvidence",
    "CoverageSummary",
    "IntervalCoverage",
    "ProspectiveRanking",
    "bayes_action_set",
    "brier_score",
    "case_rng",
    "empirical_pit",
    "energy_score",
    "expected_action_losses",
    "finite_acquisition_policy",
    "interval_coverage",
    "marginal_crps",
    "rank_prospective_measurements",
    "sbc_rank_histogram",
    "sbc_ranks",
    "summarize_coverage",
]
