#  Copyright (c) 2026. Robin Thibaut, Ghent University

"""General-purpose posterior-evaluation and finite-action decision-risk metrics.

NumPy-only. No TensorFlow/scikit-learn dependency.
"""

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
    "ProspectiveRanking",
    "bayes_action_set",
    "brier_score",
    "energy_score",
    "expected_action_losses",
    "finite_acquisition_policy",
    "marginal_crps",
    "rank_prospective_measurements",
]
