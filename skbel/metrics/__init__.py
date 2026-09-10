#  Copyright (c) 2026. Robin Thibaut, Ghent University

"""General-purpose posterior-evaluation and finite-action decision-risk metrics.

NumPy-only. No TensorFlow/scikit-learn dependency.
"""

from .posterior import (
    bayes_action_set,
    brier_score,
    expected_action_losses,
    marginal_crps,
)

__all__ = [
    "bayes_action_set",
    "brier_score",
    "expected_action_losses",
    "marginal_crps",
]
