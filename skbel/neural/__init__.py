#  Copyright (c) 2026. Robin Thibaut, Ghent University

"""Optional neural conditional posterior: mixture-density ensemble and quantile recalibration.

:mod:`skbel.neural.posterior` fits an equal-weight ensemble of mixture-density
networks on arrays and returns joint posterior draws ``(cases, draws,
targets)`` that plug into :mod:`skbel.evaluation`, :mod:`skbel.metrics` and
:mod:`skbel.design`. Training and network evaluation need PyTorch, installed
with ``pip install 'skbel[neural]'``; importing this package, the NumPy
samplers of precomputed mixtures and the recalibrator do not.
"""

from .posterior import (
    CHUNK,
    COVARIANCES,
    ENSEMBLE_FORMAT,
    FIT_SEEDS,
    MODEL_FORMAT,
    RECALIBRATOR_FORMAT,
    EqualWeightEnsemble,
    MixturePosterior,
    MixtureSettings,
    NeuralPosterior,
    QuantileRecalibrator,
    empirical_quantile,
    member_seeds,
    mid_cdf_knots,
    midrank_probabilities,
    pit_from_counts,
    pit_grid,
    pit_grid_indices,
    pit_uniforms,
    randomized_pit,
    sample_ensemble_latent,
    sample_mixture_latent,
)

__all__ = [
    "CHUNK",
    "COVARIANCES",
    "ENSEMBLE_FORMAT",
    "FIT_SEEDS",
    "MODEL_FORMAT",
    "RECALIBRATOR_FORMAT",
    "EqualWeightEnsemble",
    "MixturePosterior",
    "MixtureSettings",
    "NeuralPosterior",
    "QuantileRecalibrator",
    "empirical_quantile",
    "member_seeds",
    "mid_cdf_knots",
    "midrank_probabilities",
    "pit_from_counts",
    "pit_grid",
    "pit_grid_indices",
    "pit_uniforms",
    "randomized_pit",
    "sample_ensemble_latent",
    "sample_mixture_latent",
]
