#  Copyright (c) 2026. Robin Thibaut, Ghent University

"""Functions that take a fitted BEL posterior to scores, event probabilities and decision risks.

The chain is::

    fitted BEL --sample_bel_posterior--> draws (cases, draws, targets)
        draws + truth            --score_samples-------> CRPS, interval coverage, energy score
        caller event indicators  --event_probabilities--> P(event) per case
        P(event) + outcomes      --event_brier_scores---> retrospective Brier scores
        caller loss table        --decision_risks-------> expected losses and Bayes action sets

Only :func:`sample_bel_posterior` touches a :class:`~skbel.BEL`; the other
functions accept arrays from any sampler. Nothing here fits, refits or
simulates, and nothing turns targets into events or losses: event indicators
and loss tables are computed by the caller from the draws, in whatever units
the caller chooses. Decision risks never receive the truth, so a retrospective
score cannot change a decision.

Conventions, shared with :mod:`skbel.metrics`: shapes are checked exactly with
no broadcasting, inputs must be finite, nothing is clipped, and NaN is never
hidden. Malformed values raise even where they carry zero weight.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .learning.bel import BEL
from .metrics.calibration import IntervalCoverage, interval_coverage
from .metrics.joint import energy_score
from .metrics.posterior import (
    _as_float_array,
    _check_finite,
    _validate_weights_2d,
    bayes_action_set,
    brier_score,
    expected_action_losses,
    marginal_crps,
)

__all__ = [
    "DecisionRisks",
    "SampleScores",
    "decision_risks",
    "event_brier_scores",
    "event_probabilities",
    "sample_bel_posterior",
    "score_samples",
]

_MODES = ("mvn", "kde", "tm")


def _check_count(value, name: str) -> int:
    """Return ``value`` as a Python int; it must be a positive integer (not a bool)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _fitted_feature_count(pipeline) -> int | None:
    """Number of input columns a fitted pre-processing step reports, if it reports one."""
    try:
        count = pipeline.n_features_in_
    except AttributeError:
        return None
    return int(count)


# ---------------------------------------------------------------------------
# Sampling a fitted BEL
# ---------------------------------------------------------------------------


def sample_bel_posterior(
    bel: BEL,
    observations,
    n_draws: int,
    *,
    mode: str | None = None,
    noise: float | None = None,
) -> np.ndarray:
    """Posterior draws of a fitted BEL in original target units.

    Calls the public :meth:`BEL.predict` once, with ``return_samples=True``,
    ``inverse_transform=True`` and ``dtype="float64"``, and checks its
    output. The model is not fitted or refitted here.

    ``predict`` keeps its usual side effects on ``bel``: it stores ``mode``
    (when given), ``noise`` (``None`` resets it to the default multiplier
    0.01), ``n_posts = n_draws``, the projected observations ``X_obs_f`` and
    the posterior state of the mode (``posterior_mean`` and
    ``posterior_covariance``, ``kde_functions``, or ``tm_functions``, the
    last re-optimized on every call). If ``bel.seed`` is ``None``, sampling
    takes a fresh seed from operating-system entropy and stores it in
    ``bel.seed``; read it afterwards to repeat the draws.

    Draws for row ``i`` come from the random stream of row position ``i``
    under ``bel.seed``, so they depend on where a case sits in
    ``observations``. Reordering or subsetting the rows changes which stream
    a case receives; repeating the same call with the same seed repeats the
    draws. No other seeding is done and NumPy's global random state is not
    used.

    :param bel: a fitted :class:`~skbel.BEL`. A model with an active
        ``x_observation`` override is rejected, because ``predict`` would
        then ignore ``observations``.
    :param observations: real, finite predictor rows in original units,
        shape ``(cases, features)`` exactly. When the fitted predictor
        pre-processing reports ``n_features_in_``, ``features`` must equal
        it.
    :param n_draws: positive integer number of draws per case.
    :param mode: ``"mvn"``, ``"kde"``, ``"tm"`` or ``None`` to keep
        ``bel.mode``.
    :param noise: passed to ``predict`` unchanged; see :meth:`BEL.predict`.
    :return: float64 array ``(cases, n_draws, targets)``. When the fitted
        target pre-processing reports ``n_features_in_``, ``targets`` equals
        it. Real floating output of another precision from ``predict`` is
        converted to float64 before it is checked; lower precisions widen
        exactly.
    :raises ValueError: for malformed inputs, an unfitted model, an
        ``x_observation`` override, non-float output, or output of the wrong
        shape or with non-finite values, including values that overflow
        float64.
    """
    if not isinstance(bel, BEL):
        raise TypeError(f"bel must be a BEL instance, got {type(bel).__name__}")
    if getattr(bel, "X_f", None) is None:
        raise ValueError("bel must be fitted before sampling")
    if bel.x_observation is not None:
        raise ValueError(
            "bel.x_observation is set, so predict would ignore the supplied observations; "
            "set bel.x_observation = None first"
        )
    effective_mode = bel.mode if mode is None else mode
    if effective_mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {effective_mode!r}")
    n_draws = _check_count(n_draws, "n_draws")

    obs = _as_float_array(observations, "observations")
    if obs.ndim != 2:
        raise ValueError(f"observations must have shape (cases, features), got ndim={obs.ndim}")
    if 0 in obs.shape:
        raise ValueError(f"observations must have no empty axes, got shape {obs.shape}")
    _check_finite(obs, "observations")
    cases = obs.shape[0]
    n_features = _fitted_feature_count(bel.X_pre_processing)
    if n_features is not None and obs.shape[1] != n_features:
        raise ValueError(
            f"observations have {obs.shape[1]} features; the fitted model expects {n_features}"
        )

    samples = bel.predict(
        obs.copy(),
        n_posts=n_draws,
        mode=mode,
        noise=noise,
        return_samples=True,
        inverse_transform=True,
        dtype="float64",
    )

    samples = np.asarray(samples)
    if samples.dtype.kind != "f":
        raise ValueError(f"predict returned a non-float array of dtype {samples.dtype}")
    # Lower-precision floats widen exactly; a wider float that overflows
    # float64 becomes infinite and is rejected by the finiteness check below.
    samples = samples.astype(np.float64, copy=False)
    if samples.ndim != 3 or samples.shape[:2] != (cases, n_draws) or samples.shape[2] == 0:
        raise ValueError(
            f"predict returned shape {samples.shape}; expected ({cases}, {n_draws}, targets)"
        )
    n_targets = _fitted_feature_count(bel.Y_pre_processing)
    if n_targets is not None and samples.shape[2] != n_targets:
        raise ValueError(
            f"predict returned {samples.shape[2]} targets; the fitted model has {n_targets}"
        )
    _check_finite(samples, "posterior samples")
    return samples


# ---------------------------------------------------------------------------
# Scores against the truth
# ---------------------------------------------------------------------------


class SampleScores(NamedTuple):
    """Per-case scores from :func:`score_samples`; nothing is aggregated.

    :ivar crps: marginal CRPS, shape ``(cases, targets)``, in target units.
    :ivar coverage: per-case central intervals, an
        :class:`~skbel.metrics.IntervalCoverage` with arrays of shape
        ``(cases, levels, targets)``.
    :ivar energy: energy score, shape ``(cases,)``, or ``None`` when not
        requested.
    """

    crps: np.ndarray
    coverage: IntervalCoverage
    energy: np.ndarray | None


def score_samples(
    samples,
    truth,
    *,
    levels,
    weights=None,
    crps_estimator: str = "empirical",
    joint: bool = False,
) -> SampleScores:
    """Per-case CRPS, interval coverage and, on request, the energy score.

    A thin wrapper around :func:`~skbel.metrics.marginal_crps`,
    :func:`~skbel.metrics.interval_coverage` and
    :func:`~skbel.metrics.energy_score`, each called with the same draws,
    truth and weights, and with their conventions unchanged. Results stay per
    case: no average over cases, no pooling over targets. Summarize coverage
    explicitly with :func:`~skbel.metrics.summarize_coverage`.

    :param samples: draws, shape ``(cases, draws, targets)``.
    :param truth: realized values, shape ``(cases, targets)``, in the units of
        ``samples``.
    :param levels: central interval levels, required; a level or a
        one-dimensional array, each strictly between 0 and 1.
    :param weights: ``None`` or per-draw weights, shape ``(cases, draws)``,
        renormalized per case by each metric.
    :param crps_estimator: ``"empirical"`` or ``"unbiased"``, as in
        :func:`~skbel.metrics.marginal_crps`; ``"unbiased"`` requires
        ``weights=None`` and at least two draws.
    :param joint: compute the energy score. It adds Euclidean distances over
        all targets, so targets in different units (or scales) are mixed; it
        is off by default and the caller chooses the target scaling.
    :return: a :class:`SampleScores`.
    """
    if not isinstance(joint, (bool, np.bool_)):
        raise TypeError(f"joint must be a bool, got {joint!r}")
    if levels is None:
        raise ValueError("levels must be given explicitly")
    crps = marginal_crps(samples, truth, weights=weights, estimator=crps_estimator)
    coverage = interval_coverage(samples, truth, levels, weights=weights)
    energy = energy_score(samples, truth, weights=weights) if joint else None
    return SampleScores(crps=crps, coverage=coverage, energy=energy)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def _check_bool(arr, name: str, ndim: int, axes: str) -> np.ndarray:
    out = np.asarray(arr)
    if out.dtype.kind != "b":
        raise TypeError(f"{name} must be a boolean array, got dtype {out.dtype}")
    if out.ndim != ndim:
        raise ValueError(f"{name} must have shape {axes}, got ndim={out.ndim}")
    if 0 in out.shape:
        raise ValueError(f"{name} must have no empty axes, got shape {out.shape}")
    return out


def event_probabilities(indicators, *, weights=None) -> np.ndarray:
    """Probability of each caller-defined event under the (weighted) draws.

    The caller decides what an event is and evaluates it on every draw, for
    example ``samples[:, :, 0] > threshold``; how a draw exactly at a
    threshold counts is part of that choice. No target is turned into an
    event here.

    :param indicators: boolean array ``(cases, draws, events)``: whether
        each draw belongs to each event.
    :param weights: ``None`` (each draw counts ``1 / draws``) or per-draw
        weights, shape ``(cases, draws)`` exactly: finite, non-negative, with
        a positive total per case, renormalized per case as in
        :func:`~skbel.metrics.marginal_crps`.
    :return: float64 array ``(cases, events)`` in ``[0, 1]``. Unweighted, it is
        the fraction of draws in the event. Weighted, it is the event weight
        divided by the event weight plus the non-event weight, which stays in
        ``[0, 1]`` without clipping.
    """
    ind = _check_bool(indicators, "indicators", 3, "(cases, draws, events)")
    cases, draws, _ = ind.shape
    if weights is None:
        probs = np.count_nonzero(ind, axis=1) / draws
    else:
        w = _validate_weights_2d(weights, cases, draws)[:, :, None]
        inside = np.sum(np.where(ind, w, 0.0), axis=1)
        outside = np.sum(np.where(ind, 0.0, w), axis=1)
        probs = inside / (inside + outside)
    probs = np.asarray(probs, dtype=np.float64)
    _check_finite(probs, "event probabilities")
    return probs


def event_brier_scores(probabilities, outcomes, *, convention: str) -> np.ndarray:
    """Retrospective Brier scores of event probabilities against observed outcomes.

    This scores probabilities after the fact. It is not a decision loss and
    nothing here feeds back into :func:`decision_risks`.

    :param probabilities: float array ``(cases, events)``, finite, in
        ``[0, 1]``, for example from :func:`event_probabilities`.
    :param outcomes: boolean array ``(cases, events)``: whether each event
        occurred.
    :param convention: required, no default.

        - ``"binary"``: each event is scored on its own with the
          single-probability convention ``(p - y) ** 2``. Returns
          ``(cases, events)``.
        - ``"multiclass"``: the events are mutually exclusive and exhaustive
          classes. Each probability row must sum to 1 (within ``1e-8``) and
          each outcome row must hold exactly one ``True``. Returns
          ``(cases,)`` from :func:`~skbel.metrics.brier_score`, which sums
          the squared error over every class; for two classes this is twice
          the binary value.
    :return: float64 array, see ``convention``.
    """
    if convention not in ("binary", "multiclass"):
        raise ValueError(f"convention must be 'binary' or 'multiclass', got {convention!r}")
    probs = _as_float_array(probabilities, "probabilities")
    if probs.ndim != 2:
        raise ValueError(f"probabilities must have shape (cases, events), got ndim={probs.ndim}")
    if 0 in probs.shape:
        raise ValueError(f"probabilities must have no empty axes, got shape {probs.shape}")
    _check_finite(probs, "probabilities")
    if np.any((probs < 0) | (probs > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    obs = _check_bool(outcomes, "outcomes", 2, "(cases, events)")
    if obs.shape != probs.shape:
        raise ValueError(
            f"outcomes must have exactly shape {probs.shape}, got {obs.shape}; "
            "no broadcasting is performed"
        )

    if convention == "binary":
        result = (probs - obs.astype(np.float64)) ** 2
        _check_finite(result, "event_brier_scores result")
        return result

    if not np.all(np.count_nonzero(obs, axis=1) == 1):
        raise ValueError("multiclass outcomes must have exactly one True per case")
    labels = np.argmax(obs, axis=1).astype(np.int64)
    return brier_score(probs, labels)


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


class DecisionRisks(NamedTuple):
    """Per-case results from :func:`decision_risks`.

    :ivar expected_losses: posterior-expected loss of every action, shape
        ``(cases, actions)``.
    :ivar min_risk: smallest expected loss per case, shape ``(cases,)``.
    :ivar bayes_actions: one sorted ``int64`` array per case holding every
        action whose expected loss equals ``min_risk`` exactly (float64
        ``==``); exact ties are all kept and none is chosen.
    """

    expected_losses: np.ndarray
    min_risk: np.ndarray
    bayes_actions: tuple[np.ndarray, ...]


def decision_risks(losses, *, weights=None) -> DecisionRisks:
    """Posterior-expected loss and Bayes action set for every case.

    The caller computes the loss of every action under every draw, in its
    own loss units. Only that table and the draw weights enter; the truth is
    not an argument, so retrospective scores cannot influence the actions.
    Each case is reduced with :func:`~skbel.metrics.expected_action_losses`
    and :func:`~skbel.metrics.bayes_action_set`.

    :param losses: real, finite array ``(cases, draws, actions)``. Every entry
        is validated, including those of zero-weight draws.
    :param weights: ``None`` (uniform) or ``(cases, draws)`` exactly: finite,
        non-negative, with a positive total per case, renormalized per case.
    :return: a :class:`DecisionRisks`.
    :raises ValueError: for malformed inputs or an expected loss that is not
        representable in float64.
    """
    loss_arr = _as_float_array(losses, "losses")
    if loss_arr.ndim != 3:
        raise ValueError(
            f"losses must have shape (cases, draws, actions), got ndim={loss_arr.ndim}"
        )
    if 0 in loss_arr.shape:
        raise ValueError(f"losses must have no empty axes, got shape {loss_arr.shape}")
    _check_finite(loss_arr, "losses")
    cases, draws, actions = loss_arr.shape
    if weights is not None:
        w = _as_float_array(weights, "weights")
        # Validate every case before any reduction; each case is renormalized again below.
        _validate_weights_2d(w, cases, draws)

    expected = np.empty((cases, actions), dtype=np.float64)
    min_risk = np.empty(cases, dtype=np.float64)
    bayes = []
    for c in range(cases):
        w_c = None if weights is None else w[c]
        expected[c] = expected_action_losses(loss_arr[c], w_c)
        min_risk[c], minimizers = bayes_action_set(loss_arr[c], w_c)
        bayes.append(minimizers)
    return DecisionRisks(expected_losses=expected, min_risk=min_risk, bayes_actions=tuple(bayes))
