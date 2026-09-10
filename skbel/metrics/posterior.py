#  Copyright (c) 2026. Robin Thibaut, Ghent University

"""Posterior-sample scoring and finite-action decision-risk primitives.

Everything here is general-purpose evaluation arithmetic: it consumes arrays
the caller already has (posterior draws, class probabilities, a loss table)
and returns per-case scores or risk summaries. Nothing here runs a simulator,
a prior, or model training, and nothing here supplies a hydrological (or any
other domain) loss function -- that is an application responsibility.

Conventions enforced throughout, deliberately strict (no silent broadcasting,
no default clipping of bad inputs, no NaN hiding):

- Shapes are explicit and checked exactly; mismatches raise ``ValueError``.
- Weights, where accepted, must be finite, non-negative, and sum to a
  strictly positive value per case (or overall, for 1-D weights); they are
  then explicitly renormalized by this module (callers do not need to
  pre-normalize).
- Outputs are checked for finiteness before being returned.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "bayes_action_set",
    "brier_score",
    "expected_action_losses",
    "marginal_crps",
]


def _as_float_array(x, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype.kind not in "fiu":
        raise TypeError(f"{name} must be a numeric array, got dtype {arr.dtype}")
    return arr.astype(np.float64, copy=False)


def _check_finite(arr: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (NaN or inf)")


def _validate_weights_2d(weights, cases: int, draws: int) -> np.ndarray:
    """Validate and explicitly normalize per-case weights of shape (cases, draws)."""
    w = _as_float_array(weights, "weights")
    if w.shape != (cases, draws):
        raise ValueError(f"weights must have shape {(cases, draws)}, got {w.shape}")
    _check_finite(w, "weights")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    row_max = np.max(w, axis=1)
    if np.any(row_max <= 0) or not np.all(np.isfinite(row_max)):
        raise ValueError("every case's weights must have a strictly positive sum")
    scaled = w / row_max[:, None]
    row_sums = scaled.sum(axis=1)
    if np.any(~np.isfinite(row_sums)) or np.any(row_sums <= 0):
        raise ValueError("every case's weights must have a finite, strictly positive sum")
    normalized = scaled / row_sums[:, None]
    if np.any(~np.isfinite(normalized)) or not np.all(np.sum(normalized, axis=1) > 0):
        raise ValueError("normalized weights must have finite positive mass")
    return normalized


def _validate_weights_1d(weights, draws: int) -> np.ndarray:
    """Validate and explicitly normalize weights of shape (draws,)."""
    w = _as_float_array(weights, "weights")
    if w.shape != (draws,):
        raise ValueError(f"weights must have shape {(draws,)}, got {w.shape}")
    _check_finite(w, "weights")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    scale = np.max(w)
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("weights must have a strictly positive sum")
    scaled = w / scale
    total = scaled.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("weights must have a finite, strictly positive sum")
    normalized = scaled / total
    if np.any(~np.isfinite(normalized)) or normalized.sum() <= 0:
        raise ValueError("normalized weights must have finite positive mass")
    return normalized


def _uniform_weights_2d(cases: int, draws: int) -> np.ndarray:
    return np.full((cases, draws), 1.0 / draws, dtype=np.float64)


def _weighted_pairwise_abs_diff_sum_sorted(x_1d: np.ndarray, w_1d: np.ndarray) -> float:
    """Return sum_i sum_j w_i * w_j * |x_i - x_j| via an O(M log M) sort.

    Derivation (sorted ascending x_(1) <= ... <= x_(M), weights normalized to
    sum to 1, W_{i-1} = cumulative weight strictly before position i):

        S = 2 * sum_i w_i * x_i * (2*W_{i-1} + w_i - 1)

    which follows from expanding S = 2 * sum_{i<j} w_i w_j (x_j - x_i) and
    using sum_{i<j} w_i = W_{i-1}, sum_{j>i} w_j = 1 - W_i, W_i = W_{i-1} + w_i.
    """
    order = np.argsort(x_1d, kind="mergesort")
    x_sorted = x_1d[order]
    w_sorted = w_1d[order]
    cum_w = np.cumsum(w_sorted)
    w_prev = cum_w - w_sorted  # exclusive cumulative weight before each position
    # Center to reduce cancellation: CRPS pairwise-difference terms are
    # translation invariant, so shifting x by a constant before the
    # multiply-and-sum does not change S but keeps the summed magnitudes small.
    shift = x_sorted.mean()
    x_centered = x_sorted - shift
    terms = w_sorted * x_centered * (2.0 * w_prev + w_sorted - 1.0)
    return float(2.0 * terms.sum())


def _unweighted_pairwise_abs_diff_sum_sorted(x_1d: np.ndarray) -> float:
    """Return sum_i sum_j |x_i - x_j| (raw, unnormalized) via an O(M log M) sort.

    Closed form for sorted ascending x_(1) <= ... <= x_(M), 1-indexed i:

        sum_{i,j} |x_i - x_j| = 2 * sum_i (2*i - M - 1) * x_(i)
    """
    m = x_1d.shape[0]
    order = np.argsort(x_1d, kind="mergesort")
    x_sorted = x_1d[order]
    shift = x_sorted.mean()
    x_centered = x_sorted - shift
    i = np.arange(1, m + 1, dtype=np.float64)
    coeff = 2.0 * i - m - 1.0
    return float(2.0 * np.sum(coeff * x_centered))


def marginal_crps(
    samples: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray | None = None,
    estimator: str = "empirical",
) -> np.ndarray:
    """Per-case, per-target marginal continuous ranked probability score.

    :param samples: posterior draws, shape ``(cases, draws, targets)``.
    :param truth: realized/true values, shape ``(cases, targets)``.
    :param weights: optional per-draw weights, shape ``(cases, draws)``.
        Must be finite, non-negative, with a strictly positive sum per case;
        this function explicitly renormalizes them (they need not already
        sum to 1). If ``None``, draws are weighted uniformly.
    :param estimator: ``"empirical"`` (default) computes the exact CRPS of
        the (possibly weighted) empirical distribution against ``truth`` --
        valid for any non-negative weights and any number of draws
        ``M >= 1``. ``"unbiased"`` computes the finite-sample U-statistic
        ("fair") estimator of CRPS against the fixed value ``truth``, under
        the assumption that draws are unweighted i.i.d. samples from the
        predictive distribution; it requires ``weights is None`` and
        ``M >= 2``, and uses the pair denominator ``M*(M-1)`` (no diagonal
        terms). For a fixed observation ``y``, the triangle inequality gives
        ``sum_{i != j}|x_i - x_j| <= 2*(M-1)*sum_i|x_i - y|``. Therefore this
        U-estimator is non-negative in exact arithmetic; floating-point
        roundoff can produce tiny negative values.
    :return: array of shape ``(cases, targets)``, the marginal score for
        each case/target pair (each target scored independently -- this is
        NOT a joint multivariate score and does not by itself establish
        joint calibration).
    """
    if estimator not in ("empirical", "unbiased"):
        raise ValueError(f"estimator must be 'empirical' or 'unbiased', got {estimator!r}")

    samples_arr = _as_float_array(samples, "samples")
    truth_arr = _as_float_array(truth, "truth")

    if samples_arr.ndim != 3:
        raise ValueError(
            f"samples must have shape (cases, draws, targets), got ndim={samples_arr.ndim}"
        )
    if truth_arr.ndim != 2:
        raise ValueError(f"truth must have shape (cases, targets), got ndim={truth_arr.ndim}")

    cases, draws, targets = samples_arr.shape
    if cases == 0 or draws == 0 or targets == 0:
        raise ValueError(f"samples must have no empty axes, got shape {samples_arr.shape}")
    if truth_arr.shape != (cases, targets):
        raise ValueError(
            f"truth shape {truth_arr.shape} is not compatible with samples shape "
            f"{samples_arr.shape}: expected exactly {(cases, targets)} -- no broadcasting is performed"
        )

    _check_finite(samples_arr, "samples")
    _check_finite(truth_arr, "truth")

    if estimator == "unbiased":
        if weights is not None:
            raise ValueError(
                "estimator='unbiased' requires unweighted i.i.d. draws; got explicit weights. "
                "There is no supported weighted-U-statistic rule in this function."
            )
        if draws < 2:
            raise ValueError("estimator='unbiased' requires at least 2 draws (M >= 2)")

        result = np.empty((cases, targets), dtype=np.float64)
        for c in range(cases):
            for t in range(targets):
                x = samples_arr[c, :, t]
                y = truth_arr[c, t]
                mean_abs_diff_to_y = np.mean(np.abs(x - y))
                raw_pair_sum = _unweighted_pairwise_abs_diff_sum_sorted(x)
                result[c, t] = mean_abs_diff_to_y - raw_pair_sum / (2.0 * draws * (draws - 1))
        _check_finite(result, "marginal_crps result")
        return result

    # estimator == "empirical"
    if weights is None:
        w_norm = _uniform_weights_2d(cases, draws)
    else:
        w_norm = _validate_weights_2d(weights, cases, draws)

    result = np.empty((cases, targets), dtype=np.float64)
    for t in range(targets):
        x_t = samples_arr[:, :, t]  # (cases, draws)
        y_t = truth_arr[:, t]  # (cases,)
        weighted_abs_diff_to_y = np.sum(w_norm * np.abs(x_t - y_t[:, None]), axis=1)  # (cases,)
        for c in range(cases):
            pair_sum = _weighted_pairwise_abs_diff_sum_sorted(x_t[c], w_norm[c])
            result[c, t] = weighted_abs_diff_to_y[c] - 0.5 * pair_sum

    _check_finite(result, "marginal_crps result")
    return result


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Multiclass Brier score, summed over all classes, returned per case.

    :param probabilities: shape ``(cases, classes)``, finite, non-negative,
        each row summing to 1 (within ``1e-8`` absolute tolerance) -- not
        auto-normalized; malformed rows raise.
    :param labels: integer class labels, shape ``(cases,)``, each in
        ``[0, classes)``.
    :return: array of shape ``(cases,)``: ``sum_c (p[c] - 1{c == label})**2``.

        Convention, stated explicitly: this sums the squared error over
        *every* class, including the label's own class and every other
        class. For the binary case (``classes == 2``) this reads as exactly
        twice the single-probability binary Brier-score convention
        ``(p_positive - y)**2``, because both the positive and negative
        class terms contribute equally by symmetry. Callers who want the
        single-probability binary convention should divide this function's
        binary-case output by 2, rather than assuming this function already
        returns it.
    """
    probs = _as_float_array(probabilities, "probabilities")
    labels_arr = np.asarray(labels)

    if probs.ndim != 2:
        raise ValueError(f"probabilities must have shape (cases, classes), got ndim={probs.ndim}")
    cases, classes = probs.shape
    if cases == 0 or classes == 0:
        raise ValueError(f"probabilities must have no empty axes, got shape {probs.shape}")

    if labels_arr.ndim != 1:
        raise ValueError(f"labels must have shape (cases,), got ndim={labels_arr.ndim}")
    if labels_arr.shape[0] != cases:
        raise ValueError(
            f"labels shape {labels_arr.shape} incompatible with probabilities cases={cases}"
        )
    if labels_arr.dtype.kind not in "iu":
        raise TypeError(f"labels must be an integer array, got dtype {labels_arr.dtype}")

    _check_finite(probs, "probabilities")
    if np.any(probs < 0):
        raise ValueError("probabilities must be non-negative")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("every probability row must sum to 1 (within 1e-8 absolute tolerance)")

    if np.any(labels_arr < 0) or np.any(labels_arr >= classes):
        raise ValueError(
            f"labels must lie in [0, {classes}), got range [{labels_arr.min()}, {labels_arr.max()}]"
        )

    one_hot = np.zeros((cases, classes), dtype=np.float64)
    one_hot[np.arange(cases), labels_arr] = 1.0

    result = np.sum((probs - one_hot) ** 2, axis=1)
    _check_finite(result, "brier_score result")
    return result


def expected_action_losses(losses: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Posterior-expected loss per action, from an explicit per-draw loss table.

    This is posterior decision-risk *reduction* over actions the caller has
    already enumerated and already supplied losses for -- not an online
    acquisition/design optimizer, and it carries no hydrological (or any
    other domain) semantics; the loss table's meaning is entirely the
    caller's responsibility.

    :param losses: shape ``(draws, actions)``, finite.
    :param weights: optional shape ``(draws,)``, finite, non-negative, with
        a strictly positive sum; explicitly renormalized by this function.
        If ``None``, draws are weighted uniformly.
    :return: array of shape ``(actions,)``, the weighted-mean loss for each
        action.
    """
    losses_arr = _as_float_array(losses, "losses")
    if losses_arr.ndim != 2:
        raise ValueError(f"losses must have shape (draws, actions), got ndim={losses_arr.ndim}")
    draws, actions = losses_arr.shape
    if draws == 0 or actions == 0:
        raise ValueError(f"losses must have no empty axes, got shape {losses_arr.shape}")
    _check_finite(losses_arr, "losses")

    if weights is None:
        w_norm = np.full(draws, 1.0 / draws, dtype=np.float64)
    else:
        w_norm = _validate_weights_1d(weights, draws)

    result = w_norm @ losses_arr
    _check_finite(result, "expected_action_losses result")
    return result


def bayes_action_set(
    losses: np.ndarray, weights: np.ndarray | None = None
) -> tuple[float, np.ndarray]:
    """Minimum posterior-expected loss and the set of ALL exact minimizers.

    :param losses: shape ``(draws, actions)``, finite.
    :param weights: see :func:`expected_action_losses`.
    :return: ``(min_risk, minimizing_action_indices)`` where
        ``minimizing_action_indices`` is a sorted ``int64`` array containing
        every action index whose expected loss exactly equals ``min_risk``
        (a single best action returns a length-1 array; exact ties return
        every tied index -- there is no arbitrary tie-break).
    """
    expected = expected_action_losses(losses, weights)
    min_risk = float(np.min(expected))
    minimizers = np.flatnonzero(expected == min_risk).astype(np.int64)
    return min_risk, minimizers
