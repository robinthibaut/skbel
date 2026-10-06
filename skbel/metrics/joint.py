#  Copyright (c) 2026. Robin Thibaut, Ghent University

"""Joint (multivariate) empirical posterior scoring.

:func:`energy_score` scores the *supplied* discrete (optionally weighted)
empirical joint distribution ``Q`` of the targets against one realized truth
vector per case, using the Euclidean joint distance:

    ES(Q, y) = sum_i w_i ||x_i - y||_2 - 0.5 * sum_ij w_i w_j ||x_i - x_j||_2

Unlike :func:`skbel.metrics.posterior.marginal_crps` it is sensitive to
dependence between targets. It is an evaluation of the supplied empirical law
only: it makes no claim of being an unbiased score for an unknown continuous
generator, is not decision risk, and does not by itself prove joint
calibration. Target units and scaling are caller choices; nothing here
normalizes heterogeneous coordinates.

Deliberately strict, in line with :mod:`skbel.metrics.posterior`: exact shapes,
no broadcasting, no clipping, no epsilon, no hidden zeroing. Anything that is
not representable in float64 raises ``ValueError``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["energy_score"]

# Target size (rows * draws * targets elements) of the temporary difference
# buffer in the chunked pairwise computation. At least one row is always
# processed, so a single row (draws * targets) can exceed this target, and the
# norm computation holds a few buffers of this size. Work remains
# O(draws^2 * targets) per case; this is not a bound on arbitrary input size.
_PAIR_CHUNK_ELEMENTS = 1 << 18


def _as_float_array(x, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype.kind not in "fiu":
        raise TypeError(f"{name} must be a real numeric array, got dtype {arr.dtype}")
    return arr.astype(np.float64, copy=False)


def _check_finite(arr: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (NaN or inf)")


def _normalized_weights(weights, cases: int, draws: int) -> np.ndarray:
    """Validate ``(cases, draws)`` weights and normalize each case to unit mass.

    Zero weights are allowed (validated, later excluded). A supplied positive
    weight that is lost (becomes zero) in rescaling or normalization raises.
    """
    if weights is None:
        return np.full((cases, draws), 1.0 / draws, dtype=np.float64)
    w = _as_float_array(weights, "weights")
    if w.shape != (cases, draws):
        raise ValueError(
            f"weights must have exactly shape {(cases, draws)}, got {w.shape}; "
            "no broadcasting is performed"
        )
    _check_finite(w, "weights")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    row_max = np.max(w, axis=1)
    if np.any(row_max <= 0):
        raise ValueError("every case's weights must have a strictly positive total")
    scaled = w / row_max[:, None]
    row_sums = scaled.sum(axis=1)
    if not np.all(np.isfinite(row_sums)) or np.any(row_sums <= 0):
        raise ValueError("every case's weights must have a finite, strictly positive total")
    normalized = scaled / row_sums[:, None]
    _check_finite(normalized, "normalized weights")
    if np.any((w > 0) & ~(normalized > 0)):
        raise ValueError(
            "positive weight mass was lost (underflow) during rescaling/normalization; "
            "weights span more than float64 can represent"
        )
    return normalized


def _euclidean_norm(diff: np.ndarray) -> np.ndarray:
    """Overflow-safe Euclidean norm over the last axis, failing if unrepresentable."""
    if not np.all(np.isfinite(diff)):
        raise ValueError("a coordinate difference is not representable in float64")
    with np.errstate(over="ignore", invalid="ignore"):
        peak = np.max(np.abs(diff), axis=-1)
        safe = np.where(peak > 0, peak, 1.0)
        ratio_sq = np.sum((diff / safe[..., None]) ** 2, axis=-1)
        norm = peak * np.sqrt(ratio_sq)
    if not np.all(np.isfinite(norm)):
        raise ValueError("a Euclidean distance is not representable in float64")
    return norm


def _case_score(x: np.ndarray, w: np.ndarray, y: np.ndarray) -> float:
    """Energy score of one case; ``x`` is (n, targets), ``w`` is (n,), all w > 0."""
    n, targets = x.shape
    with np.errstate(over="ignore", invalid="ignore"):
        diff_to_truth = x - y[None, :]
    dist_to_truth = _euclidean_norm(diff_to_truth)
    with np.errstate(over="ignore", invalid="ignore"):
        term1 = float(np.sum(w * dist_to_truth))

    rows = max(1, _PAIR_CHUNK_ELEMENTS // max(1, n * targets))
    term2 = 0.0
    for start in range(0, n, rows):
        stop = min(start + rows, n)
        with np.errstate(over="ignore", invalid="ignore"):
            diff = x[start:stop, None, :] - x[None, :, :]
        dist = _euclidean_norm(diff)  # (rows, n)
        with np.errstate(over="ignore", invalid="ignore"):
            # Weight by w_j then w_i separately (not w_i * w_j) so tiny positive
            # masses are not multiplied into an underflowed product.
            inner = dist @ w
            term2 += float(np.sum(w[start:stop] * inner))
    if not (np.isfinite(term1) and np.isfinite(term2)):
        raise ValueError("energy score terms are not representable in float64")
    score = term1 - 0.5 * term2
    if not np.isfinite(score):
        raise ValueError("energy score is not representable in float64")
    return score


def energy_score(
    samples: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Per-case energy score of the empirical joint posterior law.

    :param samples: posterior draws, real numeric, finite, shape
        ``(cases, draws, targets)`` with no empty axis.
    :param truth: realized values, finite, shape ``(cases, targets)`` exactly.
    :param weights: ``None`` (uniform) or shape ``(cases, draws)`` exactly (no
        broadcasting): finite, non-negative, positive total in every case.
        They are renormalized per case by this function. Zero-weight draws are
        validated and then excluded before any distance is computed. Positive
        mass that is lost in normalization raises ``ValueError``.
    :return: float64 array ``(cases,)`` of
        ``sum_i w_i ||x_i - y|| - 0.5 sum_ij w_i w_j ||x_i - x_j||`` using
        overflow-safe Euclidean norms. Unrepresentable differences, norms or
        objectives raise ``ValueError``; nothing is clipped or scaled.

    The score is non-negative in exact arithmetic, but float64 roundoff can
    yield tiny negative values; no strict positivity is guaranteed or
    enforced.
    Coordinates are used as supplied: choose units/scaling yourself.
    """
    samples_arr = _as_float_array(samples, "samples")
    truth_arr = _as_float_array(truth, "truth")
    if samples_arr.ndim != 3:
        raise ValueError(
            f"samples must have shape (cases, draws, targets), got ndim={samples_arr.ndim}"
        )
    if samples_arr.size == 0:
        raise ValueError(f"samples must have no empty axes, got shape {samples_arr.shape}")
    cases, draws, targets = samples_arr.shape
    if truth_arr.shape != (cases, targets):
        raise ValueError(
            f"truth must have exactly shape {(cases, targets)}, got {truth_arr.shape}; "
            "no broadcasting is performed"
        )
    _check_finite(samples_arr, "samples")
    _check_finite(truth_arr, "truth")
    w_norm = _normalized_weights(weights, cases, draws)

    result = np.empty(cases, dtype=np.float64)
    for c in range(cases):
        keep = w_norm[c] > 0
        result[c] = _case_score(samples_arr[c][keep], w_norm[c][keep], truth_arr[c])
    _check_finite(result, "energy_score result")
    return result
