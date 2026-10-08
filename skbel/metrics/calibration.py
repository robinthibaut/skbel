#  Copyright (c) 2026. Robin Thibaut, Ghent University

"""Sample-based calibration checks for any posterior, given as arrays.

Every function here consumes posterior draws the caller already has, shaped
``(cases, draws, targets)``, together with the realized values, shaped
``(cases, targets)``. Nothing here runs a simulator, a prior or a model fit,
and nothing is specific to BEL: draws from any sampler can be checked.

All checks are *marginal*: each target is assessed on its own. Uniform ranks
or nominal interval coverage for every target does not establish joint
(multivariate) calibration, and none of these functions claims it.

Conventions (shared with :mod:`skbel.metrics.posterior`):

- Shapes are explicit and checked exactly; mismatches raise ``ValueError``.
  Non-numeric (including boolean) arrays raise ``TypeError``.
- Inputs must be finite; there is no clipping and no NaN hiding.
- Weights, where accepted, are finite, non-negative, with a strictly
  positive sum per case, and are explicitly renormalized here.

Randomized tie-breaking draws from a dedicated stream per ``(case, target)``
pair, derived from an integer ``seed`` and explicit case and target
identifiers (see :func:`case_rng`). A given pair therefore receives the same
random tie-break whether it is evaluated alone, in a subset, or in a
reordered batch, as long as it keeps its identifiers. NumPy's global random
state is never read or modified.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from .posterior import _as_float_array, _check_finite, _validate_weights_2d

__all__ = [
    "CoverageSummary",
    "IntervalCoverage",
    "case_rng",
    "empirical_pit",
    "interval_coverage",
    "sbc_rank_histogram",
    "sbc_ranks",
    "summarize_coverage",
]

# Stream label of the shared randomized tie-break of ``sbc_ranks`` and
# ``empirical_pit``; both use the same uniform for a given (case, target).
_TIE_STREAM = "skbel.metrics.calibration.ties"


# ---------------------------------------------------------------------------
# Labeled random streams
# ---------------------------------------------------------------------------


def _check_seed(seed, name: str = "seed") -> int:
    """Return ``seed`` as a Python int; it must be a non-negative integer."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"{name} must be a non-negative integer, got {seed!r}")
    if seed < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {seed!r}")
    return int(seed)


def _canonical_id(value, name: str) -> tuple[bytes, bytes]:
    """Return ``(type tag, payload)`` for an identifier: a non-negative int or a str."""
    if isinstance(value, str):
        return b"s", value.encode("utf-8")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a non-negative integer or a string, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer or a string, got {value!r}")
    return b"i", str(int(value)).encode("ascii")


def _stream_words(stream: str, case_id, target_id) -> tuple[int, ...]:
    """Fixed-width (4 x uint32) key of a labeled stream.

    Each field is type-tagged and length-prefixed before hashing, so the
    integer ``1`` and the string ``"1"`` give different streams, and no
    concatenation of fields can collide with another.
    """
    if not isinstance(stream, str):
        raise TypeError(f"stream must be a string, got {stream!r}")
    fields = [(b"s", stream.encode("utf-8")), _canonical_id(case_id, "case_id")]
    if target_id is None:
        fields.append((b"n", b""))
    else:
        fields.append(_canonical_id(target_id, "target_id"))
    digest = hashlib.blake2b(digest_size=16, person=b"skbel-stream")
    for tag, payload in fields:
        digest.update(tag + len(payload).to_bytes(8, "little") + payload)
    raw = digest.digest()
    return tuple(int.from_bytes(raw[i : i + 4], "little") for i in range(0, 16, 4))


def case_rng(seed, case_id, stream: str, target_id=None) -> np.random.Generator:
    """Independent random generator for one labeled case (and optionally target).

    The generator depends only on ``(seed, stream, case_id, target_id)``. It
    does not depend on how many other cases exist, their order, or which of
    them are evaluated, and it never touches NumPy's global random state.
    Different ``stream`` labels give unrelated streams for the same case, so
    separate uses of one seed (for example posterior sampling and tie
    breaking) do not reuse the same random numbers.

    :param seed: non-negative integer.
    :param case_id: non-negative integer or string identifying the case.
    :param stream: string label naming the purpose of the stream.
    :param target_id: optional non-negative integer or string identifying a
        target within the case.
    :return: a fresh :class:`numpy.random.Generator`.
    """
    words = _stream_words(stream, case_id, target_id)
    sequence = np.random.SeedSequence(entropy=_check_seed(seed), spawn_key=words)
    return np.random.default_rng(sequence)


def _resolve_ids(ids, n: int, name: str) -> list:
    """Default positional ids, or validated unique explicit ids of length ``n``."""
    if ids is None:
        return list(range(n))
    if isinstance(ids, (str, bytes)) or not isinstance(ids, (Sequence, np.ndarray)):
        raise TypeError(f"{name} must be a sequence of length {n}")
    if isinstance(ids, np.ndarray) and ids.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    values = list(ids.tolist() if isinstance(ids, np.ndarray) else ids)
    if len(values) != n:
        raise ValueError(f"{name} must have length {n}, got {len(values)}")
    keys = [_canonical_id(value, name) for value in values]
    if len(set(keys)) != n:
        raise ValueError(f"{name} must be unique")
    return values


def _tie_uniforms(mask: np.ndarray, seed, case_ids, target_ids) -> np.ndarray:
    """One uniform in [0, 1) per masked (case, target) pair, from its own stream."""
    v = np.zeros(mask.shape, dtype=np.float64)
    for c, t in zip(*np.nonzero(mask), strict=True):
        v[c, t] = case_rng(seed, case_ids[c], _TIE_STREAM, target_ids[t]).random()
    return v


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_samples_truth(samples, truth) -> tuple[np.ndarray, np.ndarray]:
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
            f"{samples_arr.shape}: expected exactly {(cases, targets)} -- no broadcasting"
        )
    _check_finite(samples_arr, "samples")
    _check_finite(truth_arr, "truth")
    return samples_arr, truth_arr


def _validate_levels(levels) -> np.ndarray:
    arr = _as_float_array(levels, "levels")
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1 or arr.shape[0] == 0:
        raise ValueError("levels must be a scalar or a non-empty one-dimensional array")
    _check_finite(arr, "levels")
    if np.any((arr <= 0) | (arr >= 1)):
        raise ValueError("every level must lie strictly between 0 and 1")
    return arr.copy()


# ---------------------------------------------------------------------------
# Ranks and PIT
# ---------------------------------------------------------------------------


def sbc_ranks(
    samples: np.ndarray,
    truth: np.ndarray,
    *,
    ties: str = "randomize",
    seed=None,
    case_ids=None,
    target_ids=None,
) -> np.ndarray:
    """Simulation-based calibration ranks of the truth among unweighted draws.

    For each case and target the rank is the number of draws strictly below
    the truth, plus a tie-break over draws exactly equal to it. With ``M``
    draws the rank lies in ``{0, ..., M}``.

    The ranks are uniform on ``{0, ..., M}`` when the truth and the ``M``
    draws are exchangeable -- for example, the truth is drawn from the
    prior, data are simulated from it, and the draws are independent exact
    posterior draws given those data. Autocorrelated draws (such as raw MCMC
    output) are not exchangeable with the truth, and thinning reduces the
    autocorrelation but does not by itself make them exchangeable. This
    function does not check exchangeability. Weighted draws are
    not accepted: a weighted analogue is :func:`empirical_pit` with
    ``weights``, which is not an SBC rank.

    :param samples: draws, shape ``(cases, draws, targets)``.
    :param truth: realized values, shape ``(cases, targets)``.
    :param ties: ``"randomize"`` (default) adds ``floor(v * (n_equal + 1))``,
        with ``v`` uniform in ``[0, 1)``, so the tied truth takes a uniformly
        random position among the ``n_equal + 1`` tied values; requires
        ``seed``. ``"error"`` raises ``ValueError`` if any draw equals the
        truth exactly and needs no seed.
    :param seed: non-negative integer, required for ``ties="randomize"``.
    :param case_ids: optional unique non-negative integers or strings, one per
        case; default ``0..cases-1``. Tie-breaks are reproducible under case
        reordering or subsetting only when explicit ids travel with the cases.
    :param target_ids: optional unique ids, one per target; default
        ``0..targets-1``.
    :return: ``int64`` array of shape ``(cases, targets)``.
    """
    samples_arr, truth_arr = _validate_samples_truth(samples, truth)
    if ties not in ("randomize", "error"):
        raise ValueError(f"ties must be 'randomize' or 'error', got {ties!r}")
    cases, _, targets = samples_arr.shape
    case_ids = _resolve_ids(case_ids, cases, "case_ids")
    target_ids = _resolve_ids(target_ids, targets, "target_ids")

    y = truth_arr[:, None, :]
    below = np.sum(samples_arr < y, axis=1)
    equal = np.sum(samples_arr == y, axis=1)

    if ties == "error":
        if np.any(equal > 0):
            raise ValueError("some draws equal the truth exactly; use ties='randomize'")
        return below.astype(np.int64)

    if seed is None:
        raise ValueError("ties='randomize' requires an integer seed")
    _check_seed(seed)
    v = _tie_uniforms(equal > 0, seed, case_ids, target_ids)
    offset = np.minimum(np.floor(v * (equal + 1)), equal)
    return (below + offset).astype(np.int64)


def sbc_rank_histogram(ranks: np.ndarray, n_draws: int, n_bins: int | None = None) -> np.ndarray:
    """Counts of SBC ranks in equal-width bins, per target.

    :param ranks: integer ranks from :func:`sbc_ranks`, shape
        ``(cases, targets)``, each in ``[0, n_draws]``.
    :param n_draws: number of draws ``M`` used to compute the ranks.
    :param n_bins: number of bins; must divide ``M + 1`` so every bin holds
        the same number of possible ranks. Default ``M + 1`` (one bin per
        rank).
    :return: ``int64`` array of shape ``(n_bins, targets)``. Under uniform
        ranks and independent cases each count is Binomial with ``cases``
        trials and probability ``1 / n_bins``.
    """
    n_draws = _check_seed(n_draws, "n_draws")
    if n_draws < 1:
        raise ValueError("n_draws must be at least 1")
    n_values = n_draws + 1
    if n_bins is None:
        n_bins = n_values
    n_bins = _check_seed(n_bins, "n_bins")
    if n_bins < 1 or n_values % n_bins != 0:
        raise ValueError(f"n_bins must be a positive divisor of n_draws + 1 = {n_values}")
    ranks_arr = np.asarray(ranks)
    if ranks_arr.dtype.kind not in "iu":
        raise TypeError(f"ranks must be an integer array, got dtype {ranks_arr.dtype}")
    if ranks_arr.ndim != 2 or 0 in ranks_arr.shape:
        raise ValueError("ranks must have non-empty shape (cases, targets)")
    if np.any(ranks_arr < 0) or np.any(ranks_arr > n_draws):
        raise ValueError(f"ranks must lie in [0, {n_draws}]")
    bins = ranks_arr // (n_values // n_bins)
    counts = np.zeros((n_bins, ranks_arr.shape[1]), dtype=np.int64)
    for t in range(ranks_arr.shape[1]):
        counts[:, t] = np.bincount(bins[:, t], minlength=n_bins)
    return counts


def empirical_pit(
    samples: np.ndarray,
    truth: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    ties: str = "randomize",
    seed=None,
    case_ids=None,
    target_ids=None,
) -> np.ndarray:
    """Probability integral transform of the truth under the empirical draws.

    With ``F(y-)`` the (weighted) fraction of draws strictly below the truth
    and ``a`` the (weighted) fraction exactly equal to it, the PIT is
    ``F(y-) + u * a`` with ``u`` set by ``ties``. Without ties this is the
    empirical CDF at the truth, so for ``M`` unweighted draws it takes values
    on the grid ``k / M``.

    For unweighted draws and ``ties="randomize"`` the PIT uses the same
    uniform as :func:`sbc_ranks` with the same seed and ids, so both describe
    the same tie position. With ``weights`` the result is the weighted
    empirical PIT (for example under importance weights); it inherits any
    weight error and is not an SBC rank.

    :param samples: draws, shape ``(cases, draws, targets)``.
    :param truth: realized values, shape ``(cases, targets)``.
    :param weights: optional per-draw weights, shape ``(cases, draws)``,
        renormalized per case.
    :param ties: atom convention. ``"randomize"`` (default, requires
        ``seed``) uses ``u`` uniform in ``[0, 1)``, the randomized PIT for
        predictive distributions with atoms; ``"midpoint"`` uses ``u = 0.5``;
        ``"error"`` raises ``ValueError`` if any draw equals the truth.
    :param seed: non-negative integer, required for ``ties="randomize"``.
    :param case_ids: optional unique case ids; see :func:`sbc_ranks`.
    :param target_ids: optional unique target ids; see :func:`sbc_ranks`.
    :return: float array of shape ``(cases, targets)`` in ``[0, 1]``
        (rounding of weighted sums is clipped to that range).
    """
    samples_arr, truth_arr = _validate_samples_truth(samples, truth)
    if ties not in ("randomize", "midpoint", "error"):
        raise ValueError(f"ties must be 'randomize', 'midpoint' or 'error', got {ties!r}")
    cases, draws, targets = samples_arr.shape
    case_ids = _resolve_ids(case_ids, cases, "case_ids")
    target_ids = _resolve_ids(target_ids, targets, "target_ids")

    y = truth_arr[:, None, :]
    is_below = samples_arr < y
    is_equal = samples_arr == y
    if weights is None:
        below = np.sum(is_below, axis=1) / draws
        atom = np.sum(is_equal, axis=1) / draws
    else:
        w = _validate_weights_2d(weights, cases, draws)[:, :, None]
        below = np.sum(np.where(is_below, w, 0.0), axis=1)
        atom = np.sum(np.where(is_equal, w, 0.0), axis=1)

    has_atom = atom > 0
    if ties == "error":
        if np.any(has_atom):
            raise ValueError("some draws equal the truth exactly; choose a tie convention")
        u = np.zeros_like(atom)
    elif ties == "midpoint":
        u = np.full_like(atom, 0.5)
    else:
        if seed is None:
            raise ValueError("ties='randomize' requires an integer seed")
        _check_seed(seed)
        u = _tie_uniforms(has_atom, seed, case_ids, target_ids)

    pit = np.clip(below + u * atom, 0.0, 1.0)
    _check_finite(pit, "empirical_pit result")
    return pit


# ---------------------------------------------------------------------------
# Central intervals
# ---------------------------------------------------------------------------


class IntervalCoverage(NamedTuple):
    """Per-case central intervals from :func:`interval_coverage`.

    Arrays indexed by case, level and target have shape
    ``(cases, levels, targets)``.

    :ivar levels: the nominal central levels, shape ``(levels,)``.
    :ivar lower: lower interval endpoints.
    :ivar upper: upper interval endpoints.
    :ivar width: ``upper - lower``; always finite.
    :ivar covered: boolean, ``lower <= truth <= upper``.
    :ivar exchangeable_coverage: for unweighted draws, the exact probability
        that the truth falls in the interval when it is exchangeable with the
        ``M`` draws and their common distribution is continuous (no atoms,
        so ties have probability zero), shape ``(levels,)``. It differs from
        the nominal level because only ``M`` draws are available. With atoms,
        tied endpoints can make the closed interval cover more often.
        ``None`` for weighted draws.
    """

    levels: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    width: np.ndarray
    covered: np.ndarray
    exchangeable_coverage: np.ndarray | None


class CoverageSummary(NamedTuple):
    """Case-averaged interval results from :func:`summarize_coverage`.

    :ivar levels: nominal levels, shape ``(levels,)``.
    :ivar n_cases: number of cases averaged.
    :ivar coverage: fraction of cases covered, shape ``(levels, targets)``.
    :ivar nominal_se: ``sqrt(level * (1 - level) / n_cases)``, shape
        ``(levels,)``: the standard deviation of ``coverage`` if every case
        were covered independently with probability equal to the level.
    :ivar mean_width: mean interval width, shape ``(levels, targets)``,
        computed without intermediate overflow.
    :ivar exchangeable_coverage: copied from the :class:`IntervalCoverage`.
    """

    levels: np.ndarray
    n_cases: int
    coverage: np.ndarray
    nominal_se: np.ndarray
    mean_width: np.ndarray
    exchangeable_coverage: np.ndarray | None


def interval_coverage(
    samples: np.ndarray,
    truth: np.ndarray,
    levels,
    *,
    weights: np.ndarray | None = None,
) -> IntervalCoverage:
    """Central empirical intervals per case, level and target, and whether they cover.

    For level ``c`` the endpoints are the empirical quantiles at
    ``p = (1 - c) / 2`` and ``p = (1 + c) / 2``, using the inverse-CDF
    definition ``Q(p) = min{x_k : F(x_k) >= p}`` on the sorted (weighted)
    draws. The interval is closed, so it holds at least a fraction ``c`` of
    the (weighted) empirical mass. Probabilities are compared in floating
    point.

    :param samples: draws, shape ``(cases, draws, targets)``.
    :param truth: realized values, shape ``(cases, targets)``.
    :param levels: a level or a one-dimensional array of levels, each
        strictly between 0 and 1.
    :param weights: optional per-draw weights, shape ``(cases, draws)``,
        renormalized per case.
    :return: an :class:`IntervalCoverage`. Aggregate it explicitly with
        :func:`summarize_coverage`.
    :raises ValueError: if a width ``upper - lower`` is not representable as
        a finite float (finite endpoints of opposite sign near the float
        limits).
    """
    samples_arr, truth_arr = _validate_samples_truth(samples, truth)
    level_arr = _validate_levels(levels)
    cases, draws, targets = samples_arr.shape
    probs = np.stack([(1.0 - level_arr) / 2.0, (1.0 + level_arr) / 2.0])  # (2, levels)

    # Cumulative weight is compared with p times the total weight. Unweighted draws
    # count 1, 2, ..., M; weights are rescaled to a maximum of 1, so uniform weights
    # give exactly the same comparisons as unweighted draws.
    if weights is None:
        w = None
        counts = np.arange(1, draws + 1, dtype=np.float64)
        # Index of the first sorted draw whose CDF reaches p: the same for every case.
        index = np.sum(counts[:, None, None] < probs[None] * draws, axis=0)  # (2, levels)
        index = np.minimum(index, draws - 1)
        exchangeable = (index[1] - index[0]) / (draws + 1.0)
    else:
        w = _validate_weights_2d(weights, cases, draws)
        w = w / np.max(w, axis=1, keepdims=True)
        exchangeable = None

    lower = np.empty((cases, level_arr.shape[0], targets), dtype=np.float64)
    upper = np.empty_like(lower)
    rows = np.arange(cases)[:, None]
    for t in range(targets):
        order = np.argsort(samples_arr[:, :, t], axis=1, kind="stable")
        x_sorted = np.take_along_axis(samples_arr[:, :, t], order, axis=1)
        if w is None:
            idx = np.broadcast_to(index[:, None, :], (2, cases, level_arr.shape[0]))
        else:
            cum = np.cumsum(np.take_along_axis(w, order, axis=1), axis=1)
            target = probs[:, None, None, :] * cum[None, :, -1:, None]  # (2, cases, 1, levels)
            idx = np.sum(cum[None, :, :, None] < target, axis=2)
            idx = np.minimum(idx, draws - 1)
        lower[:, :, t] = x_sorted[rows, idx[0]]
        upper[:, :, t] = x_sorted[rows, idx[1]]

    with np.errstate(over="ignore"):
        width = upper - lower
    _check_finite(width, "interval width")

    y = truth_arr[:, None, :]
    covered = (lower <= y) & (y <= upper)
    return IntervalCoverage(
        levels=level_arr,
        lower=lower,
        upper=upper,
        width=width,
        covered=covered,
        exchangeable_coverage=exchangeable,
    )


def summarize_coverage(result: IntervalCoverage) -> CoverageSummary:
    """Average per-case interval results over cases, separately per level and target.

    Targets are never pooled: a pooled rate would mix targets with different
    behavior and still say nothing about joint coverage. ``nominal_se``
    treats cases as independent; dependent cases (for example overlapping
    time windows) make it too small.

    The mean width is computed relative to the largest width, so finite
    widths near the float limit average to a finite value instead of
    overflowing in the sum.

    :param result: output of :func:`interval_coverage`.
    :return: a :class:`CoverageSummary`.
    :raises ValueError: if the widths are not finite.
    """
    if not isinstance(result, IntervalCoverage):
        raise TypeError("result must be an IntervalCoverage from interval_coverage")
    n_cases = result.covered.shape[0]
    width = np.asarray(result.width, dtype=np.float64)
    _check_finite(width, "interval width")
    scale = np.max(np.abs(width), axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    # The mean of width / scale lies in [-1, 1]; clipping to the largest width
    # removes rounding above it, so the product cannot overflow.
    with np.errstate(over="ignore"):
        mean_width = np.minimum(np.mean(width / scale, axis=0) * scale, np.max(width, axis=0))
    _check_finite(mean_width, "mean interval width")
    return CoverageSummary(
        levels=result.levels,
        n_cases=n_cases,
        coverage=result.covered.mean(axis=0),
        nominal_se=np.sqrt(result.levels * (1.0 - result.levels) / n_cases),
        mean_width=mean_width,
        exchangeable_coverage=result.exchangeable_coverage,
    )
