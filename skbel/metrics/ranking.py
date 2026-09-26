#  Copyright (c) 2026. Robin Thibaut, Ghent University

"""Deterministic ranking of prospective measurement candidates.

This module ranks a caller-supplied, explicit set of prospective candidates
(e.g. proposed sampling locations, sensor placements, or survey designs)
using a single scalar risk or utility value per candidate that the caller has
*already computed*. Typically that scalar comes from
:func:`skbel.metrics.expected_action_losses` / :func:`skbel.metrics.bayes_action_set`
applied to a hypothetical posterior update for each candidate, but this
module does not know or care how the score was produced: it performs no
simulation, no posterior update, and carries no hydrological (or any other
domain) semantics. Computing a *prospective* score for a not-yet-collected
measurement (e.g. by simulating an expected future data set and its effect
on posterior risk) is entirely the caller's responsibility.

This is a ranking primitive, not a sequential/greedy design optimizer or a
policy framework: it orders a fixed, explicit candidate set once from a
fixed set of scores; it does not search over designs or choose actions.

Example
-------
>>> from skbel.metrics import rank_prospective_measurements
>>> candidates = ["well_A", "well_B", "well_C"]
>>> risks = [0.42, 0.10, 0.10]  # caller-computed prospective posterior risk
>>> ranking = rank_prospective_measurements(candidates, risks, criterion="risk")
>>> ranking.order
('well_B', 'well_C', 'well_A')
>>> ranking.best  # exact ties for the best (lowest-risk) score
('well_B', 'well_C')
>>> ranking.ranks["well_A"]
3
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import NamedTuple

import numpy as np

__all__ = ["ProspectiveRanking", "rank_prospective_measurements"]

_CRITERIA = ("risk", "utility")


class ProspectiveRanking(NamedTuple):
    """Result of :func:`rank_prospective_measurements`.

    :ivar order: candidates from best to worst; ``order[0]`` is (a) best.
        Exact ties keep the candidates' original relative input order
        (a stable sort), so the result is fully deterministic for a given
        input -- there is no arbitrary or random tie-break.
    :ivar best: every candidate exactly tied for the best score, in their
        original input order (a single best candidate returns a length-1
        tuple; this mirrors :func:`skbel.metrics.bayes_action_set`, which
        likewise reports the full exact-tie set rather than picking one).
    :ivar ranks: mapping from each candidate to its 1-based standard
        competition rank (``"1224"`` ranking): candidates with exactly
        equal scores share the same rank, and the next distinct score's
        rank skips ahead by the number of tied candidates.
    """

    order: tuple[Hashable, ...]
    best: tuple[Hashable, ...]
    ranks: dict[Hashable, int]


def rank_prospective_measurements(
    candidates: Sequence[Hashable],
    scores,
    *,
    criterion: str = "risk",
) -> ProspectiveRanking:
    """Deterministically rank explicit candidates by a caller-supplied score.

    :param candidates: non-empty sequence of hashable, unique candidate
        identifiers (e.g. names, coordinates-as-tuples, or any other object
        the caller uses to identify a prospective measurement). Order is
        significant only as the deterministic tie-break (see below); it
        does not otherwise affect the ranking.
    :param scores: one prospective risk-or-utility value per candidate,
        shape ``(len(candidates),)``, finite. What the value represents
        (e.g. posterior-expected loss from
        :func:`skbel.metrics.expected_action_losses` for a simulated future
        measurement) is entirely the caller's responsibility.
    :param criterion: ``"risk"`` (default) ranks lower scores as better
        (e.g. expected loss/decision risk); ``"utility"`` ranks higher
        scores as better (e.g. expected information gain). No other value
        is accepted.
    :return: a :class:`ProspectiveRanking`.
    :raises TypeError: if ``candidates`` contains unhashable elements or
        ``scores`` is not numeric.
    :raises ValueError: if ``candidates`` is empty, contains duplicates, has
        a length mismatched with ``scores``, if ``scores`` is not exactly
        1-D, contains non-finite values, or if ``criterion`` is not one of
        ``"risk"``/``"utility"``.
    """
    if criterion not in _CRITERIA:
        raise ValueError(f"criterion must be one of {_CRITERIA}, got {criterion!r}")

    candidates_tuple = tuple(candidates)
    n = len(candidates_tuple)
    if n == 0:
        raise ValueError("candidates must be non-empty")

    try:
        seen = set(candidates_tuple)
    except TypeError as exc:
        raise TypeError("candidates must be hashable") from exc
    if len(seen) != n:
        raise ValueError("candidates must not contain duplicates")

    scores_arr = np.asarray(scores)
    if scores_arr.dtype.kind not in "fiu":
        raise TypeError(f"scores must be a numeric array, got dtype {scores_arr.dtype}")
    if scores_arr.dtype.kind == "f":
        scores_arr = scores_arr.astype(np.float64, copy=False)
    # Integer scores (dtype kind "i"/"u") are kept at their exact integer
    # dtype rather than cast to float64: float64 has only a 53-bit mantissa,
    # so adjacent large integers (e.g. 2**53 and 2**53 + 1) would collapse
    # into a false tie.

    if scores_arr.ndim != 1:
        raise ValueError(f"scores must have shape (candidates,), got ndim={scores_arr.ndim}")
    if scores_arr.shape[0] != n:
        raise ValueError(f"scores shape {scores_arr.shape} incompatible with {n} candidates")
    if not np.all(np.isfinite(scores_arr)):
        raise ValueError("scores contains non-finite values (NaN or inf)")

    if criterion == "risk":
        key = scores_arr
    elif scores_arr.dtype.kind in "iu":
        # Negate via arbitrary-precision Python ints rather than fixed-width
        # numpy negation: negating the minimum representable signed integer
        # (or any unsigned integer) in a fixed-width dtype overflows/wraps
        # silently and would corrupt utility ordering.
        key = np.array([-int(value) for value in scores_arr], dtype=object)
    else:
        key = -scores_arr

    # Stable sort: exact ties keep the original candidate order, so the
    # result depends only on (candidates order, scores), never on sort
    # implementation details or randomness.
    order_idx = np.argsort(key, kind="mergesort")
    order = tuple(candidates_tuple[i] for i in order_idx)

    sorted_key = key[order_idx]
    best_key = sorted_key[0]
    best = tuple(candidates_tuple[i] for i in order_idx if key[i] == best_key)

    ranks: dict[Hashable, int] = {}
    current_rank = 1
    for position, idx in enumerate(order_idx):
        if position > 0 and sorted_key[position] != sorted_key[position - 1]:
            current_rank = position + 1
        ranks[candidates_tuple[idx]] = current_rank

    return ProspectiveRanking(order=order, best=best, ranks=ranks)
