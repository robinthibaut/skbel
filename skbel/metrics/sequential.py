#  Copyright (c) 2026. Robin Thibaut, Ghent University

"""Finite adaptive acquisition and stopping under an explicit joint model.

The caller supplies a finite set of *atoms* that define the full joint
distribution of loss and measurement outcomes: ``weights[atom]`` is the
probability mass of an atom, ``losses[atom, action]`` the loss of each terminal
action if that atom is the truth, and ``outcomes[atom, candidate]`` the integer
label each candidate measurement would return. No independence between
measurements is assumed and nothing is simulated or fitted.

At every node of the decision tree this module compares

- stopping now, with risk ``min_a E[loss_a]`` under the node's conditional
  distribution, against
- acquiring one not-yet-acquired candidate: its cost plus the
  outcome-probability-weighted optimal value of the conditioned remaining tree.

Acquisition is without replacement; costs already paid on the path to a node
are sunk and are not part of that node's value, so each cost enters the root
objective exactly once, at the node where it is incurred. The recursion is
cut off after ``horizon`` further acquisitions (0, 1 or 2).

Tie semantics, stated precisely: all arithmetic is IEEE float64 and minimizers
are located with exact ``==``. There is no ``isclose``, tolerance or clamping.
Two choices tie in the returned policy only if their float64 values are
exactly equal under ``==`` (so ``0.0`` ties with ``-0.0``; this is not a bit
pattern comparison); values that are equal in exact arithmetic but differ by
rounding are *not* reported as ties, and no guarantee is made beyond float64. The one
rounding-sensitive step this module controls: when a measurement cannot split
the current support (a single outcome), the conditional weights are passed to
the child unchanged and the branch probability is exactly ``1.0``, so an
uninformative free measurement ties exactly with stopping for any float input.

Complexity is exponential in ``horizon``: roughly
``1 + K*B + K*(K-1)*B**2`` nodes for ``K`` candidates with at most ``B``
outcome labels, each costing ``O(atoms * actions)``. Fixture sizes used in the
tests are acceptance bounds, not a guarantee of runtime or memory for
arbitrary user input.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

__all__ = [
    "AcquisitionBranch",
    "AcquisitionNode",
    "CandidateEvidence",
    "finite_acquisition_policy",
]

_MAX_HORIZON = 2


@dataclass(frozen=True)
class AcquisitionBranch:
    """One positive-mass outcome of an acquisition.

    :param probability: conditional probability of this outcome label given the
        parent node's history (strictly positive; branches of one candidate sum
        to 1 up to float64 rounding).
    :param node: the policy node reached after observing the outcome.
    """

    probability: float
    node: AcquisitionNode


@dataclass(frozen=True)
class CandidateEvidence:
    """Auditable value decomposition of acquiring one candidate at a node.

    ``value == immediate_cost + expected_continuation_value`` exactly, and
    ``expected_continuation_value`` is the sum over ``branches`` of
    ``probability * node.value``.

    :param value: total expected objective of acquiring this candidate now.
    :param immediate_cost: the candidate's cost, paid at this node.
    :param expected_continuation_value: outcome-weighted optimal remaining value.
    :param branches: read-only map from integer outcome label to
        :class:`AcquisitionBranch`; only labels with positive conditional mass.
    """

    value: float
    immediate_cost: float
    expected_continuation_value: float
    branches: Mapping[int, AcquisitionBranch]


@dataclass(frozen=True)
class AcquisitionNode:
    """Optimal-policy decision node under one conditioning history.

    :param value: optimal expected remaining objective (risk plus future costs;
        costs already paid on the path are excluded).
    :param stop_risk: ``min_a E[loss_a]`` under this node's conditional weights.
    :param terminal_actions: sorted indices of every action whose expected loss
        exactly equals ``stop_risk`` (the terminal Bayes action set).
    :param optimal_choices: every choice whose value exactly equals ``value``:
        the literal ``"stop"`` (first, if it ties) then candidate indices in
        ascending order. A single entry means a unique optimum.
    :param candidate_values: read-only map from each legal (not yet acquired)
        candidate index to its :class:`CandidateEvidence`, ascending; empty when
        ``horizon == 0`` or no candidate is left.
    :param horizon: number of further acquisitions this node may still make.
    :param acquired: candidate indices acquired on the path to this node, in
        order.
    :param support: sorted original atom indices carrying positive conditional
        mass at this node.
    """

    value: float
    stop_risk: float
    terminal_actions: tuple[int, ...]
    optimal_choices: tuple[int | str, ...]
    candidate_values: Mapping[int, CandidateEvidence]
    horizon: int
    acquired: tuple[int, ...]
    support: tuple[int, ...]


@dataclass(frozen=True)
class _Problem:
    losses: np.ndarray
    outcomes: np.ndarray
    costs: np.ndarray


def _as_float_array(x, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype.kind not in "fiu":
        raise TypeError(f"{name} must be a numeric array, got dtype {arr.dtype}")
    return arr.astype(np.float64)


def _check_finite(arr: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (NaN or inf)")


def _validate_horizon(horizon) -> int:
    if isinstance(horizon, (bool, np.bool_)) or not isinstance(horizon, (int, np.integer)):
        raise TypeError(f"horizon must be an integer 0, 1 or 2, got {type(horizon).__name__}")
    if not 0 <= horizon <= _MAX_HORIZON:
        raise ValueError(f"horizon must be 0, 1 or 2, got {horizon}")
    return int(horizon)


def _validate_losses(losses) -> np.ndarray:
    arr = _as_float_array(losses, "losses")
    if arr.ndim != 2:
        raise ValueError(f"losses must have shape (atoms, actions), got ndim={arr.ndim}")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"losses must have no empty axes, got shape {arr.shape}")
    _check_finite(arr, "losses")
    return arr


def _validate_outcomes(outcomes, atoms: int) -> np.ndarray:
    arr = np.asarray(outcomes)
    if arr.ndim != 2:
        raise ValueError(f"outcomes must have shape (atoms, candidates), got ndim={arr.ndim}")
    if arr.shape[0] != atoms:
        raise ValueError(f"outcomes must have {atoms} rows (one per atom), got shape {arr.shape}")
    if arr.size == 0:
        # Zero candidates: an empty list has float dtype but carries no labels.
        return np.zeros(arr.shape, dtype=np.int64)
    if arr.dtype.kind not in "iu":
        raise TypeError(f"outcomes must be an integer array, got dtype {arr.dtype}")
    if arr.dtype.kind == "u" and arr.dtype.itemsize == 8 and arr.max() > np.iinfo(np.int64).max:
        raise ValueError("outcomes labels must fit in int64")
    return arr.astype(np.int64)


def _validate_costs(costs, candidates: int) -> np.ndarray:
    if costs is None:
        return np.zeros(candidates, dtype=np.float64)
    arr = _as_float_array(costs, "costs")
    if arr.shape != (candidates,):
        raise ValueError(f"costs must have shape {(candidates,)}, got {arr.shape}")
    _check_finite(arr, "costs")
    if np.any(arr < 0):
        raise ValueError("costs must be non-negative")
    return arr


def _normalize_weights(weights, atoms: int) -> np.ndarray:
    """Validate weights and divide by their total; no epsilon is introduced."""
    if weights is None:
        w = np.ones(atoms, dtype=np.float64)
    else:
        w = _as_float_array(weights, "weights")
        if w.shape != (atoms,):
            raise ValueError(f"weights must have shape {(atoms,)}, got {w.shape}")
        _check_finite(w, "weights")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
    peak = w.max()
    if not peak > 0:
        raise ValueError("weights must have a strictly positive total")
    supplied_positive = w > 0
    with np.errstate(over="ignore"):
        total = w.sum()
        if not np.isfinite(total):
            # Finite weights whose plain sum overflows: rescale, then sum.
            w = w / peak
            total = w.sum()
    normalized = w / total
    if np.any(supplied_positive & ~(normalized > 0)):
        raise ValueError(
            "a positive weight is not representable in float64 after normalization "
            "(dynamic range too large); rescale or drop it explicitly"
        )
    return normalized


def _solve(
    problem: _Problem,
    atoms: np.ndarray,
    weights: np.ndarray,
    acquired: tuple[int, ...],
    horizon: int,
) -> AcquisitionNode:
    """Optimal node for the conditional distribution ``weights`` on ``atoms``.

    ``atoms`` holds original atom indices with strictly positive conditional
    mass; ``weights`` are their conditional probabilities.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        risks = weights @ problem.losses[atoms]
    _check_finite(risks, "expected action losses")
    stop_risk = float(risks.min())
    terminal_actions = tuple(int(a) for a in np.flatnonzero(risks == stop_risk))

    evidence: dict[int, CandidateEvidence] = {}
    if horizon > 0:
        total = float(weights.sum())
        for candidate in range(problem.outcomes.shape[1]):
            if candidate in acquired:
                continue
            evidence[candidate] = _candidate_evidence(
                problem, atoms, weights, total, acquired, horizon, candidate
            )

    best = min([stop_risk, *(e.value for e in evidence.values())])
    choices: list[int | str] = []
    if stop_risk == best:
        choices.append("stop")
    choices.extend(c for c, e in evidence.items() if e.value == best)

    return AcquisitionNode(
        value=best,
        stop_risk=stop_risk,
        terminal_actions=terminal_actions,
        optimal_choices=tuple(choices),
        candidate_values=MappingProxyType(evidence),
        horizon=horizon,
        acquired=acquired,
        support=tuple(int(a) for a in atoms),
    )


def _candidate_evidence(
    problem: _Problem,
    atoms: np.ndarray,
    weights: np.ndarray,
    total: float,
    acquired: tuple[int, ...],
    horizon: int,
    candidate: int,
) -> CandidateEvidence:
    labels = problem.outcomes[atoms, candidate]
    branches: dict[int, AcquisitionBranch] = {}
    expected = 0.0
    for label in np.unique(labels):
        chosen = labels == label
        if chosen.all():
            # No split: identical conditional distribution, probability 1.
            child_weights = weights
            probability = 1.0
        else:
            mass = float(weights[chosen].sum())
            child_weights = weights[chosen] / mass
            probability = mass / total
        child = _solve(problem, atoms[chosen], child_weights, (*acquired, candidate), horizon - 1)
        branches[int(label)] = AcquisitionBranch(probability=probability, node=child)
        expected += probability * child.value
    cost = float(problem.costs[candidate])
    value = cost + expected
    if not math.isfinite(value):
        raise ValueError("acquisition value is not finite (cost or loss overflow)")
    return CandidateEvidence(
        value=value,
        immediate_cost=cost,
        expected_continuation_value=expected,
        branches=MappingProxyType(branches),
    )


def finite_acquisition_policy(
    losses: np.ndarray,
    outcomes: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    costs: np.ndarray | None = None,
    horizon: int = 2,
) -> AcquisitionNode:
    """Optimal adaptive acquisition and stopping policy over a finite joint model.

    :param losses: shape ``(atoms, actions)``, finite numeric; loss of each
        terminal action if the atom is the truth.
    :param outcomes: shape ``(atoms, candidates)``, integer categorical labels
        (any integers; bool and float arrays are rejected) that each candidate
        returns for each atom. ``candidates == 0`` is allowed and stops.
    :param weights: optional shape ``(atoms,)``, finite, non-negative, with a
        strictly positive total; explicitly normalized by this function. If
        ``None``, atoms are weighted uniformly. The atoms define the *joint*
        distribution; nothing is treated as a product of marginals. Atoms with
        a supplied weight of exactly zero are removed from all conditioning, so
        their outcome labels never become branches; their losses and labels are
        still validated. A *positive* supplied weight that normalizes to zero in
        float64 (extreme dynamic range, including after overflow rescaling) is
        a representability failure, not impossible support: it raises
        ``ValueError`` and is never dropped, clamped or approximated.
    :param costs: optional shape ``(candidates,)``, finite, non-negative cost of
        acquiring each candidate; zero if ``None``.
    :param horizon: exactly the integer 0, 1 or 2 (bool or non-integer raises
        ``TypeError``, out-of-range raises ``ValueError``): the maximum number
        of further acquisitions. Horizon 0, or no candidate left, stops.
    :return: the root :class:`AcquisitionNode`. Every legal candidate and every
        positive-mass outcome is exposed so the objective decomposition
        (``value == immediate_cost + expected_continuation_value``) can be
        audited independently. Inputs are copied, never modified.

    Minimizers use exact float64 ``==``; see the module docstring for tie
    semantics and complexity.
    """
    horizon = _validate_horizon(horizon)
    loss_arr = _validate_losses(losses)
    n_atoms = loss_arr.shape[0]
    outcome_arr = _validate_outcomes(outcomes, n_atoms)
    cost_arr = _validate_costs(costs, outcome_arr.shape[1])
    normalized = _normalize_weights(weights, n_atoms)

    # Normalization rejects lost positive mass, so this keeps exactly the supplied
    # positive-weight atoms; true zero-mass atoms are the only ones dropped.
    atoms = np.flatnonzero(normalized > 0)
    problem = _Problem(losses=loss_arr, outcomes=outcome_arr, costs=cost_arr)
    return _solve(problem, atoms, normalized[atoms], (), horizon)
