#  Copyright (c) 2026. Robin Thibaut, Ghent University

"""Model-robust finite acquisition and stopping with a whole-tree commitment.

The caller supplies one *common world* of ``N`` identified atoms: ``losses[atom,
action]`` is the terminal loss, ``outcomes[atom, candidate]`` the integer label
each candidate would return, and ``costs[candidate]`` its price, all in common
units. ``M`` caller models each place a normalized probability law
``model_weights[m]`` on that same atom axis. Nothing is simulated, fitted or
calibrated, and no independence between measurements is assumed.

A *policy* is a precommitted deterministic decision tree for horizon 0, 1 or 2.
It acquires candidates without replacement, observes only the acquired
outcomes, and ends every path in a terminal action. A policy is complete over
the *union-reachable* histories: every history with positive mass under at
least one model has a branch, even when another model gives it zero mass.

The objective of a policy is evaluated per model, with that model fixed for the
entire tree::

    risk_m(policy) = sum_atoms model_weights[m, atom]
                     * (cost of candidates acquired on the atom's path
                        + loss of the atom's terminal action)

and the policy minimizes ``max_m risk_m``. No conditional probability is ever
formed, so a history impossible under one model contributes exactly zero to that
model's risk and no posterior is invented for it.

This is *not* a branch-switching adversary: the worst model is chosen once, at
the root, for the whole tree. It is also not a regret objective, not a
randomized-policy optimum, and not a global minimax statement over all policy
classes. Randomized trees can do strictly better and are not searched. Child
subtrees are never re-optimized or recombined; returned minimizers are complete
trees, and branches of different tied minimizers must not be spliced.

Because a deterministic tree may act on any outcome, a candidate whose outcome is
independent of every model's loss-relevant state can still lower the worst-case
risk by acting as a paid random coin. That gain is randomization, not
information; see ``docs/finite_model_robustness.md``.

Tie semantics: all arithmetic is IEEE float64. A policy's per-model risk is the
exactly rounded sum (``math.fsum``) of the float products ``weight * total``;
minimizers are located with exact ``==`` and no tolerance, so only values equal
in float64 tie. Policies are returned in a canonical order: at every node the
terminal actions in ascending order come first, then each unacquired candidate
in ascending order; the children of a candidate follow ascending outcome label
and are combined with the first label varying slowest.

Supported size, enforced before any policy is generated: at most 8 models, 32
atoms, 3 candidates, 3 actions, 2 outcome labels per candidate over the
positive-mass atoms, horizon 2. That is at most 1326 complete policies, 31
history nodes and 1524 generated sub-policies per call; larger inputs raise
``ValueError`` and are never pruned. These are acceptance bounds, not a runtime
guarantee.
"""

from __future__ import annotations

import itertools
import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from .sequential import (
    _normalize_weights,
    _validate_costs,
    _validate_horizon,
    _validate_losses,
    _validate_outcomes,
)

__all__ = [
    "BaselineResult",
    "PolicyNode",
    "RobustPolicy",
    "RobustResult",
    "WorkCounts",
    "robust_acquisition_policy",
]

_MAX_MODELS = 8
_MAX_ATOMS = 32
_MAX_CANDIDATES = 3
_MAX_ACTIONS = 3
_MAX_LABELS = 2
_MAX_HORIZON = 2


def _policy_ceiling(candidates: int, actions: int, labels: int, horizon: int) -> int:
    """Largest number of complete policies at one node."""
    if horizon == 0 or candidates == 0:
        return actions
    child = _policy_ceiling(candidates - 1, actions, labels, horizon - 1)
    return actions + candidates * child**labels


def _node_ceiling(candidates: int, labels: int, horizon: int) -> int:
    """Largest number of history nodes in one subtree."""
    if horizon == 0 or candidates == 0:
        return 1
    return 1 + candidates * labels * _node_ceiling(candidates - 1, labels, horizon - 1)


def _generated_ceiling(candidates: int, actions: int, labels: int, horizon: int) -> int:
    """Largest number of sub-policies generated over all nodes of one subtree."""
    own = _policy_ceiling(candidates, actions, labels, horizon)
    if horizon == 0 or candidates == 0:
        return own
    child = _generated_ceiling(candidates - 1, actions, labels, horizon - 1)
    return own + candidates * labels * child


_MAX_POLICIES = _policy_ceiling(_MAX_CANDIDATES, _MAX_ACTIONS, _MAX_LABELS, _MAX_HORIZON)
_MAX_NODES = _node_ceiling(_MAX_CANDIDATES, _MAX_LABELS, _MAX_HORIZON)
_MAX_GENERATED = _generated_ceiling(_MAX_CANDIDATES, _MAX_ACTIONS, _MAX_LABELS, _MAX_HORIZON)

_NO_ACQUISITION = frozenset()


@dataclass(frozen=True)
class PolicyNode:
    """One node of a complete deterministic policy tree.

    :param kind: ``"stop"`` (terminal action) or ``"acquire"`` (measure a candidate).
    :param action: terminal action index for ``"stop"``, else ``None``.
    :param candidate: acquired candidate index for ``"acquire"``, else ``None``.
    :param branches: read-only map from integer outcome label to the child node;
        every label with positive mass under at least one model, none otherwise.
    :param acquired: candidates acquired on the path to this node, in order.
    :param history: ``(candidate, label)`` observed on the path to this node.
    :param support: sorted original atom indices that are consistent with the
        history and carry positive mass under at least one model.
    """

    kind: str
    action: int | None
    candidate: int | None
    branches: Mapping[int, PolicyNode]
    acquired: tuple[int, ...]
    history: tuple[tuple[int, int], ...]
    support: tuple[int, ...]


@dataclass(frozen=True)
class RobustPolicy:
    """A complete policy with its per-model risks.

    :param index: position in the canonical order (row of ``risk_table``).
    :param root: the complete decision tree.
    :param model_risks: expected total loss plus acquisition cost under each model.
    :param worst_risk: ``max(model_risks)``.
    :param worst_models: every model whose risk equals ``worst_risk`` exactly.
    """

    index: int
    root: PolicyNode
    model_risks: tuple[float, ...]
    worst_risk: float
    worst_models: tuple[int, ...]


@dataclass(frozen=True)
class BaselineResult:
    """A comparison policy chosen without knowing the true model.

    ``optimal_indices`` holds every complete policy that attains the baseline's
    own objective exactly, including every tie and every completion of histories
    that have zero mass under the baseline's law. ``selected_indices`` keeps those
    optima with the smallest worst-model risk, which is the strongest baseline.

    :param name: ``"nominal[m]"``, ``"mixture"``, ``"fixed_subset"`` or ``"stop_only"``.
    :param objective: the baseline's own optimum (model-``m`` risk, mixture risk,
        or the worst-model risk for the two common-information classes).
    :param optimal_indices: all policies attaining ``objective``, canonical order.
    :param selected_indices: the optima with the least worst-model risk.
    :param worst_risk: worst-model risk of the selected policies.
    :param model_risks: per-model risks of the first selected policy.
    """

    name: str
    objective: float
    optimal_indices: tuple[int, ...]
    selected_indices: tuple[int, ...]
    worst_risk: float
    model_risks: tuple[float, ...]


@dataclass(frozen=True)
class WorkCounts:
    """Exact work performed by one call.

    :param policies: complete policies generated and evaluated.
    :param model_evaluations: ``policies * models`` per-model risk evaluations.
    :param nodes: history nodes visited while generating policies.
    :param generated: sub-policies generated over all nodes (root included).
    :param atoms: positive-mass atoms in the policy world.
    :param models: number of models.
    """

    policies: int
    model_evaluations: int
    nodes: int
    generated: int
    atoms: int
    models: int


@dataclass(frozen=True, eq=False)
class RobustResult:
    """All global minimizers, strongest baselines and the whole risk table.

    :param robust: every complete policy attaining the minimum worst-model risk,
        in canonical order (exact float64 ties).
    :param worst_risk: that minimum.
    :param policy_count: number of complete policies enumerated.
    :param risk_table: ``(policies, models)`` array of per-model risks, a
        non-writable view of immutable bytes whose write flag cannot be re-enabled.
    :param nominal: one baseline per model: its Bayes optima, completed over the
        union-reachable histories, selected for the best worst-model risk.
    :param mixture: the same for the caller's explicit model probabilities, or
        ``None`` when ``mixture_weights`` was not supplied (no hidden prior).
    :param fixed_subset: best worst-model risk over policies that acquire the
        same set of candidates on every path, with terminal actions common to all
        models; stop-only policies are included.
    :param stop_only: best worst-model risk over a single common terminal action.
    :param atom_ids: caller atom identifiers, in input order.
    :param support_ids: identifiers of atoms with positive mass under some model.
    :param dropped_ids: identifiers of atoms with zero mass under every model.
    :param horizon: the horizon used.
    :param work: exact work counts.

    ``policy(index)`` returns the complete tree of any enumerated policy.
    """

    robust: tuple[RobustPolicy, ...]
    worst_risk: float
    policy_count: int
    risk_table: np.ndarray
    nominal: tuple[BaselineResult, ...]
    mixture: BaselineResult | None
    fixed_subset: BaselineResult
    stop_only: BaselineResult
    atom_ids: tuple[str, ...]
    support_ids: tuple[str, ...]
    dropped_ids: tuple[str, ...]
    horizon: int
    work: WorkCounts
    _trees: tuple = field(repr=False)
    _labels: tuple = field(repr=False)
    _union: tuple = field(repr=False)

    def policy(self, index: int) -> PolicyNode:
        """The complete tree of policy ``index`` (a fresh immutable object)."""
        position = operator.index(index)
        if not 0 <= position < len(self._trees):
            raise IndexError(f"policy index {position} out of range 0..{len(self._trees) - 1}")
        return _node(
            self._trees[position], self._labels, self._union, tuple(range(len(self._union))), (), ()
        )


@dataclass(frozen=True)
class _Space:
    loss: list
    labels: list
    costs: list
    actions: int
    candidates: int


def _check_cap(name: str, value: int, cap: int) -> None:
    if value > cap:
        raise ValueError(f"{name} = {value} exceeds the supported maximum {cap}")


def _split(labels, support: tuple[int, ...], candidate: int) -> list:
    """Outcome groups of ``support``: ``(label, positions within support)``, ascending label."""
    groups: dict[int, list[int]] = {}
    for position, atom in enumerate(support):
        groups.setdefault(labels[atom][candidate], []).append(position)
    return [(label, tuple(groups[label])) for label in sorted(groups)]


def _count(
    space: _Space, support: tuple[int, ...], acquired: tuple[int, ...], horizon: int, tally: list
) -> int:
    """Policies below one node, counted without allocating them; enforces the caps."""
    tally[0] += 1
    if tally[0] > _MAX_NODES:
        raise ValueError(f"policy tree exceeds the supported {_MAX_NODES} history nodes")
    policies = space.actions
    if horizon > 0:
        for candidate in range(space.candidates):
            if candidate in acquired:
                continue
            product = 1
            for _, positions in _split(space.labels, support, candidate):
                child = tuple(support[p] for p in positions)
                product *= _count(space, child, (*acquired, candidate), horizon - 1, tally)
                if product > _MAX_POLICIES:
                    raise ValueError(f"more than {_MAX_POLICIES} complete policies")
            policies += product
            if policies > _MAX_POLICIES:
                raise ValueError(f"more than {_MAX_POLICIES} complete policies")
    tally[1] += policies
    if tally[1] > _MAX_GENERATED:
        raise ValueError(f"more than {_MAX_GENERATED} generated sub-policies")
    return policies


def _enumerate(space: _Space, support: tuple[int, ...], acquired: tuple[int, ...], horizon: int):
    """Every complete policy for the atoms of ``support``.

    Each entry is ``(tree, vector, signature)``: the nested-tuple tree, the total
    path cost plus loss of every atom of ``support`` (in ``support`` order), and
    the set of candidates acquired on every leaf if that set is the same on all
    leaves, else ``None``.
    """
    entries = []
    for action in range(space.actions):
        vector = tuple(space.loss[atom][action] for atom in support)
        entries.append((("stop", action), vector, _NO_ACQUISITION))
    if horizon == 0:
        return entries
    for candidate in range(space.candidates):
        if candidate in acquired:
            continue
        groups = _split(space.labels, support, candidate)
        children = [
            _enumerate(
                space, tuple(support[p] for p in positions), (*acquired, candidate), horizon - 1
            )
            for _, positions in groups
        ]
        cost = space.costs[candidate]
        for combo in itertools.product(*children):
            vector = [0.0] * len(support)
            for (_, positions), entry in zip(groups, combo, strict=True):
                for position, value in zip(positions, entry[1], strict=True):
                    total = cost + value
                    if not math.isfinite(total):
                        raise ValueError(
                            "a policy's total loss and cost is not finite (cost or loss overflow)"
                        )
                    vector[position] = total
            subtrees = tuple(
                (label, entry[0]) for (label, _), entry in zip(groups, combo, strict=True)
            )
            shared = {entry[2] for entry in combo}
            if len(shared) == 1 and None not in shared:
                signature = next(iter(shared)) | {candidate}
            else:
                signature = None
            entries.append((("acquire", candidate, subtrees), tuple(vector), signature))
    return entries


def _node(tree, labels, union, support, acquired, history) -> PolicyNode:
    original = tuple(union[p] for p in support)
    if tree[0] == "stop":
        return PolicyNode(
            kind="stop",
            action=tree[1],
            candidate=None,
            branches=MappingProxyType({}),
            acquired=acquired,
            history=history,
            support=original,
        )
    _, candidate, subtrees = tree
    groups = dict(_split(labels, support, candidate))
    branches = {
        label: _node(
            subtree,
            labels,
            union,
            tuple(support[p] for p in groups[label]),
            (*acquired, candidate),
            (*history, (candidate, label)),
        )
        for label, subtree in subtrees
    }
    return PolicyNode(
        kind="acquire",
        action=None,
        candidate=candidate,
        branches=MappingProxyType(branches),
        acquired=acquired,
        history=history,
        support=original,
    )


def _finite_sum(terms) -> float:
    try:
        total = math.fsum(terms)
    except (OverflowError, ValueError) as exc:
        raise ValueError("an expected total loss is not finite") from exc
    if not math.isfinite(total):
        raise ValueError("an expected total loss is not finite")
    return total


def _argmin(indices, values) -> tuple[float, tuple[int, ...]]:
    best = min(values[i] for i in indices)
    return best, tuple(i for i in indices if values[i] == best)


def _baseline(name: str, pool, objective, worst, table) -> BaselineResult:
    best, optimal = _argmin(pool, objective)
    floor, selected = _argmin(optimal, worst)
    return BaselineResult(
        name=name,
        objective=best,
        optimal_indices=optimal,
        selected_indices=selected,
        worst_risk=floor,
        model_risks=tuple(table[selected[0]]),
    )


def _validate_model_weights(model_weights, atoms: int) -> np.ndarray:
    arr = np.asarray(model_weights)
    if arr.dtype.kind not in "fiu":
        raise TypeError(f"model_weights must be a numeric array, got dtype {arr.dtype}")
    if arr.ndim != 2:
        raise ValueError(f"model_weights must have shape (models, atoms), got ndim={arr.ndim}")
    if arr.shape[0] == 0:
        raise ValueError("model_weights must contain at least one model")
    if arr.shape[1] != atoms:
        raise ValueError(f"model_weights must have {atoms} columns (one per atom), got {arr.shape}")
    _check_cap("models", arr.shape[0], _MAX_MODELS)
    return arr


def _validate_ids(atom_ids, model_atom_ids, atoms: int, models: int) -> tuple[str, ...]:
    if isinstance(atom_ids, (str, bytes)):
        raise TypeError("atom_ids must be a sequence of strings, not a single string")
    ids = tuple(atom_ids)
    if not all(isinstance(i, str) for i in ids):
        raise TypeError("every atom id must be a string")
    if len(ids) != atoms:
        raise ValueError(f"atom_ids must have {atoms} entries (one per atom), got {len(ids)}")
    if len(set(ids)) != len(ids):
        raise ValueError("atom_ids must be unique")
    if isinstance(model_atom_ids, (str, bytes)):
        raise TypeError("model_atom_ids must be a sequence of atom-id sequences")
    per_model = tuple(model_atom_ids)
    if len(per_model) != models:
        raise ValueError(f"model_atom_ids must have {models} entries (one per model)")
    for m, row in enumerate(per_model):
        if isinstance(row, (str, bytes)):
            raise TypeError(f"model_atom_ids[{m}] must be a sequence of atom ids")
        if tuple(row) != ids:
            raise ValueError(f"model_atom_ids[{m}] does not match atom_ids in content and order")
    return ids


def robust_acquisition_policy(
    losses,
    outcomes,
    model_weights,
    atom_ids,
    model_atom_ids,
    *,
    costs=None,
    horizon: int = 2,
    mixture_weights=None,
) -> RobustResult:
    """Minimize the worst-model expected total loss over complete deterministic policies.

    :param losses: shape ``(atoms, actions)``, finite; loss of each terminal action
        if the atom is the truth, in the caller's common units.
    :param outcomes: shape ``(atoms, candidates)``, integer labels each candidate
        returns for each atom (bool and float arrays are rejected).
    :param model_weights: shape ``(models, atoms)``; each row finite, non-negative
        with a positive total, normalized here. Each row is a joint law on the
        same atoms. A zero weight is impossible support under that model.
    :param atom_ids: one distinct string per atom.
    :param model_atom_ids: one sequence of atom ids per model; each must equal
        ``atom_ids`` in content and order. Matching ids cannot certify that the
        atoms are physically aligned or in the same units; the caller owns that.
    :param costs: optional shape ``(candidates,)``, finite and non-negative, in the
        same units as ``losses``; zero if ``None``.
    :param horizon: exactly the integer 0, 1 or 2.
    :param mixture_weights: optional shape ``(models,)`` explicit model
        probabilities for the mixture baseline; no uniform prior is assumed.
    :return: a :class:`RobustResult`. Inputs are copied, never modified; outputs
        are immutable.

    Every input row is validated before atoms with zero mass under every model are
    dropped, so an invalid loss or label on a dropped atom still raises. A positive
    weight that normalizes to zero in float64 raises ``ValueError`` and is never
    repaired. Sizes beyond the module limits and values that overflow float64 raise
    ``ValueError``; nothing is pruned.
    """
    horizon = _validate_horizon(horizon)
    loss_arr = _validate_losses(losses)
    n_atoms, n_actions = loss_arr.shape
    _check_cap("atoms", n_atoms, _MAX_ATOMS)
    _check_cap("actions", n_actions, _MAX_ACTIONS)
    outcome_arr = _validate_outcomes(outcomes, n_atoms)
    n_candidates = outcome_arr.shape[1]
    _check_cap("candidates", n_candidates, _MAX_CANDIDATES)
    cost_arr = _validate_costs(costs, n_candidates)
    weight_arr = _validate_model_weights(model_weights, n_atoms)
    n_models = weight_arr.shape[0]
    ids = _validate_ids(atom_ids, model_atom_ids, n_atoms, n_models)

    normalized = np.empty((n_models, n_atoms), dtype=np.float64)
    for m in range(n_models):
        try:
            normalized[m] = _normalize_weights(weight_arr[m], n_atoms)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"model_weights[{m}]: {exc}") from exc
    mixture = None
    if mixture_weights is not None:
        try:
            mixture = [float(p) for p in _normalize_weights(mixture_weights, n_models)]
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"mixture_weights: {exc}") from exc

    # Everything above validated every atom; only now are globally impossible atoms dropped.
    union = tuple(int(i) for i in np.flatnonzero((normalized > 0).any(axis=0)))
    union_labels = outcome_arr[list(union)]
    for candidate in range(n_candidates):
        n_labels = int(np.unique(union_labels[:, candidate]).size)
        if n_labels > _MAX_LABELS:
            raise ValueError(
                f"candidate {candidate} has {n_labels} outcome labels over the positive-mass "
                f"atoms; at most {_MAX_LABELS} are supported"
            )

    space = _Space(
        loss=loss_arr[list(union)].tolist(),
        labels=union_labels.tolist(),
        costs=cost_arr.tolist(),
        actions=n_actions,
        candidates=n_candidates,
    )
    root_support = tuple(range(len(union)))
    tally = [0, 0]
    planned = _count(space, root_support, (), horizon, tally)

    entries = _enumerate(space, root_support, (), horizon)
    if len(entries) != planned:
        raise RuntimeError("internal error: generated policy count differs from the plan")

    model_rows = [normalized[m, list(union)].tolist() for m in range(n_models)]
    table = [
        tuple(_finite_sum(map(operator.mul, row, entry[1])) for row in model_rows)
        for entry in entries
    ]
    worst = [max(row) for row in table]
    everything = tuple(range(len(entries)))

    best_worst, robust_indices = _argmin(everything, worst)
    robust = tuple(
        RobustPolicy(
            index=i,
            root=_node(entries[i][0], space.labels, union, root_support, (), ()),
            model_risks=table[i],
            worst_risk=worst[i],
            worst_models=tuple(m for m, risk in enumerate(table[i]) if risk == worst[i]),
        )
        for i in robust_indices
    )

    nominal = tuple(
        _baseline(f"nominal[{m}]", everything, [row[m] for row in table], worst, table)
        for m in range(n_models)
    )
    mixture_baseline = None
    if mixture is not None:
        mixed = [_finite_sum(p * r for p, r in zip(mixture, row, strict=True)) for row in table]
        mixture_baseline = _baseline("mixture", everything, mixed, worst, table)
    fixed_pool = tuple(i for i in everything if entries[i][2] is not None)
    stop_pool = tuple(i for i in everything if entries[i][0][0] == "stop")
    fixed_subset = _baseline("fixed_subset", fixed_pool, worst, worst, table)
    stop_only = _baseline("stop_only", stop_pool, worst, worst, table)

    # A view onto immutable bytes: it owns nothing, so its write flag cannot be re-enabled.
    packed = np.array(table, dtype=np.float64).tobytes()
    risk_table = np.frombuffer(packed, dtype=np.float64).reshape(len(entries), n_models)
    return RobustResult(
        robust=robust,
        worst_risk=best_worst,
        policy_count=len(entries),
        risk_table=risk_table,
        nominal=nominal,
        mixture=mixture_baseline,
        fixed_subset=fixed_subset,
        stop_only=stop_only,
        atom_ids=ids,
        support_ids=tuple(ids[i] for i in union),
        dropped_ids=tuple(ids[i] for i in sorted(set(range(n_atoms)) - set(union))),
        horizon=horizon,
        work=WorkCounts(
            policies=len(entries),
            model_evaluations=len(entries) * n_models,
            nodes=tally[0],
            generated=tally[1],
            atoms=len(union),
            models=n_models,
        ),
        _trees=tuple(entry[0] for entry in entries),
        _labels=tuple(tuple(row) for row in space.labels),
        _union=union,
    )
