#  Copyright (c) 2026. Robin Thibaut, Ghent University

"""From a public BEL posterior to an adaptive decision, on a finite software model.

One runnable chain, every step through public SKBEL API:

1. a deterministic sixteen-row paired fixture (all four sign bits ``u, v, e, f``),
   ``X = (u, v)``, ``Y = (2u + e, 3v + 2f)``, fitted ONCE with
   ``StandardScaler`` pipelines, ``CCA(n_components=2)`` and ``BEL(mode='mvn')``;
2. two explicit current observations, predicted in ONE call with
   ``return_samples=False``; ``posterior_mean`` and ``posterior_covariance`` are
   read directly (no PRNG, no sampling, no cached or private state injected);
3. each canonical Gaussian is replaced by four equal-mass points
   ``mu + L z`` (``L = cholesky(Sigma)``, ``z`` the four sign pairs) which are
   back-transformed with ``BEL.inverse_transform`` into original target units;
4. a caller-supplied finite model: fixed action centres in ORIGINAL target
   units, squared Euclidean loss of every back-transformed atom to every centre,
   a joint categorical likelihood over the binary outcomes of candidates A, B, C
   and candidate costs in the same loss units;
5. ``finite_acquisition_policy`` (horizon 2) decides whether to stop or acquire,
   and the realized branch ends in a terminal action;
6. ``energy_score`` evaluates the same four-atom law against an optional,
   caller-supplied retrospective truth. It is NOT the decision loss and
   changing the truth never changes the policy.

Qualifications, all deliberate:

- The four-point law matches the canonical posterior mean and covariance only;
  it is not the Gaussian, not its tails and not a calibrated posterior.
- ``noise`` is the existing multiplier of the projected predictor covariance,
  not a physical measurement standard deviation.
- The likelihood, loss, centres and costs are assumed example inputs: they are
  not learned here, not posterior-predictive evidence and not a physical utility.
- A sign-tag 0/1 control (below) shows the known adaptive-versus-fixed
  rational values of the finite-acquisition docs. It never depends on atom
  values and is not a BEL-driven gain.

Run with ``python examples/bel_inference_to_decision.py``. Importing this module has no
side effect; ``main`` prints and writes no file.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skbel import BEL
from skbel.metrics import (
    AcquisitionNode,
    bayes_action_set,
    energy_score,
    expected_action_losses,
    finite_acquisition_policy,
)

__all__ = [
    "ACTION_CENTRES",
    "CANDIDATES",
    "CANDIDATE_COSTS",
    "NOISE",
    "OBSERVATIONS",
    "OUTCOME_TUPLES",
    "SIGN_PAIRS",
    "SIGN_TAG_COSTS",
    "SYMMETRY_TOLERANCE",
    "CaseDecision",
    "JointDecisionModel",
    "PathStep",
    "PosteriorAtoms",
    "RealizedPath",
    "action_losses",
    "canonical_cholesky",
    "decide_case",
    "expand_joint_model",
    "fit_bel",
    "follow_path",
    "main",
    "posterior_atoms",
    "predict_posterior",
    "retrospective_score",
    "run_policy",
    "sign_label_likelihood",
    "sign_tag_losses",
    "solve_policy",
    "training_design",
    "validate_likelihood",
]

# Two explicit current observations of the predictor (u, v) and the unchanged
# projected-covariance multiplier.
OBSERVATIONS = np.array([[0.0, 0.0], [0.25, -0.25]])
NOISE = 0.125

# Caller's terminal actions: fixed centres in ORIGINAL target units. The decision
# loss of an atom is its squared Euclidean distance to the chosen centre.
ACTION_CENTRES = np.array([[-1.0, 0.0], [1.0, 0.0]])

# Candidate measurements A, B, C: binary outcomes, costs in decision-loss units.
CANDIDATES = ("A", "B", "C")
CANDIDATE_COSTS = (1.0, 1.0, 1.0)
SIGN_TAG_COSTS = (1 / 50, 1 / 50, 1 / 50)

# The four sign pairs z of the equal-mass finite posterior law.
SIGN_PAIRS = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
# The eight outcome tuples (A, B, C), row t -> bits of t in lexicographic order.
OUTCOME_TUPLES = np.array(list(itertools.product((0, 1), repeat=3)), dtype=np.int64)

# Largest accepted |Sigma - Sigma.T|. Nothing is clipped, jittered or symmetrized;
# the Cholesky factor reads the lower triangle only.
SYMMETRY_TOLERANCE = 1e-12

for _array in (OBSERVATIONS, ACTION_CENTRES, SIGN_PAIRS, OUTCOME_TUPLES):
    _array.setflags(write=False)


@dataclass(frozen=True)
class PosteriorAtoms:
    """Finite four-point posterior law per observation.

    :param mean: canonical posterior means, ``(cases, 2)``.
    :param covariance: canonical posterior covariances, ``(cases, 2, 2)``.
    :param canonical_atoms: ``mu + L z`` per case, ``(cases, 4, 2)``.
    :param atoms: the same points back-transformed to original target units.
    :param labels: sign tag of the first canonical coordinate of each atom (0 or 1).
    """

    mean: np.ndarray
    covariance: np.ndarray
    canonical_atoms: np.ndarray
    atoms: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class JointDecisionModel:
    """Complete finite joint model: one row per (atom, outcome tuple) pair.

    :param weights: ``(1 / atoms) * likelihood[atom, tuple]``; zero rows are impossible.
    :param losses: ``(rows, actions)`` loss of each action if the row is the truth.
    :param outcomes: ``(rows, candidates)`` integer outcome of each candidate per row.
    :param costs: ``(candidates,)`` acquisition costs in loss units.
    :param atom_index: source atom of every row.
    :param tuple_index: source outcome tuple of every row.
    """

    weights: np.ndarray
    losses: np.ndarray
    outcomes: np.ndarray
    costs: np.ndarray
    atom_index: np.ndarray
    tuple_index: np.ndarray


@dataclass(frozen=True)
class CaseDecision:
    """Decision products of one observation, before any retrospective scoring."""

    model: JointDecisionModel
    root: AcquisitionNode
    root_expected_losses: np.ndarray
    root_bayes_risk: float
    root_bayes_actions: tuple[int, ...]


@dataclass(frozen=True)
class PathStep:
    """One realized acquisition: what the node preferred, what was observed."""

    candidate: int
    outcome: int
    probability: float
    immediate_cost: float
    choices_before: tuple[int | str, ...]
    value_before: float


@dataclass(frozen=True)
class RealizedPath:
    """An actual path through the policy tree and its terminal action.

    ``terminal_actions`` is the whole exact tie set of the final node;
    ``selected_action`` is its lowest index (a disclosed execution tie-break).
    """

    steps: tuple[PathStep, ...]
    final_node: AcquisitionNode
    total_cost: float
    expected_losses: np.ndarray
    terminal_actions: tuple[int, ...]
    selected_action: int


def training_design() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic sixteen paired rows from all sign bits ``u, v, e, f``."""
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=4)))
    u, v, e, f = signs.T
    predictor = np.column_stack([u, v])
    target = np.column_stack([2.0 * u + e, 3.0 * v + 2.0 * f])
    return predictor, target


def fit_bel(predictor: np.ndarray | None = None, target: np.ndarray | None = None) -> BEL:
    """Fit the public ``BEL(mode='mvn')`` pipeline once (no random state is used)."""
    if predictor is None or target is None:
        predictor, target = training_design()
    bel = BEL(
        mode="mvn",
        X_pre_processing=Pipeline([("scaler", StandardScaler())]),
        Y_pre_processing=Pipeline([("scaler", StandardScaler())]),
        regression_model=CCA(n_components=2),
    )
    return bel.fit(predictor, target)


def predict_posterior(
    bel: BEL, observations: np.ndarray, noise: float
) -> tuple[np.ndarray, np.ndarray]:
    """Canonical posterior mean/covariance of every observation, without sampling."""
    observed = np.array(observations, dtype=float)
    bel.predict(X_obs=observed, noise=noise, return_samples=False)
    return np.array(bel.posterior_mean, copy=True), np.array(bel.posterior_covariance, copy=True)


def canonical_cholesky(covariance: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor of a finite, symmetric, positive-definite covariance.

    Raises ``ValueError`` for non-finite, asymmetric (beyond ``SYMMETRY_TOLERANCE``)
    or not-positive-definite input; there is no clipping, jitter or symmetrization.
    """
    cov = np.array(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"covariance must be square, got shape {cov.shape}")
    if not np.all(np.isfinite(cov)):
        raise ValueError("covariance contains non-finite values")
    if np.max(np.abs(cov - cov.T)) > SYMMETRY_TOLERANCE:
        raise ValueError(f"covariance is asymmetric beyond {SYMMETRY_TOLERANCE}")
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance is not positive definite") from exc


def posterior_atoms(
    bel: BEL, posterior_mean: np.ndarray, posterior_covariance: np.ndarray
) -> PosteriorAtoms:
    """Four equal-mass points ``mu + L z`` per case, back-transformed publicly.

    The points reproduce the canonical mean and covariance exactly (up to rounding)
    and nothing else about the Gaussian.
    """
    mean = np.array(posterior_mean, dtype=float)
    cov = np.array(posterior_covariance, dtype=float)
    n_dim = SIGN_PAIRS.shape[1]
    if mean.ndim != 2 or mean.shape[1] != n_dim:
        raise ValueError(f"posterior mean must have shape (cases, {n_dim}), got {mean.shape}")
    if cov.shape != (mean.shape[0], n_dim, n_dim):
        raise ValueError(f"posterior covariance must have shape {(mean.shape[0], n_dim, n_dim)}")
    if not np.all(np.isfinite(mean)):
        raise ValueError("posterior mean contains non-finite values")
    canonical = np.empty((mean.shape[0], SIGN_PAIRS.shape[0], n_dim))
    for case in range(mean.shape[0]):
        factor = canonical_cholesky(cov[case])
        canonical[case] = mean[case] + SIGN_PAIRS @ factor.T
    atoms = bel.inverse_transform(canonical)
    if atoms.shape != canonical.shape or not np.all(np.isfinite(atoms)):
        raise ValueError("back-transformed atoms are not a finite array of the expected shape")
    labels = (SIGN_PAIRS[:, 0] > 0).astype(np.int64)
    return PosteriorAtoms(mean, cov, canonical, atoms, labels)


def hidden_state(tuples: np.ndarray) -> np.ndarray:
    """Toy hidden state ``S = B if A == 0 else C`` of outcome tuples ``(A, B, C)``."""
    tuples = np.asarray(tuples)
    return np.where(tuples[:, 0] == 0, tuples[:, 1], tuples[:, 2])


def sign_label_likelihood(labels: np.ndarray) -> np.ndarray:
    """Example caller likelihood ``(atoms, 8)`` over the outcome tuples.

    Each of the four tuples consistent with an atom's label ``S`` has conditional
    probability 1/4 and the others 0. It is a supplied joint model, not learned,
    and not a product of marginals. The second target coordinate plays no role.
    """
    states = hidden_state(OUTCOME_TUPLES)
    labels = np.asarray(labels)
    return np.where(states[None, :] == labels[:, None], 0.25, 0.0)


def validate_likelihood(likelihood: np.ndarray, n_atoms: int) -> np.ndarray:
    """Return a float copy of a finite, non-negative, exactly normalized likelihood."""
    arr = np.asarray(likelihood)
    if arr.dtype.kind not in "fiu":
        raise TypeError(f"likelihood must be numeric, got dtype {arr.dtype}")
    arr = arr.astype(np.float64)
    expected = (n_atoms, OUTCOME_TUPLES.shape[0])
    if arr.shape != expected:
        raise ValueError(f"likelihood must have shape {expected}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("likelihood contains non-finite values")
    if np.any(arr < 0):
        raise ValueError("likelihood must be non-negative")
    for row in arr:
        if math.fsum(row) != 1.0:
            raise ValueError("every likelihood row must be exactly normalized (fsum == 1.0)")
    return arr


def action_losses(atoms: np.ndarray, centres: np.ndarray = ACTION_CENTRES) -> np.ndarray:
    """Squared Euclidean loss ``|atom - centre|^2``, ``(atoms, actions)``, original units."""
    atoms = np.array(atoms, dtype=float)
    centres = np.array(centres, dtype=float)
    if atoms.ndim != 2 or centres.ndim != 2 or atoms.shape[1] != centres.shape[1]:
        raise ValueError(f"atoms {atoms.shape} and centres {centres.shape} are not compatible")
    if not (np.all(np.isfinite(atoms)) and np.all(np.isfinite(centres))):
        raise ValueError("atoms and centres must be finite")
    diff = atoms[:, None, :] - centres[None, :, :]
    losses = np.sum(diff * diff, axis=2)
    if not np.all(np.isfinite(losses)):
        raise ValueError("a squared loss is not representable in float64")
    return losses


def sign_tag_losses(labels: np.ndarray, n_actions: int = 2) -> np.ndarray:
    """Secondary control: 0/1 loss of guessing the label; independent of atom values."""
    labels = np.asarray(labels)
    return (np.arange(n_actions)[None, :] != labels[:, None]).astype(np.float64)


def expand_joint_model(
    atom_losses: np.ndarray, likelihood: np.ndarray, costs: Sequence[float]
) -> JointDecisionModel:
    """Expand atoms x outcome tuples into the joint rows consumed by the policy.

    Row weight is ``(1 / atoms) * likelihood``. Any positive product mass lost to
    float64 underflow raises before the policy is called; the supplied losses
    and costs are kept in one common unit.
    """
    losses = np.array(atom_losses, dtype=float)
    if losses.ndim != 2 or losses.shape[0] == 0 or losses.shape[1] == 0:
        raise ValueError(f"atom losses must have shape (atoms, actions), got {losses.shape}")
    if not np.all(np.isfinite(losses)):
        raise ValueError("atom losses contain non-finite values")
    n_atoms = losses.shape[0]
    like = validate_likelihood(likelihood, n_atoms)
    cost_arr = np.array(costs, dtype=float)
    if cost_arr.shape != (OUTCOME_TUPLES.shape[1],):
        raise ValueError(
            f"costs must have shape {(OUTCOME_TUPLES.shape[1],)}, got {cost_arr.shape}"
        )
    if not np.all(np.isfinite(cost_arr)) or np.any(cost_arr < 0):
        raise ValueError("costs must be finite and non-negative")

    n_tuples = like.shape[1]
    weights = ((1.0 / n_atoms) * like).reshape(-1)
    if np.count_nonzero(weights > 0) != np.count_nonzero(like > 0):
        raise ValueError("positive joint mass was lost to float64 underflow in the product")
    atom_index = np.repeat(np.arange(n_atoms), n_tuples)
    tuple_index = np.tile(np.arange(n_tuples), n_atoms)
    return JointDecisionModel(
        weights=weights,
        losses=losses[atom_index],
        outcomes=OUTCOME_TUPLES[tuple_index].copy(),
        costs=cost_arr,
        atom_index=atom_index,
        tuple_index=tuple_index,
    )


def solve_policy(model: JointDecisionModel, horizon: int = 2) -> AcquisitionNode:
    """Public exact adaptive acquisition/stopping policy of the joint model."""
    return finite_acquisition_policy(
        model.losses,
        model.outcomes,
        weights=model.weights,
        costs=model.costs,
        horizon=horizon,
    )


def decide_case(
    atoms: np.ndarray,
    likelihood: np.ndarray,
    centres: np.ndarray = ACTION_CENTRES,
    costs: Sequence[float] = CANDIDATE_COSTS,
    horizon: int = 2,
) -> CaseDecision:
    """Decision of one observation from its original-unit atoms; no truth is involved."""
    model = expand_joint_model(action_losses(atoms, centres), likelihood, costs)
    root = solve_policy(model, horizon)
    expected = expected_action_losses(model.losses, model.weights)
    risk, actions = bayes_action_set(model.losses, model.weights)
    return CaseDecision(model, root, expected, risk, tuple(int(a) for a in actions))


def follow_path(
    model: JointDecisionModel, root: AcquisitionNode, steps: Sequence[tuple[int, int]]
) -> RealizedPath:
    """Follow explicit ``(candidate, outcome)`` steps through the policy tree.

    A candidate that is already acquired or unavailable (horizon exhausted), and an
    outcome without positive conditional mass, raise ``ValueError``; no posterior
    is invented for them.
    """
    node = root
    taken: list[PathStep] = []
    for candidate, outcome in steps:
        evidence = node.candidate_values.get(int(candidate))
        if evidence is None:
            raise ValueError(
                f"candidate {candidate} is not available at this node "
                f"(acquired so far: {node.acquired}, horizon left: {node.horizon})"
            )
        branch = evidence.branches.get(int(outcome))
        if branch is None:
            raise ValueError(
                f"outcome {outcome} of candidate {candidate} has zero conditional mass "
                f"(possible outcomes: {sorted(evidence.branches)})"
            )
        taken.append(
            PathStep(
                candidate=int(candidate),
                outcome=int(outcome),
                probability=branch.probability,
                immediate_cost=evidence.immediate_cost,
                choices_before=node.optimal_choices,
                value_before=node.value,
            )
        )
        node = branch.node
    rows = np.array(node.support, dtype=np.int64)
    expected = expected_action_losses(model.losses[rows], model.weights[rows])
    return RealizedPath(
        steps=tuple(taken),
        final_node=node,
        total_cost=math.fsum(step.immediate_cost for step in taken),
        expected_losses=expected,
        terminal_actions=node.terminal_actions,
        selected_action=node.terminal_actions[0],
    )


def run_policy(
    model: JointDecisionModel, root: AcquisitionNode, realized: Mapping[int, int]
) -> RealizedPath:
    """Execute the policy against realized outcomes ``{candidate: outcome}``.

    At each node the first optimal choice is taken (``"stop"`` first on an exact
    stop tie; the whole tie set stays on every ``PathStep``). A needed outcome that
    is not supplied, or an impossible one, raises ``ValueError``.
    """
    node = root
    steps: list[tuple[int, int]] = []
    while True:
        choice = node.optimal_choices[0]
        if isinstance(choice, str):
            break
        if choice not in realized:
            raise ValueError(f"no realized outcome supplied for candidate {choice}")
        outcome = int(realized[choice])
        branch = node.candidate_values[choice].branches.get(outcome)
        if branch is None:
            raise ValueError(f"outcome {outcome} of candidate {choice} has zero conditional mass")
        steps.append((choice, outcome))
        node = branch.node
    return follow_path(model, root, steps)


def retrospective_score(atoms: np.ndarray, truth: np.ndarray) -> float:
    """Energy score of the four-atom empirical law against a caller-supplied truth.

    It scores that finite law in original target units. It is not the decision
    loss, it never enters the policy and it is not a calibration claim.
    """
    atoms = np.asarray(atoms, dtype=float)
    return float(energy_score(atoms[None], np.asarray(truth, dtype=float)[None])[0])


def _names(choices: Sequence[int | str]) -> tuple[str, ...]:
    return tuple(c if isinstance(c, str) else CANDIDATES[c] for c in choices)


def _fmt(array: np.ndarray) -> str:
    return np.array2string(np.asarray(array), precision=4, suppress_small=True)


def _print_node(title: str, node: AcquisitionNode) -> None:
    print(f"  {title}")
    print(f"    stop risk {node.stop_risk:.6f}, terminal Bayes actions {node.terminal_actions}")
    print(f"    optimal choices {_names(node.optimal_choices)}, value {node.value:.6f}")
    for candidate, evidence in node.candidate_values.items():
        print(
            f"    acquire {CANDIDATES[candidate]}: value {evidence.value:.6f} = cost "
            f"{evidence.immediate_cost:.6f} + continuation "
            f"{evidence.expected_continuation_value:.6f}"
        )
        for outcome, branch in evidence.branches.items():
            print(
                f"      outcome {outcome}: p={branch.probability:.6f}, child value "
                f"{branch.node.value:.6f}, choices {_names(branch.node.optimal_choices)}"
            )


def _print_path(label: str, path: RealizedPath) -> None:
    trail = " -> ".join(
        f"{CANDIDATES[s.candidate]}={s.outcome} (p={s.probability:.3f})" for s in path.steps
    )
    print(f"    {label}: {trail or 'stop immediately'}")
    print(f"      cost paid {path.total_cost:.6f}, support rows {len(path.final_node.support)}")
    print(f"      conditional expected losses {_fmt(path.expected_losses)}")
    print(
        f"      terminal tie set {path.terminal_actions} -> selected action "
        f"{path.selected_action} at centre {_fmt(ACTION_CENTRES[path.selected_action])}"
    )


def main() -> None:
    """Fit once, predict once, decide per observation and print everything."""
    predictor, target = training_design()
    bel = fit_bel(predictor, target)
    mean, cov = predict_posterior(bel, OBSERVATIONS, NOISE)
    post = posterior_atoms(bel, mean, cov)
    likelihood = sign_label_likelihood(post.labels)
    truths = np.array([[1.0, 1.0], [3.0, -1.0]])  # caller's retrospective truths

    print("BEL inference-to-decision on a finite software model")
    print("- finite law: four equal-mass points matching the canonical mean/covariance only;")
    print("  not the Gaussian, not its tails, not calibrated.")
    print(f"- noise {NOISE} multiplies the projected predictor covariance (not a physical SD).")
    print("- likelihood, centres, losses and costs are assumed example inputs, not learned.")
    print(f"- action centres (original units): {_fmt(ACTION_CENTRES)}; costs {CANDIDATE_COSTS}")

    decisions = []
    for case, observation in enumerate(OBSERVATIONS):
        decision = decide_case(post.atoms[case], likelihood)
        decisions.append(decision)
        print(f"\nObservation {case}: x = {_fmt(observation)}")
        print(f"  canonical posterior mean {_fmt(post.mean[case])}")
        print(f"  canonical posterior covariance {_fmt(post.covariance[case])}")
        print(f"  canonical atoms {_fmt(post.canonical_atoms[case])}")
        print(f"  original-unit atoms (labels {_fmt(post.labels)}) {_fmt(post.atoms[case])}")
        print(f"  root expected squared losses {_fmt(decision.root_expected_losses)}")
        print(f"  root public Bayes actions {decision.root_bayes_actions}")
        _print_node("MAIN decision: losses from actual original-unit atoms", decision.root)

    print("\nRealized paths (caller-supplied outcome tuples (A, B, C))")
    for case, decision in enumerate(decisions):
        print(f"  observation {case}")
        for realized in ((0, 1, 0), (1, 0, 0)):
            path = run_policy(decision.model, decision.root, dict(enumerate(realized)))
            _print_path(f"world {realized}", path)

    print("\nObservation-to-decision change (same likelihood, costs and centres)")
    for case, decision in enumerate(decisions):
        print(
            f"  observation {case}: first choice {_names(decision.root.optimal_choices)}, "
            f"expected losses {_fmt(decision.root_expected_losses)}"
        )

    print("\nSecondary control: sign-tag 0/1 loss, independent of atom values")
    control_model = expand_joint_model(sign_tag_losses(post.labels), likelihood, SIGN_TAG_COSTS)
    control = solve_policy(control_model)
    print(f"  adaptive value {control.value:.6f} (hand value 1/25 = {1 / 25:.6f})")
    print("  best fixed subset of at most two measurements: 27/100 (hand value, see docs)")
    print(f"  first choice {_names(control.optimal_choices)}")
    for outcome, branch in control.candidate_values[0].branches.items():
        print(f"  after A={outcome}: choices {_names(branch.node.optimal_choices)}")
    print("  known software oracle only, not a BEL-driven gain or an observed superiority.")

    print("\nRetrospective energy score (separate from the policy; not decision loss)")
    for case in range(len(OBSERVATIONS)):
        score = retrospective_score(post.atoms[case], truths[case])
        print(f"  observation {case}, truth {_fmt(truths[case])}: {score:.6f}")


if __name__ == "__main__":
    main()
