"""Unit tests for skbel.metrics.sequential (finite adaptive acquisition and stopping).

Import strategy: normal tests require the complete package import. An import
failure intentionally fails test collection rather than silently bypassing
``skbel.__init__`` with a direct file import.

Oracles here are independent of the production module: hand-computed rational
values for the eight-atom case, plus separate ``fractions.Fraction`` recursions
(adaptive policy and fixed measurement subsets) that share no helper with
``skbel.metrics.sequential``. Exact ``==`` assertions are only made on
dyadic-rational fixtures whose float64 arithmetic is exact.
"""

from __future__ import annotations

import itertools
import unittest
import warnings
from dataclasses import FrozenInstanceError
from fractions import Fraction

import numpy as np

from skbel.metrics import (
    AcquisitionBranch,
    AcquisitionNode,
    CandidateEvidence,
    finite_acquisition_policy,
)

# --------------------------------------------------------------------------
# Independent rational oracles
# --------------------------------------------------------------------------


def _oracle_candidate(losses, outcomes, weights, costs, horizon, alive, available, candidate):
    mass = sum(weights[i] for i in alive)
    groups = {}
    for i in alive:
        groups.setdefault(outcomes[i][candidate], []).append(i)
    expected = Fraction(0)
    for group in groups.values():
        share = sum(weights[i] for i in group) / mass
        child = _oracle_value(
            losses, outcomes, weights, costs, horizon - 1, group, available - {candidate}
        )
        expected += share * child
    return costs[candidate] + expected


def _oracle_value(losses, outcomes, weights, costs, horizon, alive=None, available=None):
    """Exact optimal remaining objective by exhaustive recursion over Fractions."""
    if alive is None:
        alive = [i for i, w in enumerate(weights) if w > 0]
    if available is None:
        available = frozenset(range(len(outcomes[0])))
    mass = sum(weights[i] for i in alive)
    n_actions = len(losses[0])
    best = min(sum(weights[i] * losses[i][a] for i in alive) / mass for a in range(n_actions))
    if horizon > 0:
        for c in sorted(available):
            value = _oracle_candidate(
                losses, outcomes, weights, costs, horizon, alive, available, c
            )
            best = min(best, value)
    return best


def _oracle_fixed_subset(losses, outcomes, weights, costs, subset):
    """Cost of a non-adaptive subset plus the risk of the best action per joint cell."""
    alive = [i for i, w in enumerate(weights) if w > 0]
    mass = sum(weights[i] for i in alive)
    cells = {}
    for i in alive:
        cells.setdefault(tuple(outcomes[i][c] for c in subset), []).append(i)
    n_actions = len(losses[0])
    risk = sum(
        min(sum(weights[i] * losses[i][a] for i in cell) for a in range(n_actions))
        for cell in cells.values()
    )
    return sum((costs[c] for c in subset), Fraction(0)) + risk / mass


def _oracle_best_fixed(losses, outcomes, weights, costs, max_size):
    n_candidates = len(outcomes[0])
    return min(
        _oracle_fixed_subset(losses, outcomes, weights, costs, subset)
        for size in range(max_size + 1)
        for subset in itertools.combinations(range(n_candidates), size)
    )


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _eight_atom_case():
    """Atoms (A, B, C) iid uniform bits; hidden state S = B if A == 0 else C.

    Hand values with every cost 1/50: stop 1/2; A alone 13/25 (A says nothing
    about S); B alone = C alone 27/100; any fixed pair 29/100; adaptive
    A-then-(B if A=0 else C) = 2/50 + 0 = 1/25.
    """
    atoms = list(itertools.product((0, 1), repeat=3))
    states = [b if a == 0 else c for a, b, c in atoms]
    losses = [[0 if action == s else 1 for action in (0, 1)] for s in states]
    outcomes = [list(atom) for atom in atoms]
    weights = [Fraction(1)] * 8
    costs = [Fraction(1, 50)] * 3
    return losses, outcomes, weights, costs


def _eight_atom_policy(horizon=2):
    losses, outcomes, _, _ = _eight_atom_case()
    return finite_acquisition_policy(losses, outcomes, costs=[0.02] * 3, horizon=horizon)


def _perfect_two_state(cost=0.0, horizon=2):
    """Two equiprobable states fully revealed by one candidate."""
    return finite_acquisition_policy([[0, 1], [1, 0]], [[0], [1]], costs=[cost], horizon=horizon)


def _flatten(node):
    """Comparable nested-tuple form of a policy tree (without ``support``)."""
    return (
        node.value,
        node.stop_risk,
        node.terminal_actions,
        node.optimal_choices,
        tuple(
            (
                c,
                e.value,
                e.immediate_cost,
                e.expected_continuation_value,
                tuple((o, b.probability, _flatten(b.node)) for o, b in e.branches.items()),
            )
            for c, e in node.candidate_values.items()
        ),
    )


def _audit_total(node, costs):
    """Expected total objective of executing the first optimal choice everywhere."""
    choice = node.optimal_choices[0]
    if choice == "stop":
        return node.stop_risk
    evidence = node.candidate_values[choice]
    return costs[choice] + sum(
        b.probability * _audit_total(b.node, costs) for b in evidence.branches.values()
    )


class _PolicyTestCase(unittest.TestCase):
    def assertClose(self, got, expected, places=12):
        self.assertAlmostEqual(got, float(expected), places=places)

    def assertNodeConsistent(self, node, costs, n_candidates, horizon):
        """Recursively verify the auditable decomposition at every node."""
        self.assertEqual(node.horizon, horizon)
        self.assertEqual(len(set(node.acquired)), len(node.acquired))
        legal = [c for c in range(n_candidates) if c not in node.acquired] if horizon > 0 else []
        self.assertEqual(list(node.candidate_values), legal)
        self.assertEqual(
            node.value, min([node.stop_risk, *(e.value for e in node.candidate_values.values())])
        )
        expected_choices = []
        if node.stop_risk == node.value:
            expected_choices.append("stop")
        expected_choices += [c for c, e in node.candidate_values.items() if e.value == node.value]
        self.assertEqual(list(node.optimal_choices), expected_choices)
        self.assertGreater(len(node.terminal_actions), 0)
        for c, evidence in node.candidate_values.items():
            self.assertEqual(evidence.immediate_cost, costs[c])
            self.assertEqual(
                evidence.value, evidence.immediate_cost + evidence.expected_continuation_value
            )
            probabilities = [b.probability for b in evidence.branches.values()]
            self.assertTrue(all(p > 0 for p in probabilities))
            self.assertAlmostEqual(sum(probabilities), 1.0, places=12)
            self.assertAlmostEqual(
                evidence.expected_continuation_value,
                sum(b.probability * b.node.value for b in evidence.branches.values()),
                places=12,
            )
            for branch in evidence.branches.values():
                self.assertEqual(branch.node.acquired, (*node.acquired, c))
                self.assertNodeConsistent(branch.node, costs, n_candidates, horizon - 1)


# --------------------------------------------------------------------------
# Eight-atom oracle and fixed-subset baseline
# --------------------------------------------------------------------------


class TestEightAtomOracle(_PolicyTestCase):
    def setUp(self):
        self.losses, self.outcomes, self.weights, self.costs = _eight_atom_case()
        self.policy = _eight_atom_policy()

    def test_hand_computed_root(self):
        root = self.policy
        self.assertIsInstance(root, AcquisitionNode)
        self.assertClose(root.value, Fraction(1, 25))
        self.assertEqual(root.stop_risk, 0.5)
        self.assertEqual(root.terminal_actions, (0, 1))
        self.assertEqual(root.optimal_choices, (0,))
        self.assertEqual(list(root.candidate_values), [0, 1, 2])
        self.assertClose(root.candidate_values[0].value, Fraction(1, 25))
        self.assertClose(root.candidate_values[1].value, Fraction(27, 100))
        self.assertClose(root.candidate_values[2].value, Fraction(27, 100))
        self.assertClose(root.candidate_values[0].immediate_cost, Fraction(1, 50))
        self.assertClose(root.candidate_values[0].expected_continuation_value, Fraction(1, 50))

    def test_second_choice_depends_on_first_outcome(self):
        a_branches = self.policy.candidate_values[0].branches
        self.assertEqual(list(a_branches), [0, 1])
        for label in (0, 1):
            self.assertIsInstance(a_branches[label], AcquisitionBranch)
            self.assertEqual(a_branches[label].probability, 0.5)
        after_a0 = a_branches[0].node
        after_a1 = a_branches[1].node
        self.assertEqual(after_a0.acquired, (0,))
        self.assertEqual(after_a0.optimal_choices, (1,))
        self.assertEqual(after_a1.optimal_choices, (2,))
        # A = 0 -> S = B: measuring B costs 1/50, measuring C is useless.
        self.assertEqual(after_a0.stop_risk, 0.5)
        self.assertClose(after_a0.candidate_values[1].value, Fraction(1, 50))
        self.assertClose(after_a0.candidate_values[2].value, Fraction(26, 50))
        self.assertClose(after_a1.candidate_values[2].value, Fraction(1, 50))
        self.assertClose(after_a1.candidate_values[1].value, Fraction(26, 50))

    def test_acquisition_is_without_replacement(self):
        a_branches = self.policy.candidate_values[0].branches
        for branch in a_branches.values():
            self.assertEqual(list(branch.node.candidate_values), [1, 2])
            for second in branch.node.candidate_values.values():
                for leaf in second.branches.values():
                    self.assertEqual(leaf.node.candidate_values, {})
                    self.assertEqual(leaf.node.horizon, 0)

    def test_leaves_carry_terminal_bayes_actions(self):
        b_after_a0 = self.policy.candidate_values[0].branches[0].node.candidate_values[1]
        leaf_b0 = b_after_a0.branches[0].node
        leaf_b1 = b_after_a0.branches[1].node
        self.assertEqual(leaf_b0.stop_risk, 0.0)
        self.assertEqual(leaf_b0.terminal_actions, (0,))
        self.assertEqual(leaf_b1.terminal_actions, (1,))
        self.assertEqual(leaf_b1.optimal_choices, ("stop",))

    def test_matches_independent_rational_recursion_for_each_horizon(self):
        for horizon in (0, 1, 2):
            policy = _eight_atom_policy(horizon)
            expected = _oracle_value(self.losses, self.outcomes, self.weights, self.costs, horizon)
            self.assertClose(policy.value, expected)
        oracle = _oracle_value(self.losses, self.outcomes, self.weights, self.costs, 2)
        self.assertEqual(oracle, Fraction(1, 25))

    def test_fixed_subset_baseline_is_strictly_worse(self):
        args = (self.losses, self.outcomes, self.weights, self.costs)
        expected = {
            (): Fraction(1, 2),
            (0,): Fraction(13, 25),
            (1,): Fraction(27, 100),
            (2,): Fraction(27, 100),
            (0, 1): Fraction(29, 100),
            (0, 2): Fraction(29, 100),
            (1, 2): Fraction(29, 100),
        }
        for subset, value in expected.items():
            self.assertEqual(_oracle_fixed_subset(*args, subset), value, msg=str(subset))
        best_fixed = _oracle_best_fixed(*args, 2)
        self.assertEqual(best_fixed, Fraction(27, 100))
        self.assertLess(self.policy.value, float(best_fixed))
        self.assertClose(self.policy.value, Fraction(1, 25))

    def test_one_step_horizon_ties_b_and_c_at_best_fixed_value(self):
        policy = _eight_atom_policy(1)
        self.assertClose(policy.value, Fraction(27, 100))
        self.assertEqual(policy.optimal_choices, (1, 2))
        self.assertGreater(policy.candidate_values[0].value, policy.value)

    def test_executing_policy_reproduces_value_with_each_cost_once(self):
        self.assertAlmostEqual(_audit_total(self.policy, [0.02] * 3), 1 / 25, places=12)
        self.assertNodeConsistent(self.policy, [0.02] * 3, 3, 2)

    def test_free_costs_would_change_the_answer(self):
        free = finite_acquisition_policy(self.losses, self.outcomes, horizon=2)
        self.assertEqual(free.value, 0.0)
        self.assertEqual(free.optimal_choices, (0,))

    def test_horizon_zero_is_prior_decision(self):
        policy = _eight_atom_policy(0)
        self.assertEqual(policy.value, 0.5)
        self.assertEqual(policy.stop_risk, 0.5)
        self.assertEqual(policy.optimal_choices, ("stop",))
        self.assertEqual(policy.candidate_values, {})
        self.assertEqual(policy.horizon, 0)

    def test_costs_above_gain_stop_immediately(self):
        policy = finite_acquisition_policy(self.losses, self.outcomes, costs=[0.3] * 3)
        self.assertEqual(policy.optimal_choices, ("stop",))
        self.assertEqual(policy.value, 0.5)
        self.assertGreater(min(e.value for e in policy.candidate_values.values()), 0.5)


# --------------------------------------------------------------------------
# Horizons, stopping, ties
# --------------------------------------------------------------------------


class TestStoppingAndTies(_PolicyTestCase):
    def test_zero_candidates_stops(self):
        for outcomes in (np.zeros((2, 0), dtype=np.int64), [[], []]):
            policy = finite_acquisition_policy([[0, 1], [1, 0]], outcomes)
            self.assertEqual(policy.value, 0.5)
            self.assertEqual(policy.optimal_choices, ("stop",))
            self.assertEqual(policy.candidate_values, {})
            self.assertEqual(policy.terminal_actions, (0, 1))

    def test_all_candidates_used_up_stops(self):
        policy = _perfect_two_state(cost=0.0, horizon=2)
        (branch0, branch1) = policy.candidate_values[0].branches.values()
        for branch in (branch0, branch1):
            self.assertEqual(branch.node.candidate_values, {})
            self.assertEqual(branch.node.optimal_choices, ("stop",))

    def test_uninformative_free_candidate_ties_with_stop(self):
        policy = finite_acquisition_policy(
            [[0, 1], [0, 1], [1, 0], [1, 0]], [[0], [0], [0], [0]], horizon=1
        )
        self.assertEqual(policy.value, 0.5)
        self.assertEqual(policy.optimal_choices, ("stop", 0))
        self.assertEqual(policy.candidate_values[0].branches[0].probability, 1.0)

    def test_uninformative_free_tie_survives_non_dyadic_weights(self):
        policy = finite_acquisition_policy(
            [[0, 1], [1, 0], [0.5, 0.5]], [[3], [3], [3]], weights=[0.1, 0.2, 0.7], horizon=1
        )
        self.assertEqual(policy.optimal_choices, ("stop", 0))

    def test_uninformative_costly_candidate_never_chosen(self):
        policy = finite_acquisition_policy(
            [[0, 1], [0, 1], [1, 0], [1, 0]], [[0], [0], [0], [0]], costs=[0.125], horizon=1
        )
        self.assertEqual(policy.optimal_choices, ("stop",))
        self.assertEqual(policy.candidate_values[0].value, 0.625)

    def test_redundant_second_measurement_ties_only_when_free(self):
        losses = [[0, 1], [0, 1], [1, 0], [1, 0]]
        outcomes = [[0, 0], [0, 0], [1, 1], [1, 1]]
        free = finite_acquisition_policy(losses, outcomes, horizon=2)
        self.assertEqual(free.value, 0.0)
        self.assertEqual(free.optimal_choices, (0, 1))
        for branch in free.candidate_values[0].branches.values():
            self.assertEqual(branch.node.stop_risk, 0.0)
            self.assertEqual(branch.node.optimal_choices, ("stop", 1))
        paid = finite_acquisition_policy(losses, outcomes, costs=[0.125, 0.125], horizon=2)
        self.assertEqual(paid.value, 0.125)
        self.assertEqual(paid.optimal_choices, (0, 1))
        for branch in paid.candidate_values[0].branches.values():
            self.assertEqual(branch.node.optimal_choices, ("stop",))
            self.assertEqual(branch.node.candidate_values[1].value, 0.125)

    def test_costly_stop_threshold_exact_tie(self):
        below = _perfect_two_state(cost=0.25)
        self.assertEqual(below.optimal_choices, (0,))
        self.assertEqual(below.value, 0.25)
        tie = _perfect_two_state(cost=0.5)
        self.assertEqual(tie.stop_risk, 0.5)
        self.assertEqual(tie.candidate_values[0].value, 0.5)
        self.assertEqual(tie.optimal_choices, ("stop", 0))
        above = _perfect_two_state(cost=0.75)
        self.assertEqual(above.optimal_choices, ("stop",))
        self.assertEqual(above.value, 0.5)

    def test_acquisition_tie_between_symmetric_candidates(self):
        policy = finite_acquisition_policy(
            [[0, 1], [0, 1], [1, 0], [1, 0]],
            [[0, 1], [0, 1], [1, 0], [1, 0]],
            costs=[0.125, 0.125],
            horizon=1,
        )
        self.assertEqual(policy.value, 0.125)
        self.assertEqual(policy.optimal_choices, (0, 1))

    def test_asymmetric_loss_with_abstention_and_action_tie(self):
        losses = [[0, 4, 1], [3, 0, 1]]  # actions: act0, act1, abstain
        outcomes = [[0], [1]]
        policy = finite_acquisition_policy(losses, outcomes, weights=[1, 3], costs=[0.5], horizon=1)
        self.assertEqual(policy.stop_risk, 1.0)
        self.assertEqual(policy.terminal_actions, (1, 2))
        self.assertEqual(policy.optimal_choices, (0,))
        self.assertEqual(policy.value, 0.5)
        branches = policy.candidate_values[0].branches
        self.assertEqual(branches[0].probability, 0.25)
        self.assertEqual(branches[1].probability, 0.75)
        self.assertEqual(branches[0].node.terminal_actions, (0,))
        self.assertEqual(branches[1].node.terminal_actions, (1,))
        tie = finite_acquisition_policy(losses, outcomes, weights=[1, 3], costs=[1.0], horizon=1)
        self.assertEqual(tie.optimal_choices, ("stop", 0))
        high = finite_acquisition_policy(losses, outcomes, weights=[1, 3], costs=[1.5], horizon=1)
        self.assertEqual(high.optimal_choices, ("stop",))

    def test_exact_equality_not_tolerance_decides_ties(self):
        # 0.1 + 0.2 == 0.30000000000000004 != 0.3 in float64: no tolerance tie.
        policy = finite_acquisition_policy([[0.1 + 0.2, 0.3]], np.zeros((1, 0), dtype=int))
        self.assertEqual(policy.terminal_actions, (1,))
        self.assertEqual(policy.stop_risk, 0.3)
        exact = finite_acquisition_policy([[0.3, 0.3]], np.zeros((1, 0), dtype=int))
        self.assertEqual(exact.terminal_actions, (0, 1))


# --------------------------------------------------------------------------
# Joint (correlated) conditioning
# --------------------------------------------------------------------------


class TestCorrelatedOutcomes(_PolicyTestCase):
    def setUp(self):
        # S = B xor C with B, C iid fair bits: each alone says nothing, jointly exact.
        atoms = list(itertools.product((0, 1), repeat=2))
        self.losses = [[0 if action == (b ^ c) else 1 for action in (0, 1)] for b, c in atoms]
        self.outcomes = [list(atom) for atom in atoms]

    def test_one_step_sees_no_value_in_either_measurement_alone(self):
        policy = finite_acquisition_policy(self.losses, self.outcomes, horizon=1)
        self.assertEqual(policy.value, 0.5)
        self.assertEqual(policy.optimal_choices, ("stop", 0, 1))

    def test_two_step_joint_value_is_found(self):
        policy = finite_acquisition_policy(self.losses, self.outcomes, horizon=2)
        self.assertEqual(policy.value, 0.0)
        self.assertEqual(policy.optimal_choices, (0, 1))
        for branch in policy.candidate_values[0].branches.values():
            self.assertEqual(branch.node.optimal_choices, (1,))
            self.assertEqual(branch.node.stop_risk, 0.5)

    def test_second_measurement_priced_into_total(self):
        policy = finite_acquisition_policy(self.losses, self.outcomes, costs=[0.125] * 2)
        self.assertEqual(policy.value, 0.25)
        self.assertEqual(policy.optimal_choices, (0, 1))
        self.assertEqual(_audit_total(policy, [0.125] * 2), 0.25)

    def test_matches_rational_oracle_and_fixed_subsets(self):
        args = (self.losses, self.outcomes, [Fraction(1)] * 4, [Fraction(1, 8)] * 2)
        self.assertEqual(_oracle_value(*args, 2), Fraction(1, 4))
        self.assertEqual(_oracle_best_fixed(*args, 2), Fraction(1, 4))
        self.assertEqual(_oracle_best_fixed(*args, 1), Fraction(1, 2))

    def test_nonuniform_joint_weights_are_not_a_product_of_marginals(self):
        # B and C are perfectly dependent here; a product-of-marginals model
        # would believe C still carries information after B is observed.
        losses = [[0, 1], [1, 0]]
        outcomes = [[0, 0], [1, 1]]
        policy = finite_acquisition_policy(losses, outcomes, weights=[1, 3], costs=[0.0625] * 2)
        self.assertEqual(policy.optimal_choices, (0, 1))
        for branch in policy.candidate_values[0].branches.values():
            self.assertEqual(branch.node.optimal_choices, ("stop",))
            self.assertEqual(branch.node.stop_risk, 0.0)


# --------------------------------------------------------------------------
# Zero mass, permutations, immutability
# --------------------------------------------------------------------------


class TestZeroMass(_PolicyTestCase):
    def test_zero_weight_atom_is_ignored_for_conditioning(self):
        losses, outcomes, _, _ = _eight_atom_case()
        base = finite_acquisition_policy(losses, outcomes, costs=[0.02] * 3)
        # Extra atom: absurd losses and a label no positive atom ever produces.
        padded = finite_acquisition_policy(
            [*losses, [1e6, -1e6]],
            [*outcomes, [7, 7, 7]],
            weights=[1] * 8 + [0],
            costs=[0.02] * 3,
        )
        self.assertEqual(_flatten(padded), _flatten(base))
        self.assertEqual(padded.support, tuple(range(8)))
        for evidence in padded.candidate_values.values():
            self.assertNotIn(7, evidence.branches)

    def test_zero_weight_atom_in_middle_keeps_original_indices(self):
        policy = finite_acquisition_policy(
            [[0, 1], [9, -9], [1, 0]], [[0], [5], [1]], weights=[1, 0, 1], costs=[0.0]
        )
        self.assertEqual(policy.support, (0, 2))
        self.assertEqual(list(policy.candidate_values[0].branches), [0, 1])
        self.assertEqual(policy.value, 0.0)

    def test_candidate_varying_only_on_zero_mass_atoms_cannot_split(self):
        policy = finite_acquisition_policy(
            [[0, 1], [1, 0], [0, 1]], [[0], [0], [1]], weights=[1, 1, 0], horizon=1
        )
        self.assertEqual(list(policy.candidate_values[0].branches), [0])
        self.assertEqual(policy.candidate_values[0].branches[0].probability, 1.0)
        self.assertEqual(policy.optimal_choices, ("stop", 0))

    def test_zero_weight_rows_are_still_validated(self):
        with self.assertRaises(ValueError):
            finite_acquisition_policy([[0, 1], [np.nan, 0]], [[0], [1]], weights=[1, 0])
        with self.assertRaises(ValueError):
            finite_acquisition_policy([[0, 1], [np.inf, 0]], [[0], [1]], weights=[1, 0])
        with self.assertRaises(TypeError):
            finite_acquisition_policy([[0, 1], [1, 0]], np.array([[0], [1.5]]), weights=[1, 0])

    def test_positive_weight_lost_to_float64_is_rejected_not_dropped(self):
        # Both supplied weights are positive, but 1e-200 / 1e200 underflows to 0.
        # Dropping atom 0 would silently change the joint model (spurious tie).
        losses = [[1e200, 0.0], [0.0, 1.0]]
        outcomes = [[1], [0]]
        with self.assertRaises(ValueError):
            finite_acquisition_policy(
                losses, outcomes, weights=[1e-200, 1e200], costs=[0.0], horizon=1
            )
        # The overflow-rescaling path rejects too.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaises(ValueError):
                finite_acquisition_policy(
                    [*losses, [0.0, 0.0]],
                    [*outcomes, [0]],
                    weights=[1e-320, 1e308, 1e308],
                    horizon=1,
                )
        # A genuine zero weight is still ignored, not rejected.
        policy = finite_acquisition_policy(losses, outcomes, weights=[0.0, 1.0], horizon=1)
        self.assertEqual(policy.support, (1,))

    def test_no_positive_mass_rejected(self):
        with self.assertRaises(ValueError):
            finite_acquisition_policy([[0, 1], [1, 0]], [[0], [1]], weights=[0, 0])

    def test_huge_and_tiny_weights_match_uniform_without_warnings(self):
        losses, outcomes, _, _ = _eight_atom_case()
        base = finite_acquisition_policy(losses, outcomes, costs=[0.02] * 3)
        for scale in (1e308, 1e-320, 1e-300):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                policy = finite_acquisition_policy(
                    losses, outcomes, weights=[scale] * 8, costs=[0.02] * 3
                )
            self.assertClose(policy.value, base.value)
            self.assertEqual(policy.optimal_choices, base.optimal_choices)

    def test_weights_need_not_be_normalized(self):
        losses = [[0, 4, 1], [3, 0, 1]]
        a = finite_acquisition_policy(losses, [[0], [1]], weights=[1, 3], costs=[0.5], horizon=1)
        b = finite_acquisition_policy(losses, [[0], [1]], weights=[10, 30], costs=[0.5], horizon=1)
        self.assertEqual(_flatten(a), _flatten(b))


class TestPermutationInvariance(_PolicyTestCase):
    def test_atom_permutation(self):
        losses, outcomes, _, _ = _eight_atom_case()
        losses = np.array(losses)
        outcomes = np.array(outcomes)
        base = finite_acquisition_policy(losses, outcomes, costs=[0.02] * 3)
        perm = np.array([5, 2, 7, 0, 3, 6, 1, 4])
        shuffled = finite_acquisition_policy(
            losses[perm], outcomes[perm], weights=np.ones(8)[perm], costs=[0.02] * 3
        )
        self.assertClose(shuffled.value, base.value)
        self.assertEqual(shuffled.optimal_choices, base.optimal_choices)
        self.assertEqual(shuffled.terminal_actions, base.terminal_actions)
        for c in range(3):
            self.assertClose(shuffled.candidate_values[c].value, base.candidate_values[c].value)
        self.assertEqual(
            shuffled.candidate_values[0].branches[1].node.optimal_choices,
            base.candidate_values[0].branches[1].node.optimal_choices,
        )

    def test_atom_permutation_with_unequal_weights(self):
        rng = np.random.default_rng(3)
        losses = rng.integers(0, 6, size=(9, 3)).astype(float)
        outcomes = rng.integers(0, 3, size=(9, 3))
        weights = rng.integers(1, 6, size=9).astype(float)
        perm = rng.permutation(9)
        a = finite_acquisition_policy(losses, outcomes, weights=weights, costs=[0.1, 0.2, 0.3])
        b = finite_acquisition_policy(
            losses[perm], outcomes[perm], weights=weights[perm], costs=[0.1, 0.2, 0.3]
        )
        self.assertAlmostEqual(a.value, b.value, places=12)

    def test_candidate_permutation_relabels_choices(self):
        losses, outcomes, _, _ = _eight_atom_case()
        losses = np.array(losses)
        outcomes = np.array(outcomes)
        base = finite_acquisition_policy(losses, outcomes, costs=[0.02] * 3)
        perm = [2, 0, 1]  # new candidate j is old candidate perm[j]
        new_of_old = {old: new for new, old in enumerate(perm)}
        shuffled = finite_acquisition_policy(losses, outcomes[:, perm], costs=[0.02] * 3)
        self.assertClose(shuffled.value, base.value)
        self.assertEqual(shuffled.optimal_choices, (new_of_old[0],))
        for new, old in enumerate(perm):
            self.assertClose(shuffled.candidate_values[new].value, base.candidate_values[old].value)
        after_a = shuffled.candidate_values[new_of_old[0]].branches
        self.assertEqual(after_a[0].node.optimal_choices, (new_of_old[1],))
        self.assertEqual(after_a[1].node.optimal_choices, (new_of_old[2],))

    def test_candidate_permutation_with_distinct_costs(self):
        losses, outcomes, _, _ = _eight_atom_case()
        outcomes = np.array(outcomes)
        costs = np.array([0.02, 0.05, 0.03])
        base = finite_acquisition_policy(losses, outcomes, costs=costs)
        perm = [1, 2, 0]
        shuffled = finite_acquisition_policy(losses, outcomes[:, perm], costs=costs[perm])
        self.assertAlmostEqual(shuffled.value, base.value, places=12)

    def test_action_permutation_relabels_terminal_actions(self):
        losses = np.array([[0, 4, 1], [3, 0, 1]], dtype=float)
        outcomes = [[0], [1]]
        base = finite_acquisition_policy(losses, outcomes, weights=[1, 3], costs=[0.5], horizon=1)
        perm = [2, 0, 1]  # new action j is old action perm[j]
        shuffled = finite_acquisition_policy(
            losses[:, perm], outcomes, weights=[1, 3], costs=[0.5], horizon=1
        )
        expected = tuple(j for j, old in enumerate(perm) if old in base.terminal_actions)
        self.assertEqual(shuffled.terminal_actions, expected)
        self.assertEqual(shuffled.value, base.value)
        self.assertEqual(shuffled.optimal_choices, base.optimal_choices)


class TestImmutabilityAndTypes(_PolicyTestCase):
    def test_inputs_are_not_modified(self):
        losses, outcomes, _, _ = _eight_atom_case()
        losses = np.array(losses, dtype=float)
        outcomes = np.array(outcomes)
        weights = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
        costs = np.array([0.02, 0.03, 0.04])
        arrays = (losses, outcomes, weights, costs)
        snapshot = [a.copy() for a in arrays]
        finite_acquisition_policy(losses, outcomes, weights=weights, costs=costs)
        for index, before in enumerate(snapshot):
            np.testing.assert_array_equal(before, arrays[index])
            self.assertEqual(before.dtype, arrays[index].dtype)

    def test_readonly_inputs_accepted(self):
        losses = np.array([[0.0, 1.0], [1.0, 0.0]])
        outcomes = np.array([[0], [1]])
        for arr in (losses, outcomes):
            arr.setflags(write=False)
        policy = finite_acquisition_policy(losses, outcomes, costs=[0.25], horizon=1)
        self.assertEqual(policy.value, 0.25)

    def test_result_is_frozen(self):
        policy = _eight_atom_policy()
        with self.assertRaises(FrozenInstanceError):
            policy.value = 0.0
        with self.assertRaises(TypeError):
            policy.candidate_values[0] = None
        evidence = policy.candidate_values[0]
        self.assertIsInstance(evidence, CandidateEvidence)
        with self.assertRaises(FrozenInstanceError):
            evidence.value = 0.0
        with self.assertRaises(TypeError):
            evidence.branches[0] = None

    def test_repeat_calls_are_identical(self):
        self.assertEqual(_flatten(_eight_atom_policy()), _flatten(_eight_atom_policy()))

    def test_result_uses_plain_python_types(self):
        policy = _eight_atom_policy()
        self.assertIs(type(policy.value), float)
        self.assertTrue(all(type(a) is int for a in policy.terminal_actions))
        self.assertTrue(all(type(k) is int for k in policy.candidate_values))
        self.assertTrue(all(type(k) is int for k in policy.candidate_values[0].branches))

    def test_arbitrary_integer_labels_and_unsigned_dtype(self):
        outcomes = np.array([[-3], [10**9]], dtype=np.int64)
        policy = finite_acquisition_policy([[0, 1], [1, 0]], outcomes, costs=[0.25], horizon=1)
        self.assertEqual(sorted(policy.candidate_values[0].branches), [-3, 10**9])
        unsigned = finite_acquisition_policy(
            [[0, 1], [1, 0]], np.array([[0], [1]], dtype=np.uint8), costs=[0.25], horizon=1
        )
        self.assertEqual(unsigned.value, 0.25)

    def test_numpy_integer_horizon_accepted(self):
        self.assertEqual(_perfect_two_state(cost=0.25, horizon=np.int64(1)).value, 0.25)


# --------------------------------------------------------------------------
# Randomized fixture at the acceptance ceiling vs the rational oracle
# --------------------------------------------------------------------------


class TestCeilingFixtureAgainstOracle(_PolicyTestCase):
    def test_128_atoms_6_candidates_4_actions_horizon_2(self):
        rng = np.random.default_rng(20261006)
        losses = rng.integers(0, 10, size=(128, 4))
        outcomes = rng.integers(0, 3, size=(128, 6))
        weights = rng.integers(0, 8, size=128)  # includes exact-zero-mass atoms
        weights[0] = 3
        cost_numerators = rng.integers(0, 8, size=6)
        costs = cost_numerators / 64.0

        policy = finite_acquisition_policy(losses, outcomes, weights=weights, costs=costs)
        self.assertNodeConsistent(policy, costs, 6, 2)

        r_losses = losses.tolist()
        r_outcomes = outcomes.tolist()
        r_weights = [Fraction(int(w)) for w in weights]
        r_costs = [Fraction(int(n), 64) for n in cost_numerators]
        expected = _oracle_value(r_losses, r_outcomes, r_weights, r_costs, 2)
        self.assertClose(policy.value, expected, places=9)
        for c, evidence in policy.candidate_values.items():
            alive = [i for i, w in enumerate(r_weights) if w > 0]
            oracle_c = _oracle_candidate(
                r_losses, r_outcomes, r_weights, r_costs, 2, alive, frozenset(range(6)), c
            )
            self.assertClose(evidence.value, oracle_c, places=9)
        for choice in policy.optimal_choices:
            if choice != "stop":
                self.assertClose(policy.candidate_values[choice].value, expected, places=9)
        self.assertClose(_audit_total(policy, costs), expected, places=9)


# --------------------------------------------------------------------------
# Malformed input
# --------------------------------------------------------------------------


class TestMalformedInput(_PolicyTestCase):
    losses = [[0.0, 1.0], [1.0, 0.0]]
    outcomes = [[0], [1]]

    def call(self, **overrides):
        kwargs = {"losses": self.losses, "outcomes": self.outcomes}
        kwargs.update(overrides)
        return finite_acquisition_policy(**kwargs)

    def test_valid_baseline(self):
        self.assertEqual(self.call().value, 0.0)

    def test_losses_shape_and_dtype(self):
        for bad in (np.zeros(3), np.zeros((2, 2, 2)), np.zeros((0, 2)), np.zeros((2, 0))):
            with self.subTest(shape=np.shape(bad)), self.assertRaises(ValueError):
                self.call(losses=bad, outcomes=np.zeros((np.shape(bad)[0], 1), dtype=int))
        non_numeric = (np.array([[True, False], [False, True]]), np.array([["a", "b"], ["c", "d"]]))
        for bad in non_numeric:
            with self.subTest(dtype=bad.dtype), self.assertRaises(TypeError):
                self.call(losses=bad)

    def test_losses_must_be_finite(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.call(losses=[[0.0, value], [1.0, 0.0]])

    def test_outcomes_shape_dtype_and_rows(self):
        for bad in (np.zeros(2, dtype=int), np.zeros((2, 1, 1), dtype=int)):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                self.call(outcomes=bad)
        with self.assertRaises(ValueError):
            self.call(outcomes=[[0], [1], [0]])
        with self.assertRaises(ValueError):
            self.call(outcomes=np.zeros((0, 1), dtype=int))
        for bad in (
            np.array([[0.0], [1.0]]),
            np.array([[True], [False]]),
            np.array([["a"], ["b"]]),
            np.array([[0.5], [1.5]]),
        ):
            with self.subTest(dtype=bad.dtype), self.assertRaises(TypeError):
                self.call(outcomes=bad)

    def test_weights(self):
        bad_weights = ([1.0], [1.0] * 3, [[1.0, 1.0]], [-1.0, 2.0], [np.nan, 1.0], [np.inf, 1.0])
        for bad in bad_weights:
            with self.subTest(weights=bad), self.assertRaises(ValueError):
                self.call(weights=bad)
        with self.assertRaises(ValueError):
            self.call(weights=[0.0, 0.0])
        with self.assertRaises(TypeError):
            self.call(weights=[True, True])
        with self.assertRaises(TypeError):
            self.call(weights=["a", "b"])

    def test_costs(self):
        for bad in ([], [0.0, 0.0], [[0.0]], [-0.1], [np.nan], [np.inf]):
            with self.subTest(costs=bad), self.assertRaises(ValueError):
                self.call(costs=bad)
        with self.assertRaises(TypeError):
            self.call(costs=[True])

    def test_costs_with_zero_candidates(self):
        with self.assertRaises(ValueError):
            self.call(outcomes=np.zeros((2, 0), dtype=int), costs=[0.1])
        self.assertEqual(self.call(outcomes=np.zeros((2, 0), dtype=int), costs=[]).value, 0.5)

    def test_horizon_validation(self):
        for bad in (True, False, np.bool_(True), 1.0, 2.0, "1", None, [1], 1.5):
            with self.subTest(horizon=bad), self.assertRaises(TypeError):
                self.call(horizon=bad)
        for bad in (-1, 3, 10, np.int64(-1), np.int64(3)):
            with self.subTest(horizon=bad), self.assertRaises(ValueError):
                self.call(horizon=bad)
        for good in (0, 1, 2, np.int32(2)):
            self.call(horizon=good)

    def test_value_overflow_rejected_not_hidden(self):
        huge = [[1.7e308, 1.7e308], [1.7e308, 1.7e308]]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaises(ValueError):
                self.call(costs=[1.7e308], losses=huge)

    def test_keyword_only_options(self):
        with self.assertRaises(TypeError):
            finite_acquisition_policy(self.losses, self.outcomes, [1, 1])  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
