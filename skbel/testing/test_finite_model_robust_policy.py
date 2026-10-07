#  Copyright (c) 2026. Robin Thibaut, Ghent University

"""Tests for model-robust finite acquisition and stopping.

All fixtures are fresh literal tables with dyadic weights, costs and integer
losses, so float64 arithmetic is exact and production values can be compared with
``Fraction`` values using ``==``. The oracle below enumerates every complete
policy itself and evaluates it from joint masses; it uses no production
conditioning, optimizer or enumeration helper.

``RobustResult.policy`` is a public call that rebuilds a tree, so it is used only
four times in this module. Everything else compares the owned risk table, the
minimizer and baseline indices and the already-built ``RobustResult.robust`` trees
with the oracle.
"""

import dataclasses
import itertools
from fractions import Fraction

import numpy as np
import pytest

from skbel.metrics import finite_acquisition_policy
from skbel.metrics import robust as robust_module
from skbel.metrics.robust import robust_acquisition_policy

# --------------------------------------------------------------------------- oracle


def stop(action):
    return ("stop", action)


def acquire(candidate, *children):
    """Tree reading ``candidate``, then one child per label 0, 1, ..."""
    return ("acquire", candidate, tuple(enumerate(children)))


def oracle_trees(world, alive, used, horizon):
    """Every complete policy for the atoms in ``alive`` (canonical order)."""
    labels, n_actions, n_candidates = world
    trees = [stop(a) for a in range(n_actions)]
    if horizon == 0:
        return trees
    for c in range(n_candidates):
        if c in used:
            continue
        values = sorted({labels[i][c] for i in alive})
        subs = []
        for v in values:
            part = [i for i in alive if labels[i][c] == v]
            subs.append(oracle_trees(world, part, (*used, c), horizon - 1))
        for combo in itertools.product(*subs):
            trees.append(("acquire", c, tuple(zip(values, combo, strict=True))))
    return trees


class Oracle:
    """Exact whole-tree evaluation under each fixed model, from joint masses."""

    def __init__(self, losses, outcomes, weights, costs, horizon):
        self.losses = [[Fraction(x) for x in row] for row in losses]
        self.labels = [list(row) for row in outcomes]
        self.costs = [Fraction(x) for x in costs]
        self.models = []
        for row in weights:
            total = sum(Fraction(x) for x in row)
            self.models.append([Fraction(x) / total for x in row])
        n_atoms = len(self.losses)
        self.alive = [i for i in range(n_atoms) if any(w[i] > 0 for w in self.models)]
        world = (self.labels, len(self.losses[0]), len(self.costs))
        self.trees = oracle_trees(world, self.alive, (), horizon)
        self.table = []
        for tree in self.trees:
            self.table.append([self.risk(tree, self.alive, w) for w in self.models])
        self.worst = [max(row) for row in self.table]

    def risk(self, tree, alive, w):
        if tree[0] == "stop":
            return sum(w[i] * self.losses[i][tree[1]] for i in alive)
        _, c, subtrees = tree
        total = self.costs[c] * sum(w[i] for i in alive)
        for value, sub in subtrees:
            part = [i for i in alive if self.labels[i][c] == value]
            total += self.risk(sub, part, w)
        return total

    def argmin(self, indices, values):
        best = min(values[i] for i in indices)
        return best, [i for i in indices if values[i] == best]

    def baseline(self, pool, objective):
        best, optimal = self.argmin(pool, objective)
        floor, selected = self.argmin(optimal, self.worst)
        return best, optimal, floor, selected

    def leaf_sets(self, tree):
        if tree[0] == "stop":
            return [frozenset()]
        _, c, subtrees = tree
        return [s | {c} for _, sub in subtrees for s in self.leaf_sets(sub)]

    def fixed_pool(self):
        pool = []
        for i, tree in enumerate(self.trees):
            if len(set(self.leaf_sets(tree))) == 1:
                pool.append(i)
        return pool

    def leaves(self, tree, alive, path_cost):
        if tree[0] == "stop":
            totals = {i: path_cost + self.losses[i][tree[1]] for i in alive}
            return [[sum(w[i] * t for i, t in totals.items()) for w in self.models]]
        _, c, subtrees = tree
        out = []
        for value, sub in subtrees:
            part = [i for i in alive if self.labels[i][c] == value]
            out += self.leaves(sub, part, path_cost + self.costs[c])
        return out

    def switching(self, tree):
        """Adversary picks the model again at every leaf (not the API's objective)."""
        return sum(max(leaf) for leaf in self.leaves(tree, self.alive, Fraction(0)))


def key(node):
    """Canonical nested-tuple form of a production ``PolicyNode``."""
    if node.kind == "stop":
        return stop(node.action)
    pairs = tuple((label, key(node.branches[label])) for label in sorted(node.branches))
    return ("acquire", node.candidate, pairs)


def table_of(result):
    return [[Fraction(x) for x in row] for row in result.risk_table.tolist()]


def assert_baseline(baseline, expected):
    objective, optimal, floor, selected = expected
    assert Fraction(baseline.objective) == objective
    assert list(baseline.optimal_indices) == optimal
    assert Fraction(baseline.worst_risk) == floor
    assert list(baseline.selected_indices) == selected


def assert_matches_oracle(result, oracle):
    """Table, order, minimizers and every baseline equal the independent oracle's."""
    assert result.policy_count == len(oracle.trees)
    assert table_of(result) == oracle.table
    everything = list(range(len(oracle.trees)))
    best, indices = oracle.argmin(everything, oracle.worst)
    assert Fraction(result.worst_risk) == best
    assert [p.index for p in result.robust] == indices
    for policy in result.robust:
        assert key(policy.root) == oracle.trees[policy.index]
    for m, baseline in enumerate(result.nominal):
        column = [row[m] for row in oracle.table]
        assert_baseline(baseline, oracle.baseline(everything, column))
    fixed = oracle.baseline(oracle.fixed_pool(), oracle.worst)
    assert_baseline(result.fixed_subset, fixed)
    stops = [i for i, tree in enumerate(oracle.trees) if tree[0] == "stop"]
    assert_baseline(result.stop_only, oracle.baseline(stops, oracle.worst))


def chosen(baseline, oracle):
    return [oracle.trees[i] for i in baseline.selected_indices]


# --------------------------------------------------------------------------- fixtures

# Main exhibit. Atom order is (x, y) = 00, 01, 10, 11. Candidate 0 returns x and
# candidate 1 returns y. Action 1 is right only for atom 01; it costs 4 on every
# other atom, while missing atom 01 with action 0 costs 16.
LOSSES = [[0, 4], [16, 0], [0, 4], [0, 4]]
OUTCOMES = [[0, 0], [0, 1], [1, 0], [1, 1]]
COSTS = [1, 2]
IDS = ["a00", "a01", "a10", "a11"]
WEIGHTS = [[7, 1, 4, 4], [2, 8, 3, 3], [7, 5, 2, 2]]  # out of 16 each
MIXTURE = [0.25, 0.5, 0.25]

STOP0 = stop(0)
STOP1 = stop(1)
X_THEN_Y_ADAPTIVE = acquire(0, acquire(1, STOP0, STOP1), STOP0)  # y only if x = 0
X_ONLY = acquire(0, STOP1, STOP0)
Y_ONLY = acquire(1, STOP0, STOP1)
Y_THEN_X_ADAPTIVE = acquire(1, STOP0, acquire(0, STOP1, STOP0))  # x only if y = 1
BOTH_X_FIRST = acquire(0, acquire(1, STOP0, STOP1), acquire(1, STOP0, STOP0))
BOTH_Y_FIRST = acquire(1, acquire(0, STOP0, STOP0), acquire(0, STOP1, STOP0))

# Paid fair coin independent of the state; model 0 sees state 0, model 1 state 1.
COIN_LOSSES = [[0, 4], [0, 4], [4, 0], [4, 0]]
COIN_OUTCOMES = [[0], [1], [0], [1]]
COIN_WEIGHTS = [[1, 1, 0, 0], [0, 0, 1, 1]]
COIN_IDS = ["s0c0", "s0c1", "s1c0", "s1c1"]
COIN_COSTS = [0.25]


def coin_call(**overrides):
    kwargs = {
        "losses": COIN_LOSSES,
        "outcomes": COIN_OUTCOMES,
        "model_weights": COIN_WEIGHTS,
        "atom_ids": COIN_IDS,
        "model_atom_ids": [COIN_IDS, COIN_IDS],
        "costs": COIN_COSTS,
        "horizon": 1,
    }
    kwargs.update(overrides)
    return robust_acquisition_policy(**kwargs)


@pytest.fixture(scope="module")
def main():
    return robust_acquisition_policy(
        LOSSES,
        OUTCOMES,
        WEIGHTS,
        IDS,
        [IDS] * 3,
        costs=COSTS,
        horizon=2,
        mixture_weights=MIXTURE,
    )


@pytest.fixture(scope="module")
def main_oracle():
    return Oracle(LOSSES, OUTCOMES, WEIGHTS, COSTS, 2)


@pytest.fixture(scope="module")
def planner_values():
    """Existing single-law planner value for each model of the main exhibit."""
    values = []
    for row in WEIGHTS:
        planner = finite_acquisition_policy(
            LOSSES,
            OUTCOMES,
            weights=row,
            costs=COSTS,
            horizon=2,
        )
        values.append(planner.value)
    return values


# --------------------------------------------------------------------------- main exhibit


def test_main_matches_independent_oracle(main, main_oracle):
    assert main.policy_count == 74
    assert_matches_oracle(main, main_oracle)
    table = main_oracle.table
    pi = [Fraction(p) for p in MIXTURE]
    mixed = [sum(p * r for p, r in zip(pi, row, strict=True)) for row in table]
    expected = main_oracle.baseline(list(range(len(table))), mixed)
    assert_baseline(main.mixture, expected)


def test_main_root_vectors_and_global_minimum(main, main_oracle):
    rows = dict(zip(main_oracle.trees, main.risk_table.tolist(), strict=True))
    expected = {
        STOP0: [1.0, 8.0, 5.0],
        STOP1: [3.75, 2.0, 2.75],
        X_ONLY: [2.75, 1.5, 2.75],
        X_THEN_Y_ADAPTIVE: [2.0, 2.25, 2.5],
        Y_ONLY: [3.0, 2.75, 2.5],
        Y_THEN_X_ADAPTIVE: [2.3125, 2.6875, 2.4375],
        BOTH_X_FIRST: [3.0, 3.0, 3.0],
        BOTH_Y_FIRST: [3.0, 3.0, 3.0],
    }
    for tree, vector in expected.items():
        assert rows[tree] == vector
    assert [key(p.root) for p in main.robust] == [X_THEN_Y_ADAPTIVE]
    only = main.robust[0]
    assert only.model_risks == (2.0, 2.25, 2.5)
    assert only.worst_risk == main.worst_risk == 2.5
    assert only.worst_models == (2,)
    # Every other policy is strictly worse in the worst model.
    others = [max(row) for i, row in enumerate(rows.values()) if i != only.index]
    assert min(others) > 2.5


def test_main_baselines_differ_and_are_strictly_worse(main, main_oracle):
    nominal_0, nominal_1, nominal_2 = main.nominal
    assert chosen(nominal_0, main_oracle) == [STOP0]
    assert (nominal_0.objective, nominal_0.worst_risk) == (1.0, 8.0)
    assert chosen(nominal_1, main_oracle) == [X_ONLY]
    assert (nominal_1.objective, nominal_1.worst_risk) == (1.5, 2.75)
    assert chosen(nominal_2, main_oracle) == [Y_THEN_X_ADAPTIVE]
    assert (nominal_2.objective, nominal_2.worst_risk) == (2.4375, 2.6875)
    for baseline in main.nominal:
        assert len(baseline.optimal_indices) == 1  # no zero-history completions here

    assert chosen(main.mixture, main_oracle) == [X_ONLY]
    assert (main.mixture.objective, main.mixture.worst_risk) == (2.125, 2.75)
    assert chosen(main.fixed_subset, main_oracle) == [X_ONLY]
    assert main.fixed_subset.worst_risk == 2.75
    assert chosen(main.stop_only, main_oracle) == [STOP1]
    assert main.stop_only.worst_risk == 3.75

    competitors = [b.worst_risk for b in main.nominal]
    competitors += [main.mixture.worst_risk, main.fixed_subset.worst_risk]
    competitors.append(main.stop_only.worst_risk)
    assert all(worst > main.worst_risk for worst in competitors)
    assert main.robust[0].root.kind == "acquire"
    assert main.robust[0].root.candidate == 0


def test_main_information_is_genuine(main, main_oracle):
    """Acquisition and outcome-dependent terminal actions carry the worst-case gain."""
    stops = [i for i, t in enumerate(main_oracle.trees) if t[0] == "stop"]
    gains = []
    for m in range(3):
        column = [row[m] for row in main_oracle.table]
        gains.append(min(column[i] for i in stops) - min(column))
    assert gains == [0, Fraction(1, 2), Fraction(5, 16)]
    root = main.robust[0].root
    assert set(root.branches) == {0, 1}
    assert root.branches[0].kind == "acquire" and root.branches[0].candidate == 1
    assert root.branches[1].kind == "stop"
    leaf_actions = {
        root.branches[1].action,
        root.branches[0].branches[0].action,
        root.branches[0].branches[1].action,
    }
    assert leaf_actions == {0, 1}
    assert main.stop_only.worst_risk - main.worst_risk == 1.25


def test_fixed_model_ranking_differs_from_branch_switching(main_oracle):
    """Pairwise reversal only: no claim about the global switching optimum."""
    trees = main_oracle.trees
    switching = [main_oracle.switching(tree) for tree in trees]
    committed = trees.index(X_THEN_Y_ADAPTIVE)
    greedy = trees.index(X_ONLY)
    assert switching[committed] == Fraction(53, 16)
    assert switching[greedy] == 3
    assert main_oracle.worst[committed] == Fraction(5, 2)
    assert main_oracle.worst[greedy] == Fraction(11, 4)
    assert switching[greedy] < switching[committed]
    assert main_oracle.worst[greedy] > main_oracle.worst[committed]


def test_work_counts_and_size_ceilings(main):
    assert main.work.policies == 74
    assert main.work.model_evaluations == 222
    assert (main.work.nodes, main.work.generated) == (13, 114)
    assert (main.work.atoms, main.work.models) == (4, 3)
    assert robust_module._policy_ceiling(2, 2, 2, 2) == 74
    assert robust_module._MAX_POLICIES == 1326
    assert robust_module._MAX_NODES == 31
    assert robust_module._MAX_GENERATED == 1524


# --------------------------------------------------------------------------- reductions


def test_single_law_reduces_to_existing_planner(main, planner_values):
    for baseline, value in zip(main.nominal, planner_values, strict=True):
        assert abs(baseline.objective - value) <= 1e-12
    for value, expected in zip(planner_values, [1.0, 1.5, 2.4375], strict=True):
        assert abs(value - expected) <= 1e-12


def test_single_and_duplicate_model_reduce_to_bayes_optimum(planner_values):
    row = WEIGHTS[2]
    single = robust_acquisition_policy(
        LOSSES,
        OUTCOMES,
        [row],
        IDS,
        [IDS],
        costs=COSTS,
    )
    assert_matches_oracle(single, Oracle(LOSSES, OUTCOMES, [row], COSTS, 2))
    assert single.mixture is None
    assert [key(p.root) for p in single.robust] == [Y_THEN_X_ADAPTIVE]
    assert abs(single.worst_risk - planner_values[2]) <= 1e-12

    rows = [row, row, row]
    duplicate = robust_acquisition_policy(
        LOSSES,
        OUTCOMES,
        rows,
        IDS,
        [IDS] * 3,
        costs=COSTS,
    )
    assert_matches_oracle(duplicate, Oracle(LOSSES, OUTCOMES, rows, COSTS, 2))
    assert [key(p.root) for p in duplicate.robust] == [Y_THEN_X_ADAPTIVE]
    assert duplicate.worst_risk == single.worst_risk
    assert duplicate.robust[0].worst_models == (0, 1, 2)
    assert np.array_equal(duplicate.risk_table[:, 0], single.risk_table[:, 0])


# --------------------------------------------------------------------------- zero mass

# The main table plus one atom that no model supports. Model 0 never sees x = 1
# and model 2 never sees x = 0. Candidate 0 is free, so under those models it
# ties with stopping and the impossible branches are completed arbitrarily.
ZERO_LOSSES = [*LOSSES, [7, 7]]
ZERO_OUTCOMES = [*OUTCOMES, [9, 9]]  # labels that would be a third branch if kept
ZERO_IDS = [*IDS, "dead"]
ZERO_WEIGHTS = [[12, 4, 0, 0, 0], [2, 8, 3, 3, 0], [0, 0, 8, 8, 0]]
ZERO_COSTS = [0, 1]


def test_global_and_per_model_zero_mass():
    result = robust_acquisition_policy(
        ZERO_LOSSES,
        ZERO_OUTCOMES,
        ZERO_WEIGHTS,
        ZERO_IDS,
        [ZERO_IDS] * 3,
        costs=ZERO_COSTS,
    )
    weights = [row[:4] for row in ZERO_WEIGHTS]
    oracle = Oracle(LOSSES, OUTCOMES, weights, ZERO_COSTS, 2)
    # Equal policy count and table: the dropped atom's label 9 never became a branch.
    assert_matches_oracle(result, oracle)
    assert result.policy_count == 74
    assert result.dropped_ids == ("dead",)
    assert result.support_ids == tuple(IDS)
    assert result.mixture is None

    model_0, _, model_2 = result.nominal
    assert model_0.objective == 1.0
    assert len(model_0.optimal_indices) > 1
    assert model_2.objective == 0.0
    assert 0 in model_2.optimal_indices  # stop with action 0
    assert len(model_2.optimal_indices) == 7  # that, plus six completions of x = 0
    optimal = model_0.optimal_indices
    reads_x = [i for i in optimal if oracle.trees[i][:2] == ("acquire", 0)]
    assert reads_x  # model 0 never sees x = 1, yet the tree still has that branch
    tree = result.policy(reads_x[0])
    assert set(tree.branches) == {0, 1}
    assert tree.branches[1].support == (2, 3)
    floor = min(max(result.risk_table[i]) for i in optimal)
    assert model_0.worst_risk == floor


# --------------------------------------------------------------------------- ties, coin, ownership


def test_exact_dyadic_ties_and_paid_coin_disclosure():
    result = coin_call()
    oracle = Oracle(COIN_LOSSES, COIN_OUTCOMES, COIN_WEIGHTS, COIN_COSTS, 1)
    assert_matches_oracle(result, oracle)
    assert result.policy_count == 6
    assert (result.work.policies, result.work.model_evaluations) == (6, 12)
    assert (result.work.nodes, result.work.generated) == (3, 10)
    act_by_coin = acquire(0, STOP0, STOP1)
    act_against_coin = acquire(0, STOP1, STOP0)
    assert [key(p.root) for p in result.robust] == [act_by_coin, act_against_coin]
    assert [p.index for p in result.robust] == [3, 4]  # canonical order
    assert result.worst_risk == 2.25
    for policy in result.robust:
        assert policy.model_risks == (2.25, 2.25)
        assert policy.worst_models == (0, 1)
    # Splicing one branch from each tied minimizer is not a minimizer.
    spliced = result.risk_table[oracle.trees.index(acquire(0, STOP0, STOP0))]
    assert spliced.tolist() == [0.25, 4.25]
    assert max(spliced) > result.worst_risk
    # The coin is independent of the state, so no single model values it,
    # yet the deterministic tree lowers the worst case: randomization, not information.
    assert [chosen(b, oracle)[0] for b in result.nominal] == [STOP0, STOP1]
    assert [b.objective for b in result.nominal] == [0.0, 0.0]
    assert result.stop_only.worst_risk == 4.0
    assert result.fixed_subset.worst_risk == 2.25


def test_inputs_are_copied_and_outputs_are_immutable():
    losses = np.array(COIN_LOSSES, dtype=float)
    outcomes = np.array(COIN_OUTCOMES)
    weights = np.array(COIN_WEIGHTS, dtype=float)
    costs = np.array(COIN_COSTS)
    ids = list(COIN_IDS)
    model_ids = [list(COIN_IDS), list(COIN_IDS)]
    result = coin_call(
        losses=losses,
        outcomes=outcomes,
        model_weights=weights,
        costs=costs,
        atom_ids=ids,
        model_atom_ids=model_ids,
    )
    root = result.policy(3)
    before = (result.risk_table.tolist(), key(root), result.atom_ids, result.worst_risk)
    losses[:] = 99
    outcomes[:] = 5
    weights[:] = 1
    costs[:] = 7
    ids[0] = "mutated"
    model_ids[0][0] = "mutated"
    again = result.policy(3)
    assert result.risk_table.tolist() == before[0]
    assert key(again) == before[1]
    assert result.atom_ids == before[2] == tuple(COIN_IDS)
    assert result.worst_risk == before[3]

    with pytest.raises(ValueError):
        result.risk_table[0, 0] = 1.0
    with pytest.raises(ValueError):
        result.risk_table.setflags(write=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.worst_risk = 0.0
    with pytest.raises(TypeError):
        root.branches[0] = root
    with pytest.raises(dataclasses.FrozenInstanceError):
        root.kind = "stop"
    assert isinstance(result.robust, tuple)
    assert isinstance(result.nominal, tuple)
    assert again is not root
    with pytest.raises(IndexError):
        result.policy(6)


def test_overflow_is_rejected_not_repaired():
    with pytest.raises(ValueError, match="not finite"):
        robust_acquisition_policy(
            [[0, 0], [0, 0]],
            [[0, 0], [1, 1]],
            [[1, 1]],
            ["p", "q"],
            [["p", "q"]],
            costs=[1.7e308, 1.7e308],
            horizon=2,
        )


# --------------------------------------------------------------------------- validation

GRID_IDS = [f"a{i}" for i in range(33)]

NEGATIVE_CASES = [
    pytest.param(
        {
            "losses": np.zeros((33, 2)),
            "outcomes": np.zeros((33, 1), dtype=int),
            "model_weights": np.ones((1, 33)),
            "atom_ids": GRID_IDS,
            "model_atom_ids": [GRID_IDS],
        },
        ValueError,
        r"atoms = 33 exceeds",
        id="too-many-atoms",
    ),
    pytest.param(
        {"losses": np.zeros((4, 4))},
        ValueError,
        r"actions = 4 exceeds",
        id="too-many-actions",
    ),
    pytest.param(
        {"outcomes": np.zeros((4, 4), dtype=int), "costs": None},
        ValueError,
        r"candidates = 4 exceeds",
        id="too-many-candidates",
    ),
    pytest.param(
        {"model_weights": np.ones((9, 4)), "model_atom_ids": [COIN_IDS] * 9},
        ValueError,
        r"models = 9 exceeds",
        id="too-many-models",
    ),
    pytest.param(
        {"outcomes": [[0], [1], [2], [0]]},
        ValueError,
        r"outcome labels",
        id="too-many-labels",
    ),
    pytest.param({"horizon": True}, TypeError, r"horizon", id="horizon-bool"),
    pytest.param({"horizon": 3}, ValueError, r"horizon", id="horizon-range"),
    pytest.param(
        {"losses": [[0, 4], [0, 4], [np.nan, 0], [4, 0]]},
        ValueError,
        r"non-finite",
        id="nan-loss",
    ),
    pytest.param(
        {"outcomes": [[0.0], [1.0], [0.0], [1.0]]},
        TypeError,
        r"integer",
        id="float-outcomes",
    ),
    pytest.param({"costs": [-0.25]}, ValueError, r"non-negative", id="negative-cost"),
    pytest.param(
        {"model_weights": [[1, 1, 0, 0], [0, np.nan, 1, 1]]},
        ValueError,
        r"model_weights\[1\]",
        id="nan-weight",
    ),
    pytest.param(
        {"model_weights": [[1, 1, 0, 0], [0, -1, 1, 1]]},
        ValueError,
        r"non-negative",
        id="negative-weight",
    ),
    pytest.param(
        {"model_weights": [[1, 1, 0, 0], [0, 0, 0, 0]]},
        ValueError,
        r"strictly positive total",
        id="zero-row",
    ),
    pytest.param(
        {"model_weights": [[1e-200, 1e200, 0, 0], [0, 0, 1, 1]]},
        ValueError,
        r"representable",
        id="lost-positive-mass",
    ),
    pytest.param(
        {"model_weights": [[1, 1, 1], [1, 1, 1]]},
        ValueError,
        r"4 columns",
        id="weights-wrong-width",
    ),
    pytest.param(
        {"model_weights": [1, 1, 1, 1]},
        ValueError,
        r"models, atoms",
        id="weights-one-dimensional",
    ),
    pytest.param(
        {"model_atom_ids": [COIN_IDS, COIN_IDS[::-1]]},
        ValueError,
        r"model_atom_ids\[1\]",
        id="ids-order-mismatch",
    ),
    pytest.param(
        {
            "atom_ids": ["x", "x", "y", "z"],
            "model_atom_ids": [["x", "x", "y", "z"]] * 2,
        },
        ValueError,
        r"unique",
        id="ids-duplicate",
    ),
    pytest.param(
        {"atom_ids": [0, 1, 2, 3]},
        TypeError,
        r"string",
        id="ids-not-strings",
    ),
    pytest.param(
        {"mixture_weights": [1, 1, 1]},
        ValueError,
        r"mixture_weights",
        id="mixture-wrong-shape",
    ),
    pytest.param(
        {
            "losses": [*COIN_LOSSES, [np.nan, 0]],
            "outcomes": [*COIN_OUTCOMES, [9]],
            "model_weights": [[1, 1, 0, 0, 0], [0, 0, 1, 1, 0]],
            "atom_ids": [*COIN_IDS, "dead"],
            "model_atom_ids": [[*COIN_IDS, "dead"]] * 2,
        },
        ValueError,
        r"non-finite",
        id="invalid-loss-on-globally-zero-atom",
    ),
]


@pytest.mark.parametrize(("overrides", "error", "pattern"), NEGATIVE_CASES)
def test_invalid_input_is_rejected(overrides, error, pattern):
    with pytest.raises(error, match=pattern):
        coin_call(**overrides)
