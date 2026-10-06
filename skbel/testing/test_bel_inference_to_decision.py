"""Tests for the BEL inference-to-decision example (``examples/bel_inference_to_decision.py``).

One shared fitted fixture (one fit, one two-case predict) serves every test; no test
fits or predicts again. Oracles are independent of the code under test: a Gaussian
Schur conditional from the fitted canonical arrays, an affine fit to the training
rows, ``math.hypot`` / ``fsum`` double loops, a ``Fraction`` recursion over every
policy node and exhaustive rational fixed-subset enumeration.

Float comparisons use the stated tolerances below. Exact ``==`` tie semantics belong to
``finite_acquisition_policy`` itself and are tested on dyadic-exact inputs.
"""

import dataclasses
import importlib.util
import itertools
import math
import sys
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "bel_inference_to_decision.py"

TOL_MOMENTS = 1e-10  # Schur conditioning, affine inverse and four-point moments
TOL_VALUE = 1e-11  # float64 policy arithmetic against exact rational values
TOL_NEAR = 1e-9  # values closer than this are "near ties" in the float/rational comparison


def _load_example():
    spec = importlib.util.spec_from_file_location("bel_inference_to_decision", _EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


app = _load_example()

_CACHE = None


def _fixture():
    """Fit once, predict once (two cases), build both decisions."""
    global _CACHE
    if _CACHE is None:
        fx = SimpleNamespace()
        fx.rng_before = np.random.get_state()
        fx.X, fx.Y = app.training_design()
        fx.bel = app.fit_bel(fx.X, fx.Y)
        fx.mean, fx.cov = app.predict_posterior(fx.bel, app.OBSERVATIONS, app.NOISE)
        fx.post = app.posterior_atoms(fx.bel, fx.mean, fx.cov)
        fx.likelihood = app.sign_label_likelihood(fx.post.labels)
        fx.decisions = [
            app.decide_case(fx.post.atoms[i], fx.likelihood) for i in range(len(app.OBSERVATIONS))
        ]
        fx.rng_after = np.random.get_state()
        _CACHE = fx
    return _CACHE


def schur_posterior(bel, x_obs, noise):
    """Gaussian conditioning of the canonical Y on X from fitted arrays (not mvn_inference).

    Joint model, as documented for the MVN mode: ``X = g Y + error`` with the
    least-squares ``g`` (entries below 1e-8 and a mean below 1e-8 are set to zero, the
    published numerical convention), modelling-error covariance, plus projected noise
    ``noise * R.T @ R``. The posterior is the Schur complement.
    """
    X = np.asarray(bel.X_f, dtype=float)
    Y = np.asarray(bel.Y_f, dtype=float)
    n = X.shape[0]
    mean_y = Y.mean(axis=0)
    mean_y = np.where(np.abs(mean_y) < 1e-8, 0.0, mean_y)
    centred = Y - Y.mean(axis=0)
    cov_y = centred.T @ centred / (n - 1)
    g = np.linalg.solve(Y.T @ Y, Y.T @ X).T
    g = np.where(np.abs(g) < 1e-8, 0.0, g)
    resid = X - Y @ g.T
    mean_err = resid.mean(axis=0)
    err = resid - mean_err
    cov_model = err.T @ err / (n - 1)
    rot = bel.regression_model.x_rotations_
    cov_noise = noise * rot.T @ rot
    s12 = cov_y @ g.T
    s21 = g @ cov_y
    s22 = g @ cov_y @ g.T + cov_noise + cov_model
    mean_x = mean_y @ g.T + mean_err
    gain = np.linalg.solve(s22, s12.T).T
    post_mean = mean_y + gain @ (np.asarray(x_obs, dtype=float) - mean_x)
    post_cov = cov_y - gain @ s21
    return post_mean, post_cov


def hypot_energy_score(atoms, truth):
    """Explicit double loop with math.hypot / fsum, uniform weights."""
    n = len(atoms)
    first = math.fsum(math.hypot(a[0] - truth[0], a[1] - truth[1]) for a in atoms) / n
    second = math.fsum(math.hypot(a[0] - b[0], a[1] - b[1]) for a in atoms for b in atoms) / (n * n)
    return first - 0.5 * second


def hypot_action_losses(atoms, centres):
    return [[math.hypot(a[0] - c[0], a[1] - c[1]) ** 2 for c in centres] for a in atoms]


def oracle_policy(losses, outcomes, weights, costs, horizon):
    """Independent exact recursion over every node, with Fraction arithmetic."""
    loss = [[Fraction(x) for x in row] for row in np.asarray(losses).tolist()]
    w = [Fraction(x) for x in np.asarray(weights).tolist()]
    out = np.asarray(outcomes).tolist()
    cost = [Fraction(x) for x in np.asarray(costs).tolist()]
    n_actions, n_candidates = len(loss[0]), len(cost)

    def solve(idx, acquired, h):
        total = sum(w[i] for i in idx)
        risks = [sum(w[i] * loss[i][a] for i in idx) / total for a in range(n_actions)]
        stop = min(risks)
        cands = {}
        if h > 0:
            for c in range(n_candidates):
                if c in acquired:
                    continue
                branches = {}
                cont = Fraction(0)
                for label in sorted({out[i][c] for i in idx}):
                    sub = [i for i in idx if out[i][c] == label]
                    prob = sum(w[i] for i in sub) / total
                    child = solve(sub, (*acquired, c), h - 1)
                    branches[label] = (prob, child)
                    cont += prob * child.value
                cands[c] = SimpleNamespace(
                    cost=cost[c], cont=cont, value=cost[c] + cont, branches=branches
                )
        best = min([stop, *(e.value for e in cands.values())])
        return SimpleNamespace(
            value=best,
            stop=stop,
            risks=risks,
            cands=cands,
            support=tuple(idx),
            acquired=acquired,
            horizon=h,
        )

    return solve([i for i, x in enumerate(w) if x > 0], (), horizon)


def assert_node_matches(tc, node, ref):
    """Compare every node: values, probabilities, supports, and near-tie-aware choice sets."""
    tc.assertEqual(node.support, ref.support)
    tc.assertEqual(node.acquired, ref.acquired)
    tc.assertEqual(node.horizon, ref.horizon)
    tc.assertAlmostEqual(node.value, float(ref.value), delta=TOL_VALUE)
    tc.assertAlmostEqual(node.stop_risk, float(ref.stop), delta=TOL_VALUE)
    near_actions = {a for a, r in enumerate(ref.risks) if float(r - ref.stop) <= TOL_NEAR}
    tc.assertTrue(set(node.terminal_actions) <= near_actions)
    if len(near_actions) == 1:
        tc.assertEqual(set(node.terminal_actions), near_actions)
    near_choices = set()
    if float(ref.stop - ref.value) <= TOL_NEAR:
        near_choices.add("stop")
    near_choices |= {c for c, e in ref.cands.items() if float(e.value - ref.value) <= TOL_NEAR}
    tc.assertTrue(set(node.optimal_choices) <= near_choices)
    if len(near_choices) == 1:
        tc.assertEqual(set(node.optimal_choices), near_choices)
    tc.assertEqual(set(node.candidate_values), set(ref.cands))
    for c, evidence in node.candidate_values.items():
        expected = ref.cands[c]
        tc.assertAlmostEqual(evidence.immediate_cost, float(expected.cost), delta=TOL_VALUE)
        tc.assertAlmostEqual(
            evidence.expected_continuation_value, float(expected.cont), delta=TOL_VALUE
        )
        tc.assertAlmostEqual(evidence.value, float(expected.value), delta=TOL_VALUE)
        tc.assertEqual(set(evidence.branches), set(expected.branches))
        for label, branch in evidence.branches.items():
            prob, child = expected.branches[label]
            tc.assertAlmostEqual(branch.probability, float(prob), delta=TOL_VALUE)
            assert_node_matches(tc, branch.node, child)


def best_fixed_subset_value(cost, max_size):
    """Exhaustive rational value of non-adaptive subsets of the toy 0/1 guess problem."""
    tuples = list(itertools.product((0, 1), repeat=3))

    def state(t):
        return t[1] if t[0] == 0 else t[2]

    best = None
    for size in range(max_size + 1):
        for subset in itertools.combinations(range(3), size):
            groups = {}
            for t in tuples:
                groups.setdefault(tuple(t[c] for c in subset), []).append(state(t))
            risk = sum(Fraction(min(g.count(0), g.count(1)), 8) for g in groups.values())
            value = risk + sum(cost[c] for c in subset)
            best = value if best is None else min(best, value)
    return best


class FittedFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = _fixture()

    def test_training_design_is_the_sixteen_sign_rows(self):
        fx = self.fx
        self.assertEqual(fx.X.shape, (16, 2))
        self.assertEqual(fx.Y.shape, (16, 2))
        self.assertEqual(len({tuple(r) for r in np.column_stack([fx.X, fx.Y - [0, 0]])}), 16)
        u, v = fx.X.T
        np.testing.assert_array_equal(fx.Y[:, 0] - 2 * u, np.sign(fx.Y[:, 0] - 2 * u))
        np.testing.assert_array_equal(np.abs(fx.Y[:, 1] - 3 * v), np.full(16, 2.0))

    def test_no_prng_state_is_touched_by_fit_predict_atoms_and_decisions(self):
        before, after = self.fx.rng_before, self.fx.rng_after
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_posterior_matches_independent_schur_conditioning(self):
        fx = self.fx
        x_canonical = fx.bel.transform(X=app.OBSERVATIONS)
        np.testing.assert_allclose(x_canonical, fx.bel.X_obs_f, atol=1e-12)
        for case in range(len(app.OBSERVATIONS)):
            mean, cov = schur_posterior(fx.bel, x_canonical[case], app.NOISE)
            np.testing.assert_allclose(fx.mean[case], mean, atol=TOL_MOMENTS)
            np.testing.assert_allclose(fx.cov[case], cov, atol=TOL_MOMENTS)

    def test_noise_enters_through_the_projected_covariance(self):
        fx = self.fx
        x_canonical = fx.bel.transform(X=app.OBSERVATIONS)
        quiet, quiet_cov = schur_posterior(fx.bel, x_canonical[1], 0.0)
        noisy, noisy_cov = schur_posterior(fx.bel, x_canonical[1], app.NOISE)
        self.assertGreater(np.trace(noisy_cov), np.trace(quiet_cov) + 1e-3)
        np.testing.assert_allclose(fx.cov[1], noisy_cov, atol=TOL_MOMENTS)

    def test_four_points_match_the_canonical_first_two_moments(self):
        post = self.fx.post
        for case in range(len(app.OBSERVATIONS)):
            points = post.canonical_atoms[case]
            self.assertEqual(points.shape, (4, 2))
            centred = points - points.mean(axis=0)
            np.testing.assert_allclose(points.mean(axis=0), post.mean[case], atol=TOL_MOMENTS)
            np.testing.assert_allclose(
                centred.T @ centred / 4.0, post.covariance[case], atol=TOL_MOMENTS
            )
            # Bounded support: a finite approximation, not the Gaussian and not its tails.
            self.assertTrue(
                np.all(np.abs(centred) <= 1.5 * np.sqrt(np.diag(post.covariance[case])))
            )

    def test_inverse_is_the_affine_map_fixed_by_the_training_rows(self):
        fx = self.fx
        design = np.hstack([fx.bel.Y_f, np.ones((16, 1))])
        coef = np.linalg.lstsq(design, fx.Y, rcond=None)[0]
        np.testing.assert_allclose(design @ coef, fx.Y, atol=TOL_MOMENTS)
        canonical = np.concatenate(
            [fx.post.canonical_atoms, np.ones(fx.post.canonical_atoms.shape[:2] + (1,))], axis=-1
        )
        np.testing.assert_allclose(fx.post.atoms, canonical @ coef, atol=TOL_MOMENTS)

    def test_atoms_are_in_original_units_and_differ_between_observations(self):
        atoms = self.fx.post.atoms
        self.assertEqual(atoms.shape, (2, 4, 2))
        self.assertTrue(np.all(np.isfinite(atoms)))
        self.assertGreater(np.max(np.abs(atoms[0] - atoms[1])), 0.1)

    def test_main_losses_are_reconstructed_from_original_unit_atoms(self):
        fx = self.fx
        for case, decision in enumerate(fx.decisions):
            direct = hypot_action_losses(fx.post.atoms[case], app.ACTION_CENTRES)
            model = decision.model
            self.assertEqual(model.losses.shape, (32, 2))
            for row in range(32):
                for action in range(2):
                    self.assertAlmostEqual(
                        model.losses[row, action],
                        direct[model.atom_index[row]][action],
                        delta=1e-12,
                    )
            # Losses vary across atoms: the decision depends on actual atom values.
            self.assertGreater(np.ptp(model.losses[:, 0]), 1.0)

    def test_joint_model_expansion_and_positive_mass(self):
        model = self.fx.decisions[0].model
        self.assertEqual(model.weights.shape, (32,))
        self.assertEqual(np.count_nonzero(model.weights), 16)
        self.assertEqual(math.fsum(model.weights), 1.0)
        np.testing.assert_array_equal(model.outcomes, app.OUTCOME_TUPLES[model.tuple_index])
        for row in range(32):
            label = int(self.fx.post.labels[model.atom_index[row]])
            a, b, c = model.outcomes[row]
            state = b if a == 0 else c
            self.assertEqual(model.weights[row], 1 / 16 if state == label else 0.0)
        np.testing.assert_array_equal(self.fx.likelihood.sum(axis=1), np.ones(4))

    def test_policy_matches_independent_rational_recursion_at_every_node(self):
        for decision in self.fx.decisions:
            model = decision.model
            ref = oracle_policy(model.losses, model.outcomes, model.weights, model.costs, 2)
            assert_node_matches(self, decision.root, ref)

    def test_observation_changes_expected_losses_and_the_decision(self):
        first, second = self.fx.decisions
        # Same likelihood, costs and centres; only the current observation differs.
        np.testing.assert_array_equal(first.model.costs, second.model.costs)
        np.testing.assert_array_equal(first.model.outcomes, second.model.outcomes)
        np.testing.assert_array_equal(first.model.weights, second.model.weights)
        self.assertEqual(first.root.optimal_choices, (0,))  # acquire A first
        self.assertEqual(second.root.optimal_choices, ("stop",))
        self.assertEqual(second.root_bayes_actions, (1,))  # centre (1, 0)
        self.assertEqual(second.root.terminal_actions, (1,))
        gap_first = abs(first.root_expected_losses[0] - first.root_expected_losses[1])
        gap_second = abs(second.root_expected_losses[0] - second.root_expected_losses[1])
        self.assertLess(gap_first, 1e-6)  # symmetric observation: near-tie, float luck
        self.assertGreater(gap_second, 1.0)
        self.assertLess(first.root.value, first.root.stop_risk)
        self.assertEqual(second.root.value, second.root.stop_risk)

    def test_public_root_expected_losses_match_direct_atom_average(self):
        fx = self.fx
        for case, decision in enumerate(fx.decisions):
            direct = np.array(hypot_action_losses(fx.post.atoms[case], app.ACTION_CENTRES))
            np.testing.assert_allclose(
                decision.root_expected_losses, direct.mean(axis=0), atol=1e-12
            )
            self.assertAlmostEqual(
                decision.root_bayes_risk, float(np.min(direct.mean(axis=0))), delta=1e-12
            )
            self.assertAlmostEqual(decision.root.stop_risk, decision.root_bayes_risk, delta=1e-12)

    def test_realized_path_a_then_b_and_a_then_c(self):
        fx = self.fx
        decision = fx.decisions[0]
        for realized, last, state in (((0, 1, 0), 1, 1), ((1, 0, 0), 2, 0)):
            path = app.run_policy(decision.model, decision.root, dict(enumerate(realized)))
            self.assertEqual([s.candidate for s in path.steps], [0, last])
            self.assertEqual([s.outcome for s in path.steps], [realized[0], realized[last]])
            self.assertEqual(path.total_cost, 2.0)
            self.assertEqual(path.steps[0].probability, 0.5)
            self.assertEqual(path.steps[0].choices_before, (0,))
            support = np.array(path.final_node.support)
            self.assertEqual(len(support), 4)
            atom_ids = decision.model.atom_index[support]
            self.assertTrue(np.all(fx.post.labels[atom_ids] == state))
            atoms = fx.post.atoms[0][np.unique(atom_ids)]
            direct = np.mean(hypot_action_losses(atoms, app.ACTION_CENTRES), axis=0)
            self.assertEqual(path.selected_action, int(np.argmin(direct)))
            self.assertEqual(path.terminal_actions, (int(np.argmin(direct)),))
            np.testing.assert_allclose(path.expected_losses, direct, atol=1e-12)
        # The two worlds end in different terminal actions: they depend on the atoms.
        a = app.run_policy(decision.model, decision.root, {0: 0, 1: 1, 2: 0})
        b = app.run_policy(decision.model, decision.root, {0: 1, 1: 0, 2: 0})
        self.assertNotEqual(a.selected_action, b.selected_action)

    def test_stop_branch_has_no_steps_and_the_bayes_action(self):
        decision = self.fx.decisions[1]
        path = app.run_policy(decision.model, decision.root, {0: 0, 1: 1, 2: 0})
        self.assertEqual(path.steps, ())
        self.assertEqual(path.total_cost, 0.0)
        self.assertEqual(path.selected_action, 1)

    def test_unrealizable_paths_raise_before_inventing_a_posterior(self):
        decision = self.fx.decisions[0]
        model, root = decision.model, decision.root
        with self.assertRaises(ValueError):
            app.follow_path(model, root, [(0, 2)])  # label that never occurs
        with self.assertRaises(ValueError):
            app.follow_path(model, root, [(0, 0), (0, 1)])  # reused candidate
        with self.assertRaises(ValueError):
            app.follow_path(model, root, [(0, 0), (1, 1), (2, 0)])  # horizon exhausted
        with self.assertRaises(ValueError):
            app.follow_path(model, root, [(7, 0)])  # not a candidate
        with self.assertRaises(ValueError):
            app.run_policy(model, root, {1: 0, 2: 0})  # outcome of A not supplied
        # A zero-mass outcome: A can only be 0 in this supplied joint model.
        states = app.hidden_state(app.OUTCOME_TUPLES)
        only_a0 = np.zeros((4, 8))
        for atom, label in enumerate(self.fx.post.labels):
            rows = (app.OUTCOME_TUPLES[:, 0] == 0) & (states == label)
            only_a0[atom, rows] = 0.5
        impossible = app.decide_case(self.fx.post.atoms[0], only_a0)
        branches = impossible.root.candidate_values[0].branches
        self.assertEqual(set(branches), {0})
        with self.assertRaises(ValueError):
            app.follow_path(impossible.model, impossible.root, [(0, 1)])

    def test_secondary_sign_tag_control_matches_the_known_rational_values(self):
        fx = self.fx
        losses = app.sign_tag_losses(fx.post.labels)
        model = app.expand_joint_model(losses, fx.likelihood, app.SIGN_TAG_COSTS)
        root = app.solve_policy(model)
        ref = oracle_policy(model.losses, model.outcomes, model.weights, model.costs, 2)
        assert_node_matches(self, root, ref)
        costs = [Fraction(1, 50)] * 3
        fixed = best_fixed_subset_value(costs, 2)
        self.assertEqual(fixed, Fraction(27, 100))
        self.assertAlmostEqual(root.value, 1 / 25, delta=TOL_VALUE)
        self.assertLess(root.value, float(fixed))
        self.assertEqual(root.optimal_choices, (0,))
        branches = root.candidate_values[0].branches
        self.assertEqual(branches[0].node.optimal_choices, (1,))  # A=0 -> B
        self.assertEqual(branches[1].node.optimal_choices, (2,))  # A=1 -> C
        self.assertAlmostEqual(root.stop_risk, 0.5, delta=TOL_VALUE)
        # The control table has no atom values in it: it is a pure function of the labels.
        self.assertTrue(np.all(losses == app.sign_tag_losses(np.array([0, 0, 1, 1]))))

    def test_energy_score_matches_the_hypot_double_loop(self):
        fx = self.fx
        truths = np.array([[1.0, 1.0], [3.0, -1.0]])
        for case in range(2):
            expected = hypot_energy_score(fx.post.atoms[case], truths[case])
            self.assertAlmostEqual(
                app.retrospective_score(fx.post.atoms[case], truths[case]), expected, delta=1e-12
            )

    def test_truth_never_changes_the_policy_but_costs_and_losses_do(self):
        fx = self.fx
        atoms = fx.post.atoms[1]
        baseline = fx.decisions[1]
        scores = [
            app.retrospective_score(atoms, truth) for truth in ([1.0, 1.0], [-3.0, 5.0], [0.0, 0.0])
        ]
        self.assertEqual(len(set(scores)), 3)
        again = app.decide_case(atoms, fx.likelihood)
        self.assertEqual(again.root.value, baseline.root.value)
        self.assertEqual(again.root.optimal_choices, baseline.root.optimal_choices)
        cheap = app.decide_case(atoms, fx.likelihood, costs=(0.1, 0.1, 0.1))
        self.assertEqual(cheap.root.optimal_choices, (0,))
        self.assertNotEqual(cheap.root.value, baseline.root.value)
        moved = app.decide_case(atoms, fx.likelihood, centres=np.array([[-1.0, 0.0], [3.0, 0.0]]))
        self.assertNotEqual(moved.root.stop_risk, baseline.root.stop_risk)
        self.assertNotEqual(moved.root_expected_losses[1], baseline.root_expected_losses[1])

    def test_costly_stop_horizon_zero_and_zero_cost_uninformative_ties(self):
        fx = self.fx
        atoms = fx.post.atoms[0]
        costly = app.decide_case(atoms, fx.likelihood, costs=(100.0, 100.0, 100.0))
        self.assertEqual(costly.root.optimal_choices, ("stop",))
        ref = oracle_policy(
            costly.model.losses, costly.model.outcomes, costly.model.weights, costly.model.costs, 2
        )
        assert_node_matches(self, costly.root, ref)

        flat = app.decide_case(atoms, fx.likelihood, horizon=0)
        self.assertEqual(flat.root.optimal_choices, ("stop",))
        self.assertEqual(len(flat.root.candidate_values), 0)
        self.assertEqual(flat.root.value, flat.root.stop_risk)

        base = fx.decisions[1].model
        free = dataclasses.replace(
            base, outcomes=np.zeros((32, 1), dtype=np.int64), costs=np.zeros(1)
        )
        node = app.solve_policy(free, horizon=1)
        self.assertEqual(node.optimal_choices, ("stop", 0))  # whole exact tie set kept
        path = app.run_policy(free, node, {0: 0})
        self.assertEqual(path.steps, ())  # stop is taken first on the exact tie

    def test_malformed_likelihood_is_rejected(self):
        fx = self.fx
        atoms = fx.post.atoms[0]
        good = fx.likelihood
        bad = [
            good[:3],
            good[:, :7],
            np.where(good > 0, np.nan, 0.0),
            np.where(good > 0, 0.5, -0.25),
            good * 0.5,
            good * 2.0,
            np.zeros((4, 8)),
            np.full((4, 8), 0.2),
        ]
        for likelihood in bad:
            with self.assertRaises(ValueError):
                app.decide_case(atoms, likelihood)
        with self.assertRaises(TypeError):
            app.decide_case(atoms, good.astype(str))

    def test_lost_positive_product_mass_is_detected_before_the_policy(self):
        tiny = np.zeros((4, 8))
        tiny[:, 0] = 1.0
        tiny[:, 1] = 5e-324  # 1/4 * 5e-324 underflows to zero
        self.assertEqual(math.fsum(tiny[0]), 1.0)
        with self.assertRaises(ValueError):
            app.expand_joint_model(np.zeros((4, 2)), tiny, app.CANDIDATE_COSTS)

    def test_malformed_losses_costs_and_atoms_are_rejected(self):
        fx = self.fx
        atoms = fx.post.atoms[0]
        for costs in ((1.0, 1.0), (1.0, -1.0, 1.0), (1.0, np.nan, 1.0), (np.inf, 1.0, 1.0)):
            with self.assertRaises(ValueError):
                app.decide_case(atoms, fx.likelihood, costs=costs)
        with self.assertRaises(ValueError):
            app.decide_case(atoms, fx.likelihood, horizon=3)
        with self.assertRaises(ValueError):
            app.action_losses(np.where(atoms > 100, 0.0, np.nan), app.ACTION_CENTRES)
        with self.assertRaises(ValueError):
            app.action_losses(atoms, np.ones((2, 3)))
        with self.assertRaises(ValueError):
            app.expand_joint_model(np.full((4, 2), np.inf), fx.likelihood, app.CANDIDATE_COSTS)
        with self.assertRaises(ValueError):
            app.expand_joint_model(np.zeros((4, 0)), fx.likelihood, app.CANDIDATE_COSTS)

    def test_malformed_covariance_is_rejected_without_repair(self):
        fx = self.fx
        cov = fx.cov[0]
        for bad in (
            np.array([[1.0, 0.0], [0.0, np.nan]]),
            np.array([[1.0, 1.0], [-1.0, 1.0]]),
            np.array([[1.0, 2.0], [2.0, 1.0]]),
            np.array([[1.0, 0.0], [0.0, 0.0]]),
            np.array([[-1.0, 0.0], [0.0, 1.0]]),
            np.ones((2, 3)),
        ):
            with self.assertRaises(ValueError):
                app.canonical_cholesky(bad)
        nudged = cov.copy()
        nudged[0, 1] += 0.5 * app.SYMMETRY_TOLERANCE
        app.canonical_cholesky(nudged)  # within the disclosed tolerance: accepted as is
        nudged[0, 1] += 2.0 * app.SYMMETRY_TOLERANCE
        with self.assertRaises(ValueError):
            app.canonical_cholesky(nudged)
        stacked = np.stack([fx.cov[0], np.array([[1.0, 0.0], [0.0, -1.0]])])
        with self.assertRaises(ValueError):
            app.posterior_atoms(fx.bel, fx.mean, stacked)
        with self.assertRaises(ValueError):
            app.posterior_atoms(fx.bel, fx.mean[:1], fx.cov)
        with self.assertRaises(ValueError):
            app.posterior_atoms(fx.bel, np.where(fx.mean > 100, 0.0, np.nan), fx.cov)

    def test_inputs_and_global_random_state_are_not_modified(self):
        fx = self.fx
        atoms, likelihood = fx.post.atoms[0].copy(), fx.likelihood.copy()
        centres, mean, cov = app.ACTION_CENTRES.copy(), fx.mean.copy(), fx.cov.copy()
        state = np.random.get_state()
        decision = app.decide_case(atoms, likelihood, centres)
        app.run_policy(decision.model, decision.root, {0: 0, 1: 1, 2: 0})
        app.retrospective_score(atoms, np.array([1.0, 1.0]))
        app.posterior_atoms(fx.bel, mean, cov)
        after = np.random.get_state()
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual((state[0], state[2:]), (after[0], after[2:]))
        np.testing.assert_array_equal(atoms, fx.post.atoms[0])
        np.testing.assert_array_equal(likelihood, fx.likelihood)
        np.testing.assert_array_equal(centres, app.ACTION_CENTRES)
        np.testing.assert_array_equal(mean, fx.mean)
        np.testing.assert_array_equal(cov, fx.cov)

    def test_module_is_import_safe_and_exposes_an_explicit_main(self):
        self.assertTrue(callable(app.main))
        self.assertFalse(app.OBSERVATIONS.flags.writeable)


if __name__ == "__main__":
    unittest.main()
