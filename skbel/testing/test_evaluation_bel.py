"""``sample_bel_posterior`` on small, seeded, fitted BEL models in the mvn, kde and tm modes.

The models are fitted on synthetic linear-Gaussian data whose targets sit far from zero and
have unequal scales, so that draws in original units are easy to tell from standardized
ones. Parity is checked against a twin model that is given the same data, the same seed and a
direct call to the public ``BEL.predict``.
"""

import contextlib
import unittest
from unittest import mock

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skbel import BEL
from skbel.evaluation import (
    decision_risks,
    event_brier_scores,
    event_probabilities,
    sample_bel_posterior,
    score_samples,
)
from skbel.metrics import case_rng

MODES = ("mvn", "kde", "tm")
SEED = 2024
N_DRAWS = 25  # differs from the 60 training rows, so tm draws are random
Y_OFFSET = np.array([100.0, -3.0])
Y_SCALE = np.array([5.0, 0.5])


def _features(z, rng):
    x = np.column_stack([z[:, 0] + 0.3 * z[:, 1], z[:, 1] - 0.2 * z[:, 0], z.sum(axis=1)])
    return x + 0.2 * rng.normal(size=x.shape)


def _data():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(60, 2))
    x = _features(z, rng)
    # Held-out cases well inside the training support, where the kernel estimate is reliable.
    z_new = np.array([[0.5, -0.3], [-0.8, 0.6], [0.2, 0.9], [-0.4, -0.7]])
    x_new = _features(z_new, rng)
    return x, z * Y_SCALE + Y_OFFSET, x_new, z_new * Y_SCALE + Y_OFFSET


X_TRAIN, Y_TRAIN, X_CASES, Y_CASES = _data()


def _make(mode, *, seed=SEED, scaled=True):
    """A fitted BEL; ``scaled=False`` leaves both pre-processing pipelines as pass-through."""
    if scaled:
        x_pre = Pipeline([("scale", StandardScaler())])
        y_pre = Pipeline([("scale", StandardScaler())])
    else:
        x_pre = y_pre = None
    bel = BEL(
        mode=mode,
        X_pre_processing=x_pre,
        Y_pre_processing=y_pre,
        regression_model=CCA(n_components=2),
        random_state=seed,
    )
    return bel.fit(X_TRAIN, Y_TRAIN)


def _direct(bel, observations, n_draws=N_DRAWS, **kwargs):
    """The public ``BEL.predict`` call that ``sample_bel_posterior`` is documented to make."""
    return bel.predict(
        np.array(observations, dtype=float),
        n_posts=n_draws,
        return_samples=True,
        inverse_transform=True,
        dtype="float64",
        **kwargs,
    )


def _within(testcase, actual, expected, tolerance):
    """Every entry of ``actual`` is within the (broadcast) ``tolerance`` of ``expected``."""
    gap = np.abs(np.asarray(actual) - np.asarray(expected))
    testcase.assertTrue(np.all(gap <= tolerance), f"largest gap {gap.max()} vs {tolerance}")


def _state():
    return np.random.get_state()


def _same_state(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and tuple(a[2:]) == tuple(b[2:])


def _fitted_snapshot(bel):
    return {
        "X_f": bel.X_f.copy(),
        "Y_f": bel.Y_f.copy(),
        "x_weights": bel.regression_model.x_weights_.copy(),
        "y_weights": bel.regression_model.y_weights_.copy(),
        "x_mean": bel.X_pre_processing.named_steps["scale"].mean_.copy(),
        "y_mean": bel.Y_pre_processing.named_steps["scale"].mean_.copy(),
    }


class _PredictSpy:
    """Stand-in for ``predict`` that records its call and returns a prepared array."""

    def __init__(self, output):
        self.output = output
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.output(args, kwargs) if callable(self.output) else self.output


def _good_output(cases, n_draws=N_DRAWS, targets=2, dtype=np.float64):
    return np.random.default_rng(1).normal(size=(cases, n_draws, targets)).astype(dtype)


def _call_with_output(bel, output, observations=X_CASES, n_draws=N_DRAWS):
    spy = _PredictSpy(output)
    with mock.patch.object(bel, "predict", side_effect=spy):
        return sample_bel_posterior(bel, observations, n_draws), spy


class ParityWithPublicPredictTest(unittest.TestCase):
    def test_draws_equal_a_direct_predict_on_a_twin_model(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                via_function = sample_bel_posterior(_make(mode), X_CASES, N_DRAWS)
                twin = _direct(_make(mode), X_CASES)
                self.assertEqual(via_function.shape, (len(X_CASES), N_DRAWS, 2))
                self.assertEqual(via_function.dtype, np.float64)
                self.assertTrue(np.all(np.isfinite(via_function)))
                np.testing.assert_array_equal(via_function, twin)

    def test_draws_are_in_original_target_units(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                samples = sample_bel_posterior(_make(mode), X_CASES, N_DRAWS)
                centre = samples.mean(axis=(0, 1))
                spread = samples.std(axis=1).mean(axis=0)
                # Standardized draws would be centred near 0 with spread near 1.
                _within(self, centre, Y_CASES.mean(axis=0), 2.0 * Y_SCALE)
                self.assertTrue(np.all(spread < 1.5 * Y_SCALE))
                self.assertTrue(np.all(spread > 0.02 * Y_SCALE))
                # Each case's draws are centred near that case's realized targets.
                _within(self, samples.mean(axis=1), Y_CASES, 2.0 * Y_SCALE)

    def test_mvn_draws_equal_the_documented_stream_of_each_row(self):
        bel = _make("mvn")
        samples = sample_bel_posterior(bel, X_CASES, N_DRAWS)
        canonical = np.stack(
            [
                case_rng(SEED, row, "skbel.learning.BEL.random_sample").multivariate_normal(
                    bel.posterior_mean[row], bel.posterior_covariance[row], size=N_DRAWS
                )
                for row in range(len(X_CASES))
            ]
        )
        np.testing.assert_array_equal(samples, bel.inverse_transform(canonical))

    def test_predict_is_called_once_with_the_documented_arguments(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                bel = _make(mode)
                observations = X_CASES.copy()
                _, spy = _call_with_output(bel, _good_output(len(X_CASES)), observations)
                self.assertEqual(len(spy.calls), 1)
                args, kwargs = spy.calls[0]
                self.assertEqual(len(args), 1)
                np.testing.assert_array_equal(args[0], X_CASES)
                self.assertEqual(kwargs["n_posts"], N_DRAWS)
                self.assertIs(kwargs["return_samples"], True)
                self.assertIs(kwargs["inverse_transform"], True)
                self.assertEqual(kwargs["dtype"], "float64")
                self.assertIsNone(kwargs["mode"])
                self.assertIsNone(kwargs["noise"])

    def test_mode_and_noise_are_passed_through_unchanged(self):
        bel = _make("mvn")
        _, spy = _call_with_output(bel, _good_output(len(X_CASES)))
        self.assertIsNone(spy.calls[0][1]["mode"])
        spy = _PredictSpy(_good_output(len(X_CASES)))
        with mock.patch.object(bel, "predict", side_effect=spy):
            sample_bel_posterior(bel, X_CASES, N_DRAWS, mode="kde", noise=0.5)
        self.assertEqual(spy.calls[0][1]["mode"], "kde")
        self.assertEqual(spy.calls[0][1]["noise"], 0.5)

    def test_observations_array_and_dtype_variants_give_the_same_draws(self):
        reference = sample_bel_posterior(_make("mvn"), X_CASES, N_DRAWS)
        as_list = sample_bel_posterior(_make("mvn"), X_CASES.tolist(), N_DRAWS)
        as_float32 = sample_bel_posterior(_make("mvn"), X_CASES.astype(np.float32), N_DRAWS)
        np.testing.assert_array_equal(as_list, reference)
        np.testing.assert_allclose(as_float32, reference, rtol=0, atol=1e-3)
        ints = np.round(X_CASES * 4).astype(int)
        np.testing.assert_array_equal(
            sample_bel_posterior(_make("mvn"), ints, N_DRAWS),
            _direct(_make("mvn"), ints.astype(float)),
        )

    def test_integer_draw_counts_of_any_integer_type_are_accepted(self):
        reference = sample_bel_posterior(_make("mvn"), X_CASES, N_DRAWS)
        for count in (np.int64(N_DRAWS), np.int32(N_DRAWS), np.uint8(N_DRAWS)):
            with self.subTest(count=type(count).__name__):
                np.testing.assert_array_equal(
                    sample_bel_posterior(_make("mvn"), X_CASES, count), reference
                )

    def test_a_single_case_is_passed_as_one_row(self):
        samples = sample_bel_posterior(_make("mvn"), X_CASES[:1], N_DRAWS)
        self.assertEqual(samples.shape, (1, N_DRAWS, 2))
        with self.assertRaises(ValueError):
            sample_bel_posterior(_make("mvn"), X_CASES[0], N_DRAWS)

    def test_a_single_draw_is_allowed(self):
        samples = sample_bel_posterior(_make("mvn"), X_CASES, 1)
        self.assertEqual(samples.shape, (len(X_CASES), 1, 2))


class NoRefitAndNoGlobalMutationTest(unittest.TestCase):
    def test_the_fitted_model_is_not_refitted(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                bel = _make(mode)
                before = _fitted_snapshot(bel)
                scalers = [
                    step
                    for pipeline in (bel.X_pre_processing, bel.Y_pre_processing)
                    for step in pipeline.named_steps.values()
                ]
                with contextlib.ExitStack() as stack:
                    for target, name in (
                        (BEL, "fit"),
                        (BEL, "fit_transform"),
                        (CCA, "fit"),
                        (CCA, "fit_transform"),
                        (Pipeline, "fit"),
                        (Pipeline, "fit_transform"),
                    ):
                        stack.enter_context(
                            mock.patch.object(
                                target, name, side_effect=AssertionError(f"{name} called")
                            )
                        )
                    for scaler in scalers:  # the model's own fitted steps
                        for name in ("fit", "fit_transform", "partial_fit"):
                            stack.enter_context(
                                mock.patch.object(
                                    scaler, name, side_effect=AssertionError(f"{name} called")
                                )
                            )
                    sample_bel_posterior(bel, X_CASES, N_DRAWS)
                after = _fitted_snapshot(bel)
                for key, value in before.items():
                    np.testing.assert_array_equal(after[key], value, err_msg=key)

    def test_numpy_global_random_state_is_not_read_or_modified(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                bel = _make(mode)
                before = _state()
                sample_bel_posterior(bel, X_CASES, N_DRAWS)
                self.assertTrue(_same_state(before, _state()))

    def test_draws_do_not_depend_on_the_global_random_state(self):
        saved = _state()
        try:
            for mode in MODES:
                with self.subTest(mode=mode):
                    np.random.seed(1)
                    first = sample_bel_posterior(_make(mode), X_CASES, N_DRAWS)
                    np.random.seed(987654)
                    second = sample_bel_posterior(_make(mode), X_CASES, N_DRAWS)
                    np.testing.assert_array_equal(first, second)
        finally:
            np.random.set_state(saved)

    def test_observations_given_by_the_caller_are_not_modified(self):
        bel = _make("mvn")
        observations = X_CASES.copy()

        def mutate(args, kwargs):
            args[0][...] = 0.0  # a predict that edits its argument in place
            return _good_output(len(observations))

        _call_with_output(bel, mutate, observations)
        np.testing.assert_array_equal(observations, X_CASES)

    def test_returned_draws_are_independent_of_the_model(self):
        bel = _make("mvn")
        first = sample_bel_posterior(bel, X_CASES, N_DRAWS)
        kept = first.copy()
        first[...] = np.nan
        again = sample_bel_posterior(bel, X_CASES, N_DRAWS)
        np.testing.assert_array_equal(again, kept)


class StreamsAndRepeatabilityTest(unittest.TestCase):
    def test_same_call_with_the_same_seed_repeats_the_draws(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                bel = _make(mode)
                first = sample_bel_posterior(bel, X_CASES, N_DRAWS)
                again = sample_bel_posterior(bel, X_CASES, N_DRAWS)
                np.testing.assert_array_equal(first, again)
                fresh = sample_bel_posterior(_make(mode), X_CASES, N_DRAWS)
                np.testing.assert_array_equal(first, fresh)

    def test_another_seed_gives_other_draws(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                a = sample_bel_posterior(_make(mode, seed=SEED), X_CASES, N_DRAWS)
                b = sample_bel_posterior(_make(mode, seed=SEED + 1), X_CASES, N_DRAWS)
                self.assertFalse(np.allclose(a, b))

    def test_changing_the_seed_on_the_model_changes_the_next_draws(self):
        bel = _make("mvn")
        a = sample_bel_posterior(bel, X_CASES, N_DRAWS)
        bel.seed = SEED + 5
        b = sample_bel_posterior(bel, X_CASES, N_DRAWS)
        bel.seed = SEED
        c = sample_bel_posterior(bel, X_CASES, N_DRAWS)
        self.assertFalse(np.allclose(a, b))
        np.testing.assert_array_equal(a, c)

    def test_an_unset_seed_is_drawn_once_stored_and_reusable(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                bel = _make(mode, seed=None)
                self.assertIsNone(bel.seed)
                before = _state()
                first = sample_bel_posterior(bel, X_CASES, N_DRAWS)
                self.assertTrue(_same_state(before, _state()))
                self.assertIsInstance(bel.seed, int)
                self.assertGreaterEqual(bel.seed, 0)
                np.testing.assert_array_equal(first, sample_bel_posterior(bel, X_CASES, N_DRAWS))
                replay = sample_bel_posterior(_make(mode, seed=bel.seed), X_CASES, N_DRAWS)
                np.testing.assert_array_equal(first, replay)

    def test_a_case_draws_from_the_stream_of_its_row_position(self):
        for mode in ("mvn", "kde"):
            with self.subTest(mode=mode):
                bel = _make(mode)
                full = sample_bel_posterior(bel, X_CASES, N_DRAWS)
                # Leading rows keep their streams, so a prefix reproduces them.
                prefix = sample_bel_posterior(_make(mode), X_CASES[:2], N_DRAWS)
                np.testing.assert_array_equal(prefix, full[:2])
                # Moving a case to another position gives it another stream: different draws,
                # but the same posterior, so the same centre.
                swapped = sample_bel_posterior(_make(mode), X_CASES[[1, 0]], N_DRAWS)
                self.assertFalse(np.array_equal(swapped[0], full[0]))
                self.assertFalse(np.array_equal(swapped[1], full[1]))
                spread = Y_SCALE * 0.5
                _within(self, swapped[0].mean(axis=0), full[1].mean(axis=0), 3.0 * spread)

    def test_a_case_alone_does_not_repeat_its_draws_from_the_batch(self):
        for mode in ("mvn", "kde"):
            with self.subTest(mode=mode):
                full = sample_bel_posterior(_make(mode), X_CASES, N_DRAWS)
                alone = sample_bel_posterior(_make(mode), X_CASES[2:3], N_DRAWS)
                self.assertFalse(np.array_equal(alone[0], full[2]))
                self.assertTrue(
                    np.array_equal(
                        alone[0],
                        sample_bel_posterior(
                            _make(mode), np.vstack([X_CASES[2:3], X_CASES[:1]]), N_DRAWS
                        )[0],
                    )
                )

    def test_identical_rows_get_independent_draws(self):
        observations = np.vstack([X_CASES[0], X_CASES[1], X_CASES[0]])
        for mode in ("mvn", "kde"):
            with self.subTest(mode=mode):
                samples = sample_bel_posterior(_make(mode), observations, N_DRAWS)
                self.assertFalse(np.allclose(samples[0], samples[2]))

    def test_tm_with_as_many_draws_as_training_rows_is_deterministic(self):
        n_train = len(X_TRAIN)
        a = sample_bel_posterior(_make("tm", seed=1), X_CASES[:2], n_train)
        b = sample_bel_posterior(_make("tm", seed=2), X_CASES[:2], n_train)
        # The draws are a map of the training samples: the seed does not enter.
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)
        # They are not independent across rows: identical rows get identical draws.
        repeated = sample_bel_posterior(
            _make("tm", seed=3), np.vstack([X_CASES[0], X_CASES[0]]), n_train
        )
        np.testing.assert_allclose(repeated[0], repeated[1], rtol=0, atol=1e-9)

    def test_tm_with_another_draw_count_is_random_and_seeded(self):
        a = sample_bel_posterior(_make("tm", seed=1), X_CASES[:2], N_DRAWS)
        b = sample_bel_posterior(_make("tm", seed=2), X_CASES[:2], N_DRAWS)
        self.assertFalse(np.allclose(a, b))
        repeated = sample_bel_posterior(
            _make("tm", seed=1), np.vstack([X_CASES[0], X_CASES[0]]), N_DRAWS
        )
        self.assertFalse(np.allclose(repeated[0], repeated[1]))


class ModeNoiseAndStateTest(unittest.TestCase):
    def test_mode_none_keeps_the_model_mode_and_a_given_mode_is_stored(self):
        bel = _make("mvn")
        sample_bel_posterior(bel, X_CASES, N_DRAWS)
        self.assertEqual(bel.mode, "mvn")
        override = sample_bel_posterior(bel, X_CASES, N_DRAWS, mode="kde")
        self.assertEqual(bel.mode, "kde")
        np.testing.assert_array_equal(override, _direct(_make("mvn"), X_CASES, mode="kde"))
        again = sample_bel_posterior(bel, X_CASES, N_DRAWS)
        self.assertEqual(bel.mode, "kde")
        np.testing.assert_array_equal(again, override)

    def test_modes_give_different_posteriors_of_the_same_scale(self):
        mvn = sample_bel_posterior(_make("mvn"), X_CASES, N_DRAWS)
        kde = sample_bel_posterior(_make("kde"), X_CASES, N_DRAWS)
        self.assertFalse(np.allclose(mvn, kde))
        _within(self, kde.mean(axis=1), mvn.mean(axis=1), 3.0 * Y_SCALE)

    def test_unknown_modes_are_rejected_before_predict(self):
        for bad in ("gaussian", "", "MVN", 3, ["mvn"]):
            with self.subTest(bad=bad):
                bel = _make("mvn")
                spy = _PredictSpy(_good_output(len(X_CASES)))
                with (
                    mock.patch.object(bel, "predict", side_effect=spy),
                    self.assertRaises(ValueError),
                ):
                    sample_bel_posterior(bel, X_CASES, N_DRAWS, mode=bad)
                self.assertEqual(spy.calls, [])
                self.assertEqual(bel.mode, "mvn")

    def test_mvn_noise_widens_the_posterior_and_is_stored(self):
        quiet = _make("mvn")
        loud = _make("mvn")
        a = sample_bel_posterior(quiet, X_CASES, N_DRAWS, noise=0.01)
        b = sample_bel_posterior(loud, X_CASES, N_DRAWS, noise=50.0)
        self.assertEqual(quiet.noise, 0.01)
        self.assertEqual(loud.noise, 50.0)
        self.assertFalse(np.allclose(a, b))
        self.assertGreater(b.std(axis=1).mean(), a.std(axis=1).mean())
        np.testing.assert_array_equal(b, _direct(_make("mvn"), X_CASES, noise=50.0))

    def test_noise_none_resets_the_multiplier_to_its_default(self):
        bel = _make("mvn")
        sample_bel_posterior(bel, X_CASES, N_DRAWS, noise=3.0)
        self.assertEqual(bel.noise, 3.0)
        reset = sample_bel_posterior(bel, X_CASES, N_DRAWS)
        self.assertEqual(bel.noise, 0.01)
        np.testing.assert_array_equal(reset, sample_bel_posterior(_make("mvn"), X_CASES, N_DRAWS))

    def test_noise_does_not_enter_the_kde_posterior(self):
        a = sample_bel_posterior(_make("kde"), X_CASES, N_DRAWS, noise=0.01)
        b = sample_bel_posterior(_make("kde"), X_CASES, N_DRAWS, noise=50.0)
        np.testing.assert_array_equal(a, b)

    def test_invalid_noise_is_rejected(self):
        for bad in (-1.0, np.nan, np.inf, True, "0.1", [0.1]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sample_bel_posterior(_make("mvn"), X_CASES, N_DRAWS, noise=bad)

    def test_n_draws_and_observations_are_stored_by_predict(self):
        bel = _make("mvn")
        sample_bel_posterior(bel, X_CASES, 7)
        self.assertEqual(bel.n_posts, 7)
        self.assertEqual(bel.X_obs_f.shape[0], len(X_CASES))
        samples = sample_bel_posterior(bel, X_CASES[:3], N_DRAWS)
        self.assertEqual(bel.n_posts, N_DRAWS)
        self.assertEqual(samples.shape, (3, N_DRAWS, 2))

    def test_a_stale_n_posts_on_the_model_is_overridden(self):
        bel = _make("mvn")
        bel.n_posts = 3
        self.assertEqual(sample_bel_posterior(bel, X_CASES, N_DRAWS).shape[1], N_DRAWS)

    def test_no_state_from_an_earlier_call_leaks_into_the_next(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                bel = _make(mode)
                sample_bel_posterior(bel, X_CASES, N_DRAWS)  # four cases
                after = sample_bel_posterior(bel, X_CASES[:2], N_DRAWS)  # then two
                fresh = sample_bel_posterior(_make(mode), X_CASES[:2], N_DRAWS)
                self.assertEqual(after.shape, (2, N_DRAWS, 2))
                np.testing.assert_array_equal(after, fresh)
                other = sample_bel_posterior(bel, X_CASES[2:], N_DRAWS)
                np.testing.assert_array_equal(
                    other, sample_bel_posterior(_make(mode), X_CASES[2:], N_DRAWS)
                )

    def test_kde_functions_are_rebuilt_for_every_call(self):
        bel = _make("kde")
        sample_bel_posterior(bel, X_CASES, N_DRAWS)
        self.assertEqual(bel.kde_functions.shape[0], len(X_CASES))
        sample_bel_posterior(bel, X_CASES[:1], N_DRAWS)
        self.assertEqual(bel.kde_functions.shape[0], 1)

    def test_tm_functions_are_rebuilt_for_every_call(self):
        bel = _make("tm")
        sample_bel_posterior(bel, X_CASES[:2], N_DRAWS)
        first = bel.tm_functions
        sample_bel_posterior(bel, X_CASES[:2], N_DRAWS)
        self.assertIsNot(bel.tm_functions, first)
        self.assertEqual(bel.tm_functions.shape, first.shape)


class RejectionBeforePredictTest(unittest.TestCase):
    def _rejected(self, bel, error, observations=X_CASES, n_draws=N_DRAWS, **kwargs):
        spy = _PredictSpy(_good_output(len(np.atleast_2d(observations))))
        with mock.patch.object(bel, "predict", side_effect=spy), self.assertRaises(error):
            sample_bel_posterior(bel, observations, n_draws, **kwargs)
        self.assertEqual(spy.calls, [], "predict must not run")

    def test_an_active_x_observation_is_rejected(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                bel = _make(mode)
                bel.x_observation = np.zeros((len(X_CASES), 3))
                self._rejected(bel, ValueError)
        bel = _make("mvn")
        bel.x_observation = np.zeros((1, 3))
        self._rejected(bel, ValueError, mode="kde")

    def test_clearing_x_observation_restores_sampling(self):
        bel = _make("mvn")
        bel.x_observation = np.zeros((len(X_CASES), 3))
        self._rejected(bel, ValueError)
        bel.x_observation = None
        samples = sample_bel_posterior(bel, X_CASES, N_DRAWS)
        self.assertEqual(samples.shape, (len(X_CASES), N_DRAWS, 2))

    def test_the_x_observation_override_really_would_ignore_the_observations(self):
        bel = _make("mvn")
        pre = bel.X_pre_processing.transform(X_CASES[:1])
        bel.x_observation = pre
        ignored = _direct(bel, X_CASES[1:2])  # a different case, same answer as the override
        bel.x_observation = None
        expected = _direct(bel, X_CASES[:1])
        np.testing.assert_array_equal(ignored, expected)

    def test_models_that_are_not_fitted_or_not_models_are_rejected(self):
        self._rejected(BEL(mode="mvn", random_state=1), ValueError)
        for bad in (None, object(), "bel", np.zeros(3)):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                sample_bel_posterior(bad, X_CASES, N_DRAWS)

    def test_malformed_observations_are_rejected(self):
        bad_values = {
            "one-dimensional": X_CASES[0],
            "three-dimensional": X_CASES[None],
            "scalar": np.float64(1.0),
            "no rows": X_CASES[:0],
            "no columns": X_CASES[:, :0],
            "nan": np.where(np.arange(12).reshape(4, 3) == 5, np.nan, X_CASES),
            "inf": np.where(np.arange(12).reshape(4, 3) == 5, np.inf, X_CASES),
            "-inf": np.where(np.arange(12).reshape(4, 3) == 5, -np.inf, X_CASES),
            "too few columns": X_CASES[:, :2],
            "too many columns": np.hstack([X_CASES, X_CASES[:, :1]]),
        }
        for label, observations in bad_values.items():
            with self.subTest(label=label):
                self._rejected(_make("mvn"), ValueError, observations)
        for label, observations in (
            ("boolean", X_CASES > 0),
            ("complex", X_CASES.astype(complex)),
            ("string", X_CASES.astype(str)),
            ("object", X_CASES.astype(object)),
        ):
            with self.subTest(label=label):
                self._rejected(_make("mvn"), TypeError, observations)

    def test_malformed_draw_counts_are_rejected(self):
        for bad in (0, -1, -100):
            with self.subTest(bad=bad):
                self._rejected(_make("mvn"), ValueError, n_draws=bad)
        for bad in (True, False, np.bool_(True), 2.0, 2.5, "3", None, [3], np.float64(3)):
            with self.subTest(bad=bad):
                self._rejected(_make("mvn"), TypeError, n_draws=bad)

    def test_feature_count_is_checked_only_when_the_fitted_model_reports_one(self):
        # With a pre-processing step that reports n_features_in_, a wrong width is rejected
        # by sample_bel_posterior itself, before predict.
        reporting = _make("mvn")
        self.assertEqual(reporting.X_pre_processing.n_features_in_, 3)
        self._rejected(reporting, ValueError, X_CASES[:, :2])
        # A pass-through pre-processing step reports nothing, so this function does not
        # claim a rejection: the observations reach predict, which decides.
        silent = _make("mvn", scaled=False)
        self.assertFalse(hasattr(silent.X_pre_processing, "n_features_in_"))
        output, spy = _call_with_output(silent, _good_output(len(X_CASES)), X_CASES[:, :2])
        self.assertEqual(len(spy.calls), 1)
        self.assertEqual(output.shape, (len(X_CASES), N_DRAWS, 2))
        with self.assertRaises(ValueError):
            sample_bel_posterior(silent, X_CASES[:, :2], N_DRAWS)  # real predict refuses


class MalformedPredictOutputTest(unittest.TestCase):
    cases = len(X_CASES)

    def _rejects(self, output, *, bel=None, errors=(ValueError,)):
        bel = _make("mvn") if bel is None else bel
        spy = _PredictSpy(output)
        with mock.patch.object(bel, "predict", side_effect=spy), self.assertRaises(errors):
            sample_bel_posterior(bel, X_CASES, N_DRAWS)
        self.assertEqual(len(spy.calls), 1)

    def test_a_valid_output_is_returned_as_float64(self):
        expected = _good_output(self.cases)
        output, _ = _call_with_output(_make("mvn"), expected)
        np.testing.assert_array_equal(output, expected)
        self.assertEqual(output.dtype, np.float64)

    def test_wrong_number_of_axes(self):
        good = _good_output(self.cases)
        for label, bad in (
            ("two axes", good[:, :, 0]),
            ("one axis", good[0, :, 0]),
            ("four axes", good[None]),
            ("scalar", np.float64(1.0)),
        ):
            with self.subTest(label=label):
                self._rejects(bad)

    def test_wrong_case_count(self):
        for label, bad in (
            ("one fewer", _good_output(self.cases - 1)),
            ("one more", _good_output(self.cases + 1)),
            ("single", _good_output(1)),
            ("none", _good_output(0)),
        ):
            with self.subTest(label=label):
                self._rejects(bad)

    def test_wrong_draw_count(self):
        for label, bad in (
            ("one fewer", _good_output(self.cases, N_DRAWS - 1)),
            ("one more", _good_output(self.cases, N_DRAWS + 1)),
            ("single", _good_output(self.cases, 1)),
            ("none", _good_output(self.cases, 0)),
        ):
            with self.subTest(label=label):
                self._rejects(bad)

    def test_axes_in_the_wrong_order(self):
        # (draws, cases, targets) has the right size but the wrong layout.
        self._rejects(np.swapaxes(_good_output(self.cases), 0, 1))

    def test_zero_targets(self):
        self._rejects(_good_output(self.cases, targets=0))

    def test_wrong_target_count_when_the_fitted_model_reports_it(self):
        bel = _make("mvn")
        self.assertEqual(bel.Y_pre_processing.n_features_in_, 2)
        for targets in (1, 3, 5):
            with self.subTest(targets=targets):
                self._rejects(_good_output(self.cases, targets=targets), bel=_make("mvn"))

    def test_target_count_is_not_checked_when_the_fitted_model_reports_none(self):
        silent = _make("mvn", scaled=False)
        self.assertFalse(hasattr(silent.Y_pre_processing, "n_features_in_"))
        for targets in (1, 2, 5):
            with self.subTest(targets=targets):
                output, _ = _call_with_output(silent, _good_output(self.cases, targets=targets))
                self.assertEqual(output.shape, (self.cases, N_DRAWS, targets))

    def test_nonfinite_output(self):
        for bad_value in (np.nan, np.inf, -np.inf):
            for position in ((0, 0, 0), (self.cases - 1, N_DRAWS - 1, 1)):
                with self.subTest(bad=bad_value, position=position):
                    bad = _good_output(self.cases)
                    bad[position] = bad_value
                    self._rejects(bad)

    def test_boolean_complex_integer_and_object_output(self):
        good = _good_output(self.cases)
        for label, bad in (
            ("boolean", good > 0),
            ("complex", good.astype(complex)),
            ("complex with imaginary part", good + 1j * good),
            ("integer", np.round(good * 10).astype(np.int64)),
            ("unsigned", np.abs(np.round(good * 10)).astype(np.uint8)),
            ("object", good.astype(object)),
            ("string", good.astype(str)),
        ):
            with self.subTest(label=label):
                self._rejects(bad, errors=(ValueError, TypeError))

    def test_non_array_output(self):
        for label, bad in (("none", None), ("string", "samples"), ("dict", {"a": 1})):
            with self.subTest(label=label):
                self._rejects(bad, errors=(ValueError, TypeError))

    def test_lists_of_floats_are_read_as_arrays(self):
        output, _ = _call_with_output(_make("mvn"), _good_output(self.cases).tolist())
        self.assertEqual(output.dtype, np.float64)

    def test_reduced_precision_output_never_comes_back_as_float32(self):
        # The documented return is float64: a lower-precision predict result must be rejected
        # or converted, not handed on silently.
        for dtype in (np.float32, np.float16):
            with self.subTest(dtype=np.dtype(dtype).name):
                bel = _make("mvn")
                spy = _PredictSpy(_good_output(self.cases, dtype=dtype))
                with mock.patch.object(bel, "predict", side_effect=spy):
                    try:
                        output = sample_bel_posterior(bel, X_CASES, N_DRAWS)
                    except (ValueError, TypeError):
                        continue
                self.assertEqual(output.dtype, np.float64)

    def test_predict_output_is_checked_for_every_mode(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                self._rejects(_good_output(self.cases + 1), bel=_make(mode))


class FeedsTheOtherFunctionsTest(unittest.TestCase):
    """The draws of a real fitted model flow through scores, events and decisions."""

    def test_chain_on_a_fitted_model(self):
        bel = _make("mvn")
        samples = sample_bel_posterior(bel, X_CASES, 200)
        scores = score_samples(samples, Y_CASES, levels=[0.5, 0.9], joint=False)
        self.assertEqual(scores.crps.shape, (4, 2))
        self.assertTrue(np.all(scores.crps >= 0.0))
        self.assertEqual(scores.coverage.lower.shape, (4, 2, 2))
        threshold = Y_OFFSET[0]
        probs = event_probabilities(samples[:, :, :1] > threshold)
        self.assertEqual(probs.shape, (4, 1))
        brier = event_brier_scores(probs, Y_CASES[:, :1] > threshold, convention="binary")
        self.assertEqual(brier.shape, (4, 1))
        losses = np.stack(
            [(samples[:, :, 0] < threshold).astype(float), np.full(samples.shape[:2], 0.5)], -1
        )
        risks = decision_risks(losses)
        self.assertEqual(risks.expected_losses.shape, (4, 2))
        # The first action's expected loss is the probability of the opposite event.
        np.testing.assert_allclose(
            risks.expected_losses[:, 0], 1.0 - probs[:, 0], rtol=0, atol=1e-12
        )

    def test_scores_of_an_informative_model_beat_the_prior_spread(self):
        samples = sample_bel_posterior(_make("mvn"), X_CASES, 400)
        crps = score_samples(samples, Y_CASES, levels=0.5).crps
        # A prior-only forecast has CRPS near 0.234 * sd of the target; the posterior is sharper.
        self.assertTrue(np.all(crps.mean(axis=0) < 0.3 * Y_SCALE))


if __name__ == "__main__":
    unittest.main()
