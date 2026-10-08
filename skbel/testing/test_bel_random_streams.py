"""Per-observation random streams of ``BEL.random_sample`` in the mvn, kde and tm modes.

Every check runs on a small synthetic linear-Gaussian training set fitted once per mode.
"""

import unittest
from functools import cache

import numpy as np
from scipy import interpolate
from sklearn.cross_decomposition import CCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skbel import BEL
from skbel.algorithms import it_sampling
from skbel.metrics import case_rng

MODES = ("mvn", "kde", "tm")
N_POSTS = 7
SEED = 2024


def _training():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(60, 2))
    x = np.column_stack([y[:, 0] + 0.3 * y[:, 1], y[:, 1] - 0.2 * y[:, 0], y.sum(axis=1)])
    x = x + 0.2 * rng.normal(size=x.shape)
    return x, y


X_TRAIN, Y_TRAIN = _training()
# Rows 0 and 2 are the same observation, so their posteriors are identical.
X_OBS = np.array([X_TRAIN[3], X_TRAIN[10], X_TRAIN[3]])


def _state():
    return np.random.get_state()


def _same_state(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and tuple(a[2:]) == tuple(b[2:])


@cache
def _fitted(mode):
    """One fitted BEL per mode with posterior functions for ``X_OBS``."""
    bel = BEL(
        mode=mode,
        X_pre_processing=Pipeline([("scale", StandardScaler())]),
        Y_pre_processing=Pipeline([("scale", StandardScaler())]),
        regression_model=CCA(n_components=2),
        random_state=SEED,
    )
    bel.fit(X_TRAIN, Y_TRAIN)
    bel.predict(X_OBS, n_posts=N_POSTS, return_samples=False)
    return bel


def _sample(bel, **kwargs):
    kwargs.setdefault("n_posts", N_POSTS)
    return bel.random_sample(X_obs_f=bel.X_obs_f, **kwargs)


class SeedPropertyTest(unittest.TestCase):
    def test_setters_store_without_touching_global_state(self):
        bel = BEL()
        before = _state()
        bel.seed = 5
        self.assertEqual(bel.seed, 5)
        bel.random_state = np.uint32(7)
        self.assertEqual(bel.seed, 7)
        self.assertIsInstance(bel.seed, int)
        bel.seed = None
        self.assertIsNone(bel.random_state)
        self.assertTrue(_same_state(before, _state()))

    def test_setters_reject_invalid_seeds(self):
        for attr in ("seed", "random_state"):
            for bad, error in (
                (True, TypeError),
                (1.5, TypeError),
                ("3", TypeError),
                (-1, ValueError),
            ):
                with self.subTest(attr=attr, bad=bad), self.assertRaises(error):
                    setattr(BEL(), attr, bad)

    def test_constructor_seed_is_a_sklearn_parameter(self):
        bel = BEL(random_state=3)
        self.assertEqual(bel.get_params()["random_state"], 3)
        bel.set_params(random_state=4)
        self.assertEqual(bel.seed, 4)


class RandomStreamTest(unittest.TestCase):
    def test_shapes_finite_and_global_state_untouched(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                bel = _fitted(mode)
                before = _state()
                out = _sample(bel)
                self.assertTrue(_same_state(before, _state()))
                self.assertEqual(out.shape, (len(X_OBS), N_POSTS, 2))
                self.assertTrue(np.all(np.isfinite(out)))

    def test_same_seed_repeats_and_other_seed_differs(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                bel = _fitted(mode)
                first, again = _sample(bel), _sample(bel)
                np.testing.assert_array_equal(first, again)
                bel.seed = SEED + 1
                try:
                    other = _sample(bel)
                finally:
                    bel.seed = SEED
                self.assertFalse(np.allclose(first, other))

    def test_selected_row_matches_full_batch_row(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                bel = _fitted(mode)
                full = _sample(bel)
                for obs_n, row in ((0, 0), (1, 1), (-1, 2)):
                    selected = _sample(bel, obs_n=obs_n)
                    self.assertEqual(selected.shape, (1, N_POSTS, 2))
                    np.testing.assert_array_equal(selected[0], full[row])

    def test_identical_observations_get_independent_draws(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                full = _sample(_fitted(mode))
                self.assertFalse(np.allclose(full[0], full[2]))

    def test_draws_depend_only_on_seed_and_row(self):
        """A prefix batch reproduces the leading rows of the longer batch."""
        for mode in MODES:
            with self.subTest(mode=mode):
                bel = _fitted(mode)
                full = _sample(bel)
                fresh = BEL(
                    mode=mode,
                    X_pre_processing=bel.X_pre_processing,
                    Y_pre_processing=bel.Y_pre_processing,
                    regression_model=bel.regression_model,
                    random_state=SEED,
                )
                fresh.X_f, fresh.Y_f = bel.X_f, bel.Y_f
                prefix = fresh.predict(X_OBS[:2], n_posts=N_POSTS, inverse_transform=False)
                np.testing.assert_allclose(prefix, full[:2], rtol=0, atol=1e-12)

    def test_mvn_rows_use_their_own_labeled_stream(self):
        bel = _fitted("mvn")
        full = _sample(bel)
        for row in range(len(X_OBS)):
            expected = case_rng(SEED, row, "skbel.learning.BEL.random_sample").multivariate_normal(
                bel.posterior_mean[row], bel.posterior_covariance[row], size=N_POSTS
            )
            np.testing.assert_array_equal(full[row], expected)

    def test_unseeded_sampling_stores_a_fresh_seed(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                bel = _fitted(mode)
                bel.seed = None
                try:
                    before = _state()
                    first = _sample(bel)
                    self.assertTrue(_same_state(before, _state()))
                    self.assertIsInstance(bel.seed, int)
                    np.testing.assert_array_equal(first, _sample(bel))
                finally:
                    bel.seed = SEED

    def test_invalid_stored_seed_is_rejected_at_sampling(self):
        bel = _fitted("mvn")
        bel._seed = -3
        try:
            with self.assertRaises(ValueError):
                _sample(bel)
        finally:
            bel.seed = SEED


class KDECacheTest(unittest.TestCase):
    def test_cached_cdfs_reproduce_uncached_draws_full_and_selected(self):
        bel = _fitted("kde")
        full = _sample(bel)
        cache_all = bel.kde_init(bel.X_obs_f)
        np.testing.assert_array_equal(_sample(bel, init_kde=cache_all), full)
        for obs_n, row in ((1, 1), (-1, 2)):
            for cache_rows in (cache_all, bel.kde_init(bel.X_obs_f, obs_n=obs_n)):
                selected = _sample(bel, obs_n=obs_n, init_kde=cache_rows)
                np.testing.assert_array_equal(selected[0], full[row])


class ItSamplingRngTest(unittest.TestCase):
    def setUp(self):
        x = np.linspace(-1.0, 1.0, 129)
        self.pdf = interpolate.interp1d(x, np.exp(-(x**2)), kind="linear")
        self.kwargs = dict(pdf=self.pdf, num_samples=5, lower_bd=-1.0, upper_bd=1.0, k=129)

    def test_default_caller_keeps_global_uniforms(self):
        state = _state()
        try:
            np.random.seed(11)
            legacy = it_sampling(**self.kwargs)
            np.random.seed(11)
            uniforms = np.random.uniform(0, 1, 5)
        finally:
            np.random.set_state(state)
        cdf = it_sampling(**self.kwargs, return_cdf=True)
        np.testing.assert_array_equal(legacy, np.interp(uniforms, cdf, self.pdf.x))

    def test_local_generator_leaves_global_state_alone(self):
        before = _state()
        local = it_sampling(**self.kwargs, rng=np.random.default_rng(3))
        self.assertTrue(_same_state(before, _state()))
        cdf = it_sampling(**self.kwargs, return_cdf=True)
        uniforms = np.random.default_rng(3).uniform(0, 1, 5)
        np.testing.assert_array_equal(local, np.interp(uniforms, cdf, self.pdf.x))


if __name__ == "__main__":
    unittest.main()
