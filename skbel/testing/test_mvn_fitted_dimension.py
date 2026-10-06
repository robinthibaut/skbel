"""Regression tests: MVN projected noise uses the fitted predictor dimension."""

import unittest

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skbel import BEL

N_POSTS = 3
NOISE = 0.05


def _data():
    rng = np.random.RandomState(0)
    latent = rng.normal(size=(20, 3))
    x = latent + 0.1 * rng.normal(size=(20, 3))
    y = latent[:, :2] @ np.array([[1.0, 0.5], [0.2, 1.0]]) + 0.1 * rng.normal(size=(20, 2))
    return x, y


def _bel(x_pre):
    return BEL(
        mode="mvn",
        X_pre_processing=x_pre,
        Y_pre_processing=Pipeline([("scaler", StandardScaler())]),
        regression_model=CCA(n_components=2),
        random_state=1,
    )


def _pca(n):
    return Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=n))])


class MVNFittedDimensionTest(unittest.TestCase):
    def setUp(self):
        self.x, self.y = _data()
        self._state = np.random.get_state()

    def tearDown(self):
        np.random.set_state(self._state)

    def _check(self, bel, expected_dim):
        bel.fit(self.x, self.y)
        rot = bel.regression_model.x_rotations_
        self.assertEqual(rot.shape[0], expected_dim)
        out = bel.predict(self.x[:2], n_posts=N_POSTS, noise=NOISE, inverse_transform=True)
        self.assertEqual(out.shape, (2, N_POSTS, 2))
        self.assertTrue(np.all(np.isfinite(out)))
        expected = rot.T @ (np.eye(expected_dim) * NOISE) @ rot
        np.testing.assert_allclose(expected, NOISE * rot.T @ rot, atol=1e-12)
        self.assertEqual(bel.posterior_covariance.shape, (2, 2, 2))
        self.assertTrue(np.all(np.isfinite(bel.posterior_covariance)))
        return expected

    def test_integer_pca(self):
        self._check(_bel(_pca(3)), 3)

    def test_default_pca(self):
        self._check(_bel(_pca(None)), 3)

    def test_fraction_pca(self):
        bel = _bel(_pca(0.9))
        bel.fit(self.x, self.y)
        retained = bel.X_pre_processing["pca"].n_components_
        self.assertIsInstance(bel.X_pre_processing["pca"].n_components, float)
        self._check(bel, retained)

    def test_retained_dimension_changes(self):
        bel2 = _bel(_pca(2))
        bel2.fit(self.x, self.y)
        self.assertEqual(bel2.regression_model.x_rotations_.shape[0], 2)
        out = bel2.predict(self.x[:2], n_posts=N_POSTS, noise=NOISE)
        self.assertEqual(out.shape, (2, N_POSTS, 2))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_non_pca_preprocessing(self):
        self._check(_bel(Pipeline([("scaler", StandardScaler())])), 3)
        self._check(_bel(StandardScaler()), 3)

    def test_projected_covariance_algebra(self):
        bel = _bel(_pca(None))
        bel.fit(self.x, self.y)
        captured = {}
        import skbel.learning.bel as mod

        orig = mod.mvn_inference

        def spy(**kwargs):
            captured["x_cov"] = kwargs["x_cov"]
            return orig(**kwargs)

        mod.mvn_inference = spy
        try:
            bel.predict(self.x[:1], n_posts=N_POSTS, noise=NOISE)
        finally:
            mod.mvn_inference = orig
        rot = bel.regression_model.x_rotations_
        np.testing.assert_allclose(captured["x_cov"], NOISE * (rot.T @ rot), atol=1e-12)

    def test_inverse_transform_false_shape(self):
        bel = _bel(_pca(None))
        bel.fit(self.x, self.y)
        out = bel.predict(self.x[:2], n_posts=N_POSTS, noise=NOISE, inverse_transform=False)
        self.assertEqual(out.shape, (2, N_POSTS, 2))
        self.assertTrue(np.all(np.isfinite(out)))

    def test_malformed_noise_rejected(self):
        bel = _bel(_pca(None))
        bel.fit(self.x, self.y)
        for bad in (-1.0, float("nan"), True, "a"):
            with self.assertRaises(ValueError):
                bel.predict(self.x[:1], n_posts=N_POSTS, noise=bad)


if __name__ == "__main__":
    unittest.main()
