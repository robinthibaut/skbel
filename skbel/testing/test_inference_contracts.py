"""Regression tests for inference and preprocessing contracts."""

import importlib
import unittest
from unittest import mock

import numpy as np
from sklearn.exceptions import NotFittedError

from skbel import BEL
from skbel.preprocessing import CompositePCA


class _IdentityTransformer:
    def transform(self, X):
        return np.asarray(X)


class _PCAHolder:
    n_components = 2


class _PredictorPipeline(_IdentityTransformer):
    def __getitem__(self, key):
        assert key == "pca"
        return _PCAHolder()


class _IdentityRegression(_IdentityTransformer):
    n_components = 2
    x_rotations_ = np.eye(2)


def _tiny_mvn_bel():
    """Construct the minimum fitted-state surface used by the MVN branch."""
    bel = BEL(
        mode="mvn",
        X_pre_processing=_PredictorPipeline(),
        X_post_processing=_IdentityTransformer(),
        regression_model=_IdentityRegression(),
    )
    bel.X_f = np.array([[0.0, 0.1], [0.2, 0.3]])
    bel.Y_f = np.array([[0.4, 0.5], [0.6, 0.7]])
    return bel


def _two_training_blocks():
    return [
        np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 4.0], [3.0, 3.0], [4.0, 8.0]]),
        np.array([[2.0, 3.0], [3.0, 1.0], [5.0, 4.0], [7.0, 9.0], [11.0, 6.0]]),
    ]


class TestInferenceContracts(unittest.TestCase):
    def _sample_with_recorder(self, means, covs, **kwargs):
        bel_module = importlib.import_module("skbel.learning.bel")
        bel = _tiny_mvn_bel()
        bel.random_state = 11
        bel.posterior_mean = np.asarray(means, dtype=float)
        bel.posterior_covariance = np.asarray(covs, dtype=float)
        calls = []

        def _recording_sampler(mean, cov, size):
            calls.append((np.array(mean), np.array(cov), size))
            return np.tile(mean, (size, 1))

        stream = mock.Mock(multivariate_normal=_recording_sampler)
        with (
            mock.patch.object(bel_module, "check_is_fitted"),
            mock.patch.object(bel_module, "case_rng", return_value=stream),
        ):
            samples = bel.random_sample(X_obs_f=np.zeros((len(means), len(means[0]))), **kwargs)
        return samples, calls

    def test_selected_mvn_observation_keeps_paired_square_covariance(self):
        means = [[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]]
        covs = [
            np.eye(2),
            np.array([[2.0, 0.5], [0.5, 3.0]]),
            np.array([[4.0, -1.0], [-1.0, 5.0]]),
        ]
        for obs_n, expected in ((0, 0), (1, 1), (2, 2), (-1, 2)):
            samples, calls = self._sample_with_recorder(means, covs, obs_n=obs_n, n_posts=3)
            self.assertEqual(samples.shape, (1, 3, 2))
            self.assertEqual(len(calls), 1)
            np.testing.assert_array_equal(calls[0][0], means[expected])
            np.testing.assert_array_equal(calls[0][1], covs[expected])
            self.assertEqual(calls[0][1].shape, (2, 2))

    def test_selected_mvn_observation_one_dimension(self):
        samples, calls = self._sample_with_recorder(
            [[1.0], [5.0]], [[[2.0]], [[7.0]]], obs_n=1, n_posts=4
        )
        self.assertEqual(samples.shape, (1, 4, 1))
        np.testing.assert_array_equal(calls[0][0], [5.0])
        np.testing.assert_array_equal(calls[0][1], [[7.0]])

    def test_all_observation_mvn_sampling_unchanged(self):
        means = [[0.0, 0.0], [1.0, 2.0]]
        covs = [np.eye(2), np.array([[2.0, 0.5], [0.5, 3.0]])]
        samples, calls = self._sample_with_recorder(means, covs, n_posts=3)
        self.assertEqual(samples.shape, (2, 3, 2))
        self.assertEqual(len(calls), 2)
        for (mean, cov, _), m, c in zip(calls, means, covs, strict=True):
            np.testing.assert_array_equal(mean, m)
            np.testing.assert_array_equal(cov, c)

    def test_explicit_mvn_noise_is_used_and_default_resets(self):
        """Capture actual covariance passed into inference across call sequences."""
        bel_module = importlib.import_module("skbel.learning.bel")
        captured_covariances = []

        def _recording_mvn_inference(*, x_cov, **kwargs):
            captured_covariances.append(x_cov.copy())
            return np.zeros(2), np.eye(2)

        with mock.patch.object(bel_module, "mvn_inference", _recording_mvn_inference):
            bel = _tiny_mvn_bel()
            observation = np.array([[1.0, 2.0]])
            bel.predict(observation, noise=0.25, return_samples=False)
            bel.predict(observation, noise=0.75, return_samples=False)
            bel.predict(observation, noise=None, return_samples=False)

        np.testing.assert_allclose(
            captured_covariances,
            [np.eye(2) * 0.25, np.eye(2) * 0.75, np.eye(2) * 0.01],
        )
        self.assertEqual(bel.noise, 0.01)

    def test_mvn_noise_rejects_nonfinite_negative_boolean_and_nonscalar_values(self):
        invalid_values = [
            True,
            np.bool_(False),
            -0.1,
            np.inf,
            -np.inf,
            np.nan,
            10**1000,
            [0.1],
            np.array([0.1]),
            "0.1",
        ]
        for noise in invalid_values:
            with self.subTest(noise=noise):
                with self.assertRaisesRegex(ValueError, "finite non-negative scalar"):
                    _tiny_mvn_bel().predict(
                        np.array([[1.0, 2.0]]), noise=noise, return_samples=False
                    )

    def test_composite_pca_full_component_roundtrip_for_each_scaling_mode(self):
        blocks = _two_training_blocks()
        for scale in (False, True):
            with self.subTest(scale=scale):
                model = CompositePCA(n_components=[2, 2], scale=scale)
                scores = model.fit_transform(blocks)
                restored = model.inverse_transform(scores)
                self.assertEqual([block.shape for block in restored], [(5, 2), (5, 2)])
                for original, reconstructed in zip(blocks, restored, strict=True):
                    np.testing.assert_allclose(reconstructed, original, atol=1e-12)

    def test_scaled_composite_transform_is_batch_invariant_and_scalers_are_frozen(self):
        model = CompositePCA(n_components=[2, 2], scale=True).fit(_two_training_blocks())
        means_before = [scaler.mean_.copy() for scaler in model.scalers_]
        scales_before = [scaler.scale_.copy() for scaler in model.scalers_]
        withheld = [
            np.array([[8.0, 13.0], [9.0, 7.0]]),
            np.array([[13.0, 8.0], [17.0, 12.0]]),
        ]
        batch_scores = model.transform(withheld)
        row_scores = np.vstack(
            [model.transform([withheld[0][i : i + 1], withheld[1][i : i + 1]]) for i in range(2)]
        )
        np.testing.assert_allclose(batch_scores, row_scores)
        for scaler, mean, scale in zip(model.scalers_, means_before, scales_before, strict=True):
            np.testing.assert_allclose(scaler.mean_, mean)
            np.testing.assert_allclose(scaler.scale_, scale)

    def test_composite_pca_inverse_accepts_one_row_vector_and_rejects_bad_contracts(self):
        blocks = _two_training_blocks()
        model = CompositePCA(n_components=[2, 2], scale=True).fit(blocks)
        one_row_scores = model.transform([block[:1] for block in blocks])[0]
        restored = model.inverse_transform(one_row_scores)
        self.assertEqual([block.shape for block in restored], [(1, 2), (1, 2)])
        for original, reconstructed in zip(blocks, restored, strict=True):
            # PCA/scaler inversion can leave machine-precision residuals at exact zero.
            np.testing.assert_allclose(reconstructed, original[:1], rtol=0, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "Expected 4 transformed features"):
            model.inverse_transform(np.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, "Expected 2 input block"):
            model.transform(blocks[:1])
        with self.assertRaises(ValueError):
            model.transform([np.ones((2, 3)), np.ones((2, 2))])

    def test_composite_pca_inverse_uses_fitted_widths_for_none_components(self):
        blocks = _two_training_blocks()
        model = CompositePCA(n_components=[None, None], scale=True)
        scores = model.fit_transform(blocks)

        restored = model.inverse_transform(scores)

        self.assertEqual(scores.shape[1], sum(pca.n_components_ for pca in model.pca_objects))
        for original, reconstructed in zip(blocks, restored, strict=True):
            np.testing.assert_allclose(reconstructed, original, atol=1e-12)

    def test_composite_pca_inverse_uses_fitted_widths_for_variance_components(self):
        blocks = _two_training_blocks()
        model = CompositePCA(n_components=[0.75, 0.75], scale=True)
        scores = model.fit_transform(blocks)

        restored = model.inverse_transform(scores)
        projected_again = model.transform(restored)

        self.assertEqual(scores.shape[1], sum(pca.n_components_ for pca in model.pca_objects))
        np.testing.assert_allclose(projected_again, scores, atol=1e-12)

    def test_composite_pca_requires_fit_before_transform_or_inverse(self):
        model = CompositePCA(n_components=[1, 1], scale=True)
        with self.assertRaises(NotFittedError):
            model.transform([np.ones((2, 2)), np.ones((2, 2))])
        with self.assertRaises(NotFittedError):
            model.inverse_transform(np.ones((1, 2)))
