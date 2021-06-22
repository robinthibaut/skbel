import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import linalg
from scipy.linalg import pinv2
from sklearn.base import (
    TransformerMixin,
    RegressorMixin,
    MultiOutputMixin,
    BaseEstimator,
)
from sklearn.cross_decomposition._pls import (
    _center_scale_xy,
    _get_first_singular_vectors_power_method,
    _svd_flip_1d,
    _get_first_singular_vectors_svd,
    PLSRegression,
)
from sklearn.exceptions import NotFittedError
from sklearn.metrics import pairwise_kernels
from sklearn.preprocessing import KernelCenterer
from sklearn.utils import check_consistent_length, check_array
from sklearn.utils.validation import check_is_fitted

from skbel.utils import FLOAT_DTYPES


class KernelCCA(TransformerMixin, RegressorMixin, MultiOutputMixin, BaseEstimator):
    """ """

    def __init__(
        self,
        n_components=2,
        *,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        alpha=1.0,
        fit_inverse_transform=False,
        scale=True,
        deflation_mode="canonical",
        mode="B",
        algorithm="nipals",
        max_iter=500,
        tol=1e-06,
        n_jobs=None,
        copy=True,
    ):
        self.n_components = n_components
        # Kernel params
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform
        # CCA params
        self.deflation_mode = deflation_mode
        self.mode = mode
        self.scale = scale
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs
        self.copy = copy

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
        return pairwise_kernels(
            X, Y, metric=self.kernel, filter_params=True, n_jobs=self.n_jobs, **params
        )

    def _fit_transform(self, Kx, Ky=None):
        """Fit's using kernel K"""
        # center kernel
        Kx = self._centerer.fit_transform(Kx)
        Ky = self._centerer.fit_transform(Ky)

        if self.n_components is None:
            n_components = min(Kx.shape[0], Ky.shape[0])
        else:
            n_components = min(Kx.shape[0], Ky.shape[0], self.n_components)

        n = Kx.shape[0]
        p = Kx.shape[1]
        q = Ky.shape[1]
        if self.deflation_mode == "regression":
            # With PLSRegression n_components is bounded by the rank of (X.T X)
            # see Wegelin page 25
            rank_upper_bound = p
            if not 1 <= n_components <= rank_upper_bound:
                # TODO: raise an error in 1.1
                warnings.warn(
                    f"As of version 0.24, n_components({n_components}) should "
                    f"be in [1, n_features]."
                    f"n_components={rank_upper_bound} will be used instead. "
                    f"In version 1.1 (renaming of 0.26), an error will be "
                    f"raised.",
                    FutureWarning,
                )
                n_components = rank_upper_bound
        else:
            # With CCA and PLSCanonical, n_components is bounded by the rank of
            # X and the rank of Y: see Wegelin page 12
            rank_upper_bound = min(n, p, q)
            if not 1 <= self.n_components <= rank_upper_bound:
                # TODO: raise an error in 1.1
                warnings.warn(
                    f"As of version 0.24, n_components({n_components}) should "
                    f"be in [1, min(n_features, n_samples, n_targets)] = "
                    f"[1, {rank_upper_bound}]. "
                    f"n_components={rank_upper_bound} will be used instead. "
                    f"In version 1.1 (renaming of 0.26), an error will be "
                    f"raised.",
                    FutureWarning,
                )
                n_components = rank_upper_bound

        if self.algorithm not in ("svd", "nipals"):
            raise ValueError(
                "algorithm should be 'svd' or 'nipals', got " f"{self.algorithm}."
            )

        self._norm_y_weights = self.deflation_mode == "canonical"  # 1.1
        norm_y_weights = self._norm_y_weights

        # Scale (in place)
        Xk, Yk, self._x_mean, self._y_mean, self._x_std, self._y_std = _center_scale_xy(
            Kx, Ky, self.scale
        )

        self.x_weights_ = np.zeros((p, n_components))  # U
        self.y_weights_ = np.zeros((q, n_components))  # V
        self._x_scores = np.zeros((n, n_components))  # Xi
        self._y_scores = np.zeros((n, n_components))  # Omega
        self.x_loadings_ = np.zeros((p, n_components))  # Gamma
        self.y_loadings_ = np.zeros((q, n_components))  # Delta
        self.n_iter_ = []

        # This whole thing corresponds to the algorithm in section 4.1 of the
        # review from Wegelin. See above for a notation mapping from code to
        # paper.
        Y_eps = np.finfo(Yk.dtype).eps
        for k in range(n_components):
            # Find first left and right singular vectors of the X.T.dot(Y)
            # cross-covariance matrix.
            if self.algorithm == "nipals":
                # Replace columns that are all close to zero with zeros
                Yk_mask = np.all(np.abs(Yk) < 10 * Y_eps, axis=0)
                Yk[:, Yk_mask] = 0.0

                try:
                    (
                        x_weights,
                        y_weights,
                        n_iter_,
                    ) = _get_first_singular_vectors_power_method(
                        Xk,
                        Yk,
                        mode=self.mode,
                        max_iter=self.max_iter,
                        tol=self.tol,
                        norm_y_weights=norm_y_weights,
                    )
                except StopIteration as e:
                    if str(e) != "Y residual is constant":
                        raise
                    warnings.warn(f"Y residual is constant at iteration {k}")
                    break

                self.n_iter_.append(n_iter_)

            elif self.algorithm == "svd":
                x_weights, y_weights = _get_first_singular_vectors_svd(Xk, Yk)

            # inplace sign flip for consistency across solvers and archs
            _svd_flip_1d(x_weights, y_weights)

            # compute scores, i.e. the projections of X and Y
            x_scores = np.dot(Xk, x_weights)
            if norm_y_weights:
                y_ss = 1
            else:
                y_ss = np.dot(y_weights, y_weights)
            y_scores = np.dot(Yk, y_weights) / y_ss

            # Deflation: subtract rank-one approx to obtain Xk+1 and Yk+1
            x_loadings = np.dot(x_scores, Xk) / np.dot(x_scores, x_scores)
            Xk -= np.outer(x_scores, x_loadings)

            if self.deflation_mode == "canonical":
                # regress Yk on y_score
                y_loadings = np.dot(y_scores, Yk) / np.dot(y_scores, y_scores)
                Yk -= np.outer(y_scores, y_loadings)
            if self.deflation_mode == "regression":
                # regress Yk on x_score
                y_loadings = np.dot(x_scores, Yk) / np.dot(x_scores, x_scores)
                Yk -= np.outer(x_scores, y_loadings)

            self.x_weights_[:, k] = x_weights
            self.y_weights_[:, k] = y_weights
            self._x_scores[:, k] = x_scores
            self._y_scores[:, k] = y_scores
            self.x_loadings_[:, k] = x_loadings
            self.y_loadings_[:, k] = y_loadings

        # X was approximated as Xi . Gamma.T + X_(R+1)
        # Xi . Gamma.T is a sum of n_components rank-1 matrices. X_(R+1) is
        # whatever is left to fully reconstruct X, and can be 0 if X is of rank
        # n_components.
        # Similarly, Y was approximated as Omega . Delta.T + Y_(R+1)

        # Compute transformation matrices (rotations_). See User Guide.
        self.x_rotations_ = np.dot(
            self.x_weights_,
            pinv2(np.dot(self.x_loadings_.T, self.x_weights_), check_finite=False),
        )
        self.y_rotations_ = np.dot(
            self.y_weights_,
            pinv2(np.dot(self.y_loadings_.T, self.y_weights_), check_finite=False),
        )

        self.coef_ = np.dot(self.x_rotations_, self.y_loadings_.T)
        self.coef_ = self.coef_ * self._y_std

        return Kx, Ky

    def _fit_inverse_transform(self, X_transformed, X):
        if hasattr(X, "tocsr"):
            raise NotImplementedError(
                "Inverse transform not implemented for " "sparse matrices!"
            )

        n_samples = X_transformed.shape[0]
        K = self._get_kernel(X_transformed)
        K.flat[:: n_samples + 1] += self.alpha
        dual_coef_ = linalg.solve(K, X, sym_pos=True, overwrite_a=True)
        X_transformed_fit_ = X_transformed

        return dual_coef_, X_transformed_fit_

    def fit(self, X, Y):
        """Fit model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target vectors, where `n_samples` is the number of samples and
            `n_targets` is the number of response variables.
        """

        check_consistent_length(X, Y)
        X = self._validate_data(
            X, dtype=np.float64, copy=self.copy, ensure_min_samples=2
        )
        Y = check_array(Y, dtype=np.float64, copy=self.copy, ensure_2d=False)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        self._centerer = KernelCenterer()
        Kx = self._get_kernel(X)
        Ky = self._get_kernel(Y)
        self._fit_transform(Kx, Ky)

        if self.fit_inverse_transform:
            # no need to use the kernel to transform X, use shortcut expression
            X_transformed = self._centerer.transform(self._get_kernel(X))
            Y_transformed = self._centerer.transform(self._get_kernel(Y))

            self.dual_coef_X, self.X_transformed_fit_ = self._fit_inverse_transform(
                X_transformed, X
            )
            self.dual_coef_Y, self.Y_transformed_fit_ = self._fit_inverse_transform(
                Y_transformed, Y
            )

        self.X_fit_ = X
        self.Y_fit_ = Y

        return self

    def fit_transform(self, X, y=None, **params):
        """Fit the model from data in X and transform X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
        """
        self.fit(X, y)

        # no need to use the kernel to transform X, use shortcut expression
        X_transformed = self._centerer.transform(self._get_kernel(X, self.X_fit_))
        x_scores = np.dot(X_transformed, self.x_rotations_)

        if self.fit_inverse_transform:
            self.dual_coef_X, self.X_transformed_fit_ = self._fit_inverse_transform(
                X_transformed, X
            )

        if y is not None:
            Y_transformed = self._centerer.transform(self._get_kernel(y, self.Y_fit_))
            y_scores = np.dot(Y_transformed, self.y_rotations_)

            if self.fit_inverse_transform:
                self.dual_coef_Y, self.Y_transformed_fit_ = self._fit_inverse_transform(
                    Y_transformed, y
                )
            return x_scores, y_scores

        return x_scores

    def transform(self, X, Y=None, copy=True):
        """Apply the dimension reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        Y : array-like of shape (n_samples, n_targets), default=None
            Target vectors.

        copy : bool, default=True
            Whether to copy `X` and `Y`, or perform in-place normalization.

        Returns
        -------
        `x_scores` if `Y` is not given, `(x_scores, y_scores)` otherwise.
        """
        check_is_fitted(self)
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)

        # Compute centered gram matrix between X and training data X_fit_
        Kx = self._centerer.transform(self._get_kernel(X, self.X_fit_))

        # Normalize
        # Kx -= self._x_mean
        # Kx /= self._x_std
        # Apply rotation
        x_scores = np.dot(Kx, self.x_rotations_)
        if Y is not None:
            Y = check_array(Y, ensure_2d=False, copy=copy, dtype=FLOAT_DTYPES)
            Ky = self._centerer.transform(self._get_kernel(Y, self.Y_fit_))

            if Ky.ndim == 1:
                Ky = Ky.reshape(-1, 1)
            # Ky -= self._y_mean
            # Ky /= self._y_std
            y_scores = np.dot(Ky, self.y_rotations_)
            return x_scores, y_scores

        return x_scores

    def inverse_transform(self, X):
        """Transform data back to its original space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            New data, where `n_samples` is the number of samples
            and `n_components` is the number of pls components.

        Returns
        -------
        x_reconstructed : array-like of shape (n_samples, n_features)

        Notes
        -----
        This transformation will only be exact if `n_components=n_features`.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=FLOAT_DTYPES)

        if not self.fit_inverse_transform:
            raise NotFittedError(
                "The fit_inverse_transform parameter was not"
                " set to True when instantiating and hence "
                "the inverse transform is not available."
            )

        Kx = self._get_kernel(X, self.X_transformed_fit_)

        # From pls space to original space
        X_reconstructed = np.dot(Kx, self.dual_coef_X)

        return X_reconstructed

    def predict(self, X, copy=True):
        """Predict targets of given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        copy : bool, default=True
            Whether to copy `X` and `Y`, or perform in-place normalization.

        Notes
        -----
        This call requires the estimation of a matrix of shape
        `(n_features, n_targets)`, which may be an issue in high dimensional
        space.
        """
        check_is_fitted(self)
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        Kx = self._centerer.transform(self._get_kernel(X, self.X_fit_))
        # Normalize
        # X -= self._x_mean
        # X /= self._x_std
        Ypred = np.dot(Kx, self.coef_)
        return Ypred + self._y_mean

    @property
    def norm_y_weights(self):
        return self._norm_y_weights

    @property
    def x_scores_(self):
        # TODO: raise error in 1.1 instead
        if not isinstance(self, PLSRegression):
            pass
            warnings.warn(
                "Attribute x_scores_ was deprecated in version 0.24 and "
                "will be removed in 1.1 (renaming of 0.26). Use "
                "est.transform(X) on the training data instead.",
                FutureWarning,
            )
        return self._x_scores

    @property
    def y_scores_(self):
        # TODO: raise error in 1.1 instead
        if not isinstance(self, PLSRegression):
            warnings.warn(
                "Attribute y_scores_ was deprecated in version 0.24 and "
                "will be removed in 1.1 (renaming of 0.26). Use "
                "est.transform(X) on the training data instead.",
                FutureWarning,
            )
        return self._y_scores

    def _more_tags(self):
        return {"poor_score": True, "requires_y": False}
