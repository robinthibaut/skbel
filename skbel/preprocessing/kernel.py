import numpy as np
from scipy import linalg
from sklearn.base import (
    TransformerMixin,
    BaseEstimator,
)
from sklearn.exceptions import NotFittedError
from sklearn.metrics import pairwise_kernels
from sklearn.utils.validation import check_is_fitted


class Kernel(TransformerMixin, BaseEstimator):
    """"""

    def __init__(
        self,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        alpha=1.0,
        fit_inverse_transform=False,
        n_jobs=None,
        copy=False,
    ):
        # Kernel params
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform
        self.n_jobs = n_jobs
        self.copy_X = copy
        self.X_fit_ = None

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
        return pairwise_kernels(
            X, Y, metric=self.kernel, filter_params=True, n_jobs=self.n_jobs, **params
        )

    def _fit_inverse_transform(self, X, K=None):
        if hasattr(X, "tocsr"):
            raise NotImplementedError(
                "Inverse transform not implemented for sparse matrices!"
            )
        if K is None:
            K = self._get_kernel(X)
        n_samples = X.shape[0]
        K.flat[:: n_samples + 1] += self.alpha
        self.dual_coef_ = linalg.solve(K, X, sym_pos=True, overwrite_a=True)

    def fit(self, X, y=None):
        """Fit the model from data in X.

        :param X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        :return: self : object -- returns the instance itself.
        """
        X = self._validate_data(X, accept_sparse="csr", copy=self.copy_X)
        K = self._get_kernel(X)
        #
        if self.fit_inverse_transform:
            self._fit_inverse_transform(X, K)

        self.X_fit_ = X

        return self

    def transform(self, X, y=None):
        """Transforms X.

        :param X: {array-like, sparse matrix} of shape (n_samples, n_features)

        :return: X_new: ndarray of shape (n_samples, n_components)
        """
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse="csr", reset=False)

        # Compute centered gram matrix between X and training data X_fit_
        # K = self._centerer.transform(self._get_kernel(X, self.X_fit_))
        K = self._get_kernel(X, self.X_fit_)
        return K

    def fit_transform(self, X, y=None, **params):
        """Fit the model from data in X and transform X.

        :param X: {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples  and n_features is the number of features.
        :return: X_new: ndarray of shape (n_samples, n_components)
        """
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, X):
        """Transform X back to original space.

         ``inverse_transform`` approximates the inverse transformation using
         a learned pre-image. The pre-image is learned by kernel ridge
         regression of the original data on their low-dimensional representation
         vectors.

         .. [2] "Learning to Find Pre-Images", G BakIr et al, 2004.

        :param X: {array-like, sparse matrix} of shape (n_samples, n_components)
        :return: X_new: ndarray of shape (n_samples, n_features)
        """
        if not self.fit_inverse_transform:
            raise NotFittedError(
                "The fit_inverse_transform parameter was not"
                " set to True when instantiating and hence "
                "the inverse transform is not available."
            )

        return np.dot(X, self.dual_coef_)

    def _more_tags(self):
        return {
            "preserves_dtype": [np.float64, np.float32],
            "pairwise": self.kernel == "precomputed",
        }
