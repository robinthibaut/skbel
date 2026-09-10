import numpy as np
from sklearn.base import BaseEstimator, MultiOutputMixin, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

"""
Collection of classes to combine multiple-features transformation/dimension reduction.
The classes below take a base scikit-learn object and sequentially apply the desired algorithm to each set of features
part of the same dataset and concatenates the results.
Scikit-Learn does implement its own "column_transformer", but it is not supported by pipelines, and does not have an
"inverse_transform" method. The code here solves these shortcomings.
"""

__all__ = ["CompositePCA", "CompositeTransformer", "Dummy"]


class CompositePCA(TransformerMixin, BaseEstimator):
    def __init__(self, n_components: list, scale: bool = False):
        """Initiate the class by specifying a list of number of components to
        keep for each different datasets.

        :param n_components: list of number of components to keep for each dataset
        :param scale: whether to scale the data before applying PCA
        """
        if type(n_components) is not list:
            n_components = [n_components]
        self.n_components = n_components
        self.scale = scale
        self.pca_objects = [PCA(n_components=n) for n in self.n_components]  # list of PCA objects

    def _as_blocks(self, Xc):
        """Normalize one-block input and validate the configured block count."""
        blocks = Xc if isinstance(Xc, list) else [Xc]
        if len(blocks) != len(self.pca_objects):
            raise ValueError(f"Expected {len(self.pca_objects)} input block(s), got {len(blocks)}.")
        return blocks

    def fit(self, Xc: list, yc=None, **fit_params):
        """Fit all PCA objects for the different datasets with their specified
        n_components.

        :param Xc: list of datasets
        :param yc: Only here to satisfy the scikit-learn API
        :return: self
        """
        Xc = self._as_blocks(Xc)
        [pca.fit(Xc[i], yc) for i, pca in enumerate(self.pca_objects)]
        if self.scale:
            self.scalers_ = [
                StandardScaler().fit(pca.transform(Xc[i])) for i, pca in enumerate(self.pca_objects)
            ]
        return self

    def transform(self, Xc: list, yc=None, **fit_params) -> np.array:
        """Transforms all datasets and concatenates the output.

        :param Xc: list of datasets
        :param yc: Only here to satisfy the scikit-learn API
        :return: concatenated output
        """
        Xc = self._as_blocks(Xc)
        [check_is_fitted(p) for p in self.pca_objects]  # Check if fitted
        scores = [pca.transform(Xc[i]) for i, pca in enumerate(self.pca_objects)]  # Transform
        if self.scale:  # Use scalers fitted on training PCA scores only.
            check_is_fitted(self, "scalers_")
            scores = [scaler.transform(scores[i]) for i, scaler in enumerate(self.scalers_)]
        return np.concatenate(scores, axis=1)

    def fit_transform(self, Xc: list, yc=None, **fit_params):
        """Fit and transform all datasets.

        :param Xc: list of datasets
        :param yc: Only here to satisfy the scikit-learn API
        :return: concatenated output
        """
        Xc = self._as_blocks(Xc)
        return self.fit(Xc, yc).transform(Xc, yc)

    def inverse_transform(self, Xr: np.array, yc=None, **fit_params) -> list:
        """Inverse transform the data back to the original space.

        :param Xr: concatenated transformed scores. A two-dimensional array has
            shape ``(n_samples, sum(fitted PCA component widths))``. A
            one-dimensional array of that length is treated as one sample.
        :param yc: Only here to satisfy the scikit-learn API
        :return: list of transformed datasets
        """
        [check_is_fitted(p) for p in self.pca_objects]
        if self.scale:
            check_is_fitted(self, "scalers_")
        Xr = np.asarray(Xr)
        component_widths = [pca.n_components_ for pca in self.pca_objects]
        expected_width = sum(component_widths)
        if Xr.ndim == 1:
            if Xr.shape[0] != expected_width:
                raise ValueError(
                    f"Expected {expected_width} transformed features, got {Xr.shape[0]}."
                )
            Xr = Xr.reshape(1, -1)
        elif Xr.ndim != 2:
            raise ValueError("Xr must be a one- or two-dimensional array of transformed scores.")
        if Xr.shape[1] != expected_width:
            raise ValueError(f"Expected {expected_width} transformed features, got {Xr.shape[1]}.")
        rm = np.cumsum(
            np.concatenate([[0], component_widths])
        )  # Cumulative fitted component widths
        Xc = [Xr[:, rm[i] : rm[i + 1]] for i in range(len(rm) - 1)]
        if self.scale:
            Xc = [scaler.inverse_transform(Xc[i]) for i, scaler in enumerate(self.scalers_)]
        Xit = [
            pca.inverse_transform(Xc[i]) for i, pca in enumerate(self.pca_objects)
        ]  # Successively inverse transform
        return Xit


class CompositeTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, base_function, **fit_params):
        """Initiate the class by specifying a base scikit-learn object and the
        parameters to use for each dataset.

        :param base_function: function to apply to the data
        :param fit_params: parameters to pass to the base function
        """
        self.base_function = base_function
        self.t_objects = None
        self.params = fit_params

    def fit(self, Xc: list, yc=None, **fit_params):
        """Fit all transformations for the different datasets with their
        specified parameters.

        :param Xc: list of datasets
        :param yc: Only here to satisfy the scikit-learn API
        :return: self
        """
        self.t_objects = [self.base_function(**self.params) for _ in Xc]  # list of transformations
        [obj.fit(Xc[i], yc) for i, obj in enumerate(self.t_objects)]  # Fit
        return self

    def transform(self, Xc: list, yc=None, **fit_params) -> np.array:
        """Transforms all datasets and concatenates the output.

        :param Xc: list of datasets
        :param yc: Only here to satisfy the scikit-learn API
        :return: concatenated output
        """
        [check_is_fitted(p) for p in self.t_objects]
        output = [obj.transform(Xc[i]) for i, obj in enumerate(self.t_objects)]
        return output

    def fit_transform(self, Xc: list, yc=None, **fit_params):
        """Fit and transform all datasets.

        :param Xc: list of datasets
        :param yc: Only here to satisfy the scikit-learn API
        :return: concatenated output
        """
        return self.fit(Xc, yc).transform(Xc, yc)

    def inverse_transform(self, Xr: np.array, yc=None, **fit_params) -> list:
        """Inverse transform the data back to the original space.

        :param Xr: transformed data
        :param yc: Only here to satisfy the scikit-learn API
        :return: list of transformed datasets
        """
        Xit = [
            obj.inverse_transform(Xr[i].reshape(1, -1)) for i, obj in enumerate(self.t_objects)
        ]  # Successively inverse transform
        return Xit


class Dummy(TransformerMixin, MultiOutputMixin, BaseEstimator):
    """Dummy transformer that does nothing."""

    def __init__(self):
        self.fake_fit_ = np.zeros(1)

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None, y=None):  # noqa
        if X is not None and y is None:
            return X

        elif y is not None and X is None:
            return y

        else:
            return X, y

    def inverse_transform(self, X=None, y=None):  # noqa
        if X is not None and y is None:
            return X

        elif y is not None and X is None:
            return y

        else:
            return X, y

    def fit_transform(self, X=None, y=None, **fit_params):
        return self.fit(X, y).transform(X, y)
