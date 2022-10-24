import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator, MultiOutputMixin
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
        self.pca_objects = [
            PCA(n_components=n) for n in self.n_components
        ]  # list of PCA objects

    def fit(self, Xc: list, yc=None, **fit_params):
        """Fit all PCA objects for the different datasets with their specified
        n_components.

        :param Xc: list of datasets
        :param yc: Only here to satisfy the scikit-learn API
        :return: self
        """
        if type(Xc) is not list:
            Xc = [Xc]
        [pca.fit(Xc[i], yc) for i, pca in enumerate(self.pca_objects)]
        return self

    def transform(self, Xc: list, yc=None, **fit_params) -> np.array:
        """Transforms all datasets and concatenates the output.

        :param Xc: list of datasets
        :param yc: Only here to satisfy the scikit-learn API
        :return: concatenated output
        """
        if type(Xc) is not list:
            Xc = [Xc]
        [check_is_fitted(p) for p in self.pca_objects]  # Check if fitted
        scores = [
            pca.transform(Xc[i]) for i, pca in enumerate(self.pca_objects)
        ]  # Transform
        if self.scale:  # Scale the data if specified
            scaler = StandardScaler()
            scores = [scaler.fit_transform(s) for s in scores]
        return np.concatenate(scores, axis=1)

    def fit_transform(self, Xc: list, yc=None, **fit_params):
        """Fit and transform all datasets.

        :param Xc: list of datasets
        :param yc: Only here to satisfy the scikit-learn API
        :return: concatenated output
        """
        if type(Xc) is not list:
            Xc = [Xc]
        return self.fit(Xc, yc).transform(Xc, yc)

    def inverse_transform(self, Xr: np.array, yc=None, **fit_params) -> list:
        """Inverse transform the data back to the original space.

        :param Xr: transformed data
        :param yc: Only here to satisfy the scikit-learn API
        :return: list of transformed datasets
        """
        if type(Xr) is not list:
            Xr = [Xr]
        rm = np.cumsum(
            np.concatenate([[0], self.n_components])
        )  # Cumulative sum of n_components
        Xr = Xr.reshape(-1)
        Xc = [
            Xr[rm[i] : rm[i + 1]] for i in range(len(rm) - 1)
        ]  # Separates the concatenated features into the different original datasets
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
        self.t_objects = [
            self.base_function(**self.params) for _ in Xc
        ]  # list of transformations
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
            obj.inverse_transform(Xr[i].reshape(1, -1))
            for i, obj in enumerate(self.t_objects)
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
