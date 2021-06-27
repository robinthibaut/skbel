import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator, MultiOutputMixin
from sklearn.decomposition import PCA
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
    def __init__(self, n_components: list):
        """Initiate the class by specifying a list of number of components to keep for each
        different datasets.
        """
        self.n_components = n_components
        self.pca_objects = [PCA(n_components=n) for n in self.n_components]

    def fit(self, Xc: list, yc=None, **fit_params):
        """Fit all PCA objects for the different datasets with their specified n_components"""
        [pca.fit(Xc[i], yc) for i, pca in enumerate(self.pca_objects)]
        return self

    def transform(self, Xc: list, yc=None, **fit_params) -> np.array:
        """Transforms all datasets and concatenates the output"""
        [check_is_fitted(p) for p in self.pca_objects]
        scores = [pca.transform(Xc[i]) for i, pca in enumerate(self.pca_objects)]
        return np.concatenate(scores, axis=1)

    def fit_transform(self, Xc: list, yc=None, **fit_params):
        return self.fit(Xc, yc).transform(Xc, yc)

    def inverse_transform(self, Xr: np.array, yc=None, **fit_params) -> list:
        rm = np.concatenate([[0], self.n_components])
        Xc = [
            Xr[rm[i] : rm[i + 1]] for i in range(len(rm) - 1)
        ]  # Separates the concatenated features into the
        # different original datasets
        Xit = [
            pca.inverse_transform(Xc[i]) for i, pca in enumerate(self.pca_objects)
        ]  # Successively inverse transform
        return Xit


class CompositeTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, base_function, **fit_params):
        self.base_function = base_function
        self.t_objects = None
        self.params = fit_params

    def fit(self, Xc: list, yc=None, **fit_params):
        self.t_objects = [self.base_function(self.params) for _ in Xc]
        [obj.fit(Xc[i], yc) for i, obj in enumerate(self.t_objects)]
        return self

    def transform(self, Xc: list, yc=None, **fit_params) -> np.array:
        [check_is_fitted(p) for p in self.t_objects]
        output = [obj.transform(Xc[i]) for i, obj in enumerate(self.t_objects)]
        return output

    def fit_transform(self, Xc: list, yc=None, **fit_params):
        return self.fit(Xc, yc).transform(Xc, yc)

    def inverse_transform(self, Xr: np.array, yc=None, **fit_params) -> list:
        Xit = [
            obj.inverse_transform(Xr[i]) for i, obj in enumerate(self.t_objects)
        ]  # Successively inverse transform
        return Xit


class Dummy(TransformerMixin, MultiOutputMixin, BaseEstimator):
    """Dummy transformer that does nothing"""

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
