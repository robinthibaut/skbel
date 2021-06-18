from abc import ABCMeta

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted


class CompositePCA(TransformerMixin, BaseEstimator, metaclass=ABCMeta):
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
        return np.concatenate(scores)

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
