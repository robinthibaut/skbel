#  Copyright (c) 2021. Robin Thibaut, Ghent University

"""Discrete Cosinus Transform written in scikit-learn style"""

import numpy as np
from scipy.fftpack import dct, idct
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array

__all__ = ["DiscreteCosinusTransform2D"]


def dct2(a):
    return dct(dct(a.T, norm="ortho").T, norm="ortho")


def idct2(a):
    return idct(idct(a.T, norm="ortho").T, norm="ortho")


class DiscreteCosinusTransform2D(TransformerMixin, BaseEstimator):
    def __init__(self, *, m_cut: int = None, n_cut: int = None):
        # Original shape
        self.n_rows = None
        self.n_cols = None
        # Number of components to keep
        self.m_cut = m_cut
        self.n_cut = n_cut

    def fit(self, X, y):
        return self

    def transform(self, X):
        try:
            X = check_array(X, allow_nd=True)
        except ValueError:
            X = check_array(X.reshape(1, -1))

        self.n_rows = X.shape[1]
        self.n_cols = X.shape[2]

        if self.m_cut is None:
            self.m_cut = self.n_rows

        if self.n_cut is None:
            self.n_cut = self.n_cols

        X_dct = np.array([dct2(e)[: self.m_cut, : self.n_cut] for e in X])

        X_dct = X_dct.reshape((X_dct.shape[0], -1))

        return X_dct

    def inverse_transform(self, X):
        try:
            X = check_array(X, allow_nd=True)
        except ValueError:
            X = check_array(self.X.reshape(1, -1))

        X = X.reshape(-1, self.m_cut, self.n_cut)

        dummy = np.zeros((X.shape[0], self.n_rows, self.n_cols))
        dummy[:, : self.m_cut, : self.n_cut] = X

        X_ivt = np.array([idct2(e) for e in dummy])

        return X_ivt

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)
