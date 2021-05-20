#  Copyright (c) 2021. Robin Thibaut, Ghent University

from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.base import MultiOutputMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES

__all__ = ["_PFA"]


class _PFA(
    TransformerMixin, RegressorMixin, MultiOutputMixin, BaseEstimator, metaclass=ABCMeta
):
    """Prediction Focused Approach (PFA)"""

    @abstractmethod
    def __init__(self, learner, mode="mvn"):

        self.learner = learner
        self.n_components = learner.n_components

        self.mode = mode

    def fit(self, X, Y):
        """Fit model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        Y : array-like of shape (n_samples, n_targets)
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables.
        """

        cca = self.learner(n_components=self.n_components)
        cca.fit(X=X, Y=Y)

        return self

    def transform(self, X, Y=None, copy=True):
        """Apply the dimension reduction learned on the train data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        Y : array-like of shape (n_samples, n_targets)
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables.

        copy : boolean, default True
            Whether to copy X and Y, or perform in-place normalization.

        Returns
        -------
        x_scores if Y is not given, (x_scores, y_scores) otherwise.
        """
        if Y is not None:
            x_scores, y_scores = self.learner.fit_transform(X, Y)
            return x_scores, y_scores
        else:
            x_scores = self.learner.fit_transform(X)
            return x_scores

    def inverse_transform(self, X):
        """Transform data back to its original space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            New data, where n_samples is the number of samples
            and n_components is the number of pls components.

        Returns
        -------
        x_reconstructed : array-like of shape (n_samples, n_features)

        Notes
        -----
        This transformation will only be exact if n_components=n_features
        """
        check_is_fitted(self)
        X = check_array(X, dtype=FLOAT_DTYPES)
        # From pls space to original space
        X_reconstructed = np.matmul(X, self.x_loadings_.T)

        # Denormalize
        X_reconstructed *= self.x_std_
        X_reconstructed += self.x_mean_
        return X_reconstructed

    def predict(self, X, copy=True):
        """Apply the dimension reduction learned on the train data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        copy : boolean, default True
            Whether to copy X and Y, or perform in-place normalization.

        Notes
        -----
        This call requires the estimation of a p x q matrix, which may
        be an issue in high dimensional space.
        """
        check_is_fitted(self)
        X = check_array(X, copy=copy, dtype=FLOAT_DTYPES)
        # Normalize
        X -= self.x_mean_
        X /= self.x_std_
        Ypred = np.dot(X, self.coef_)
        return Ypred + self.y_mean_

    def fit_transform(self, X, y=None):
        """Learn and apply the dimension reduction on the train data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        y : array-like of shape (n_samples, n_targets)
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables.

        Returns
        -------
        x_scores if Y is not given, (x_scores, y_scores) otherwise.
        """
        return self.fit(X, y).transform(X, y)

    def _more_tags(self):
        return {"poor_score": True, "requires_y": False}
