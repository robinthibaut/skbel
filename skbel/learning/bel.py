"""
Bayesian Evidential Learning Framework

Currently, the common practice is to first transform predictor and target variables
through PCA, and then apply CCA.

It would be interesting to try other techniques and implement it in the framework.

Alternative blueprints could be written in the same style as the BEL class implementing the classic scheme.

"""

#  Copyright (c) 2021. Robin Thibaut, Ghent University

import numpy as np
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    MultiOutputMixin,
)
from sklearn.utils import check_array
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import (
    check_is_fitted,
    check_consistent_length,
)

from ..algorithms import mvn_inference, posterior_conditional, it_sampling


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


class BEL(TransformerMixin, MultiOutputMixin, BaseEstimator):
    """
    Heart of the framework. Inherits from scikit-learn base classes.
    """

    def __init__(
        self,
        mode: str = "mvn",
        copy: bool = True,
        *,
        X_pre_processing=None,
        Y_pre_processing=None,
        X_post_processing=None,
        Y_post_processing=None,
        cca=None,
        x_pc=None,
        y_pc=None,
        x_dim=None,
        y_dim=None,
    ):
        """
        :param mode: How to infer the posterior distribution. "mvn" (default) or "kde"
        :param copy: Whether to copy arrays or not (default is True).
        :param X_pre_processing: sklearn pipeline for pre-processing the predictor.
        :param Y_pre_processing: sklearn pipeline for pre-processing the target.
        :param X_post_processing: sklearn pipeline for post-processing the predictor.
        :param X_post_processing: sklearn pipeline for post-processing the target.
        :param cca: sklearn cca object
        :param x_pc: Number of principal components to keep (predictor).
        :param y_pc: Number of principal components to keep (target).
        :param x_dim: Predictor original dimensions.
        :param y_dim: Target original dimensions.
        """
        self.copy = copy
        # How to infer the posterior parameters
        self.mode = mode

        # Processing pipelines
        if X_pre_processing is None:
            X_pre_processing = Pipeline([("nothing", Dummy())])
        if Y_pre_processing is None:
            Y_pre_processing = Pipeline([("nothing", Dummy())])
        if X_post_processing is None:
            X_post_processing = Pipeline([("nothing", Dummy())])
        if Y_post_processing is None:
            Y_post_processing = Pipeline([("nothing", Dummy())])
        if cca is None:
            cca = Pipeline([("nothing", Dummy())])

        self.X_pre_processing = X_pre_processing
        self.Y_pre_processing = Y_pre_processing
        self.X_post_processing = X_post_processing
        self.Y_post_processing = Y_post_processing
        self.cca = cca

        # Posterior parameters
        # MVN inference
        self.posterior_mean = None
        self.posterior_covariance = None
        # KDE inference
        self.kde_functions = None

        # Parameters for sampling
        self._seed = None
        self._n_posts = None

        # Original dataset
        self._X_shape, self._Y_shape = x_dim, y_dim
        self.X, self.Y = None, None
        self.X_obs, self.Y_obs = None, None  # Observation data

        # Dataset after preprocessing (dimension-reduced by self.X_n_pc, self.Y_n_pc)
        self._X_n_pc, self._Y_n_pc = x_pc, y_pc
        self.X_pc, self.Y_pc = None, None
        self.X_obs_pc, self.Y_obs_pc = None, None
        # Dataset after learning
        self.X_c, self.Y_c = None, None
        self.X_obs_c, self.Y_obs_c = None, None
        # Dataset after postprocessing
        self.X_f, self.Y_f = None, None
        self.X_obs_f, self.Y_obs_f = None, None

    # The following properties are central to the BEL framework
    @property
    def X_shape(self):
        """Predictor original shape"""
        return self._X_shape

    @X_shape.setter
    def X_shape(self, x_shape):
        self._X_shape = x_shape

    @property
    def Y_shape(self):
        """Predictor original shape"""
        return self._Y_shape

    @Y_shape.setter
    def Y_shape(self, y_shape):
        self._Y_shape = y_shape

    @property
    def X_n_pc(self):
        """Number of components to keep after pre-processing (dimensionality reduction)"""
        return self._X_n_pc

    @X_n_pc.setter
    def X_n_pc(self, x_n_pc):
        self._X_n_pc = x_n_pc

    @property
    def Y_n_pc(self):
        """Number of components to keep after pre-processing (dimensionality reduction)"""
        return self._Y_n_pc

    @Y_n_pc.setter
    def Y_n_pc(self, y_n_pc):
        self._Y_n_pc = y_n_pc

    @property
    def n_posts(self):
        """Number of sample to extract from the posterior multivariate distribution after post-processing"""
        return self._n_posts

    @n_posts.setter
    def n_posts(self, n_p):
        self._n_posts = n_p

    @property
    def seed(self):
        """Seed a.k.a. random state to reproduce the same samples"""
        return self._seed

    @seed.setter
    def seed(self, s):
        self._seed = s
        np.random.seed(self._seed)

    def fit(self, X, Y):
        """
        Fit all pipelines.
        :param X: Predictor array.
        :param Y: Target array.
        :return:
        """
        check_consistent_length(X, Y)
        self.X, self.Y = X, Y  # Save array with names

        _X = self._validate_data(
            X,
            dtype=np.float64,
            copy=self.copy,
            ensure_min_samples=2,
            allow_nd=True,
        )
        _Y = check_array(
            Y,
            dtype=np.float64,
            copy=self.copy,
            ensure_2d=False,
            allow_nd=True,
        )

        _xt, _yt = (
            self.X_pre_processing.fit_transform(_X),
            self.Y_pre_processing.fit_transform(_Y),
        )

        _xt, _yt = (
            _xt[:, : self.X_n_pc],
            _yt[:, : self.Y_n_pc],
        )  # Cut PC

        # Dataset after preprocessing
        self.X_pc, self.Y_pc = _xt, _yt

        # Canonical variates
        _xc, _yc = self.cca.fit_transform(X=_xt, y=_yt)

        self.X_c, self.Y_c = _xc, _yc

        # CV Normalized
        _xf, _yf = (
            self.X_post_processing.fit_transform(_xc),
            self.Y_post_processing.fit_transform(_yc),
        )

        self.X_f, self.Y_f = _xf, _yf

        return self

    def transform(self, X=None, Y=None) -> (np.array, np.array):
        """
        Transform data across all pipelines
        :param X: Predictor array.
        :param Y: Target array.
        :return: Post-processed variables
        """

        check_is_fitted(self.cca)
        # The key here is to cut PC's based on the number defined in configuration file

        if X is not None and Y is None:
            X = check_array(X, copy=self.copy)
            _xt = self.X_pre_processing.transform(X)
            _xt = _xt[:, : self.X_n_pc]
            _xc = self.cca.transform(X=_xt)
            _xp = self.X_post_processing.transform(_xc)

            return _xp

        elif Y is not None and X is None:
            Y = check_array(Y, copy=self.copy, ensure_2d=False, allow_nd=True)
            _xt, _yt = (
                self.X_pre_processing.transform(self.X),
                self.Y_pre_processing.transform(Y),
            )
            _xt, _yt = (
                _xt[:, : self.X_n_pc],
                _yt[:, : self.Y_n_pc],
            )
            _, _yc = self.cca.transform(X=_xt, Y=_yt)
            _yp = self.Y_post_processing.transform(_yc)

            return _yp

        else:
            _xt, _yt = (
                self.X_pre_processing.transform(self.X),
                self.Y_pre_processing.transform(self.Y),
            )
            _xt, _yt = (
                _xt[:, : self.X_n_pc],
                _yt[:, : self.Y_n_pc],
            )
            _xc, _yc = self.cca.transform(X=_xt, Y=_yt)

            _xp, _yp = (
                self.X_post_processing.transform(_xc),
                self.Y_post_processing.transform(_yc),
            )

            return _xp, _yp

    def random_sample(self, n_posts: int = None, mode: str = None) -> np.array:
        """
        Random sample the inferred posterior Gaussian distribution
        :param n_posts: int
        :param mode: str
        :return:
        """
        if mode is not None:
            self.mode = mode

        # Set the seed for later use
        if self.seed is None:
            self.seed = np.random.randint(2 ** 32 - 1, dtype="uint32")

        check_is_fitted(self.cca)
        if n_posts is None:
            n_posts = self.n_posts
        else:
            self.n_posts = n_posts
        # Draw n_posts random samples from the multivariate normal distribution :
        # Pay attention to the transpose operator
        np.random.seed(self.seed)

        if self.mode == "mvn":
            Y_samples = np.random.multivariate_normal(
                mean=self.posterior_mean, cov=self.posterior_covariance, size=n_posts
            )

        if self.mode == "kde":
            Y_samples = np.zeros((self.n_posts, self.kde_functions.shape[0]))
            # PArses the functions dict
            for i, fun in enumerate(self.kde_functions):
                if fun["kind"] == "pdf":
                    pdf = fun["function"]
                    uniform_samples = it_sampling(
                        pdf=pdf,
                        num_samples=self.n_posts,
                        lower_bd=pdf.x.min(),
                        upper_bd=pdf.x.max(),
                        chebyshev=False,
                    )
                elif fun["kind"] == "linear":
                    rel1d = fun["function"]
                    uniform_samples = np.ones(self.n_posts)*rel1d.predict(np.array([self.X_obs_f.T[i]]))

                Y_samples[:, i] = uniform_samples

        if self.mode == "kde_chebyshev":
            Y_samples = np.zeros((self.n_posts, self.kde_functions.shape[0]))
            for i, fun in enumerate(self.kde_functions):
                if fun["kind"] == "pdf":
                    pdf = fun["function"]
                    uniform_samples = it_sampling(
                        pdf=pdf,
                        num_samples=self.n_posts,
                        lower_bd=pdf.x.min(),
                        upper_bd=pdf.x.max(),
                        chebyshev=True,
                    )
                elif fun["kind"] == "linear":
                    rel1d = fun["function"]
                    uniform_samples = np.ones(self.n_posts)*rel1d(self.X_obs_f)

                Y_samples[:, i] = uniform_samples

        return Y_samples

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit-Transform across all pipelines
        :param X:
        :param y:
        :return: If mode == "mvn" - returns the posterior mean and covariance. If mode == "kde" - returns a dictionary
        of functions.
        """

        return self.fit(X, y).transform(X, y)

    def predict(self, X_obs, mode: str = None) -> (np.array, np.array):
        """
        Make predictions, in the BEL fashion.
        """
        if mode is not None:
            self.mode = mode
        self.X_obs = X_obs  # Save array with name
        try:
            X_obs = check_array(self.X_obs, allow_nd=True)
        except ValueError:
            try:
                X_obs = check_array(self.X_obs.to_numpy().reshape(1, -1))
            except AttributeError:
                X_obs = check_array(self.X_obs.reshape(1, -1))
        # Project observed data into canonical space.
        X_obs = self.X_pre_processing.transform(X_obs)
        X_obs = X_obs[:, : self.X_n_pc]
        self.X_obs_pc = X_obs
        X_obs = self.cca.transform(X_obs)
        self.X_obs_c = X_obs
        X_obs = self.X_post_processing.transform(X_obs)
        self.X_obs_f = X_obs

        # Estimate the posterior mean and covariance
        if self.mode == "mvn":

            # Evaluate the covariance in d (here we assume no data error, so C is identity times a given factor)
            # Number of PCA components for the curves
            x_dim = self.X_n_pc
            noise = 0.01
            # I matrix. (n_comp_PCA, n_comp_PCA)
            x_cov = np.eye(x_dim) * noise
            # (n_comp_CCA, n_comp_CCA)
            # Get the rotation matrices
            x_rotations = self.cca.x_rotations_
            x_cov = x_rotations.T @ x_cov @ x_rotations
            dict_args = {"x_cov": x_cov}

            X, Y = self.X_f, self.Y_f

            self.posterior_mean, self.posterior_covariance = mvn_inference(
                X=X,
                Y=Y,
                X_obs=X_obs,
                **dict_args,
            )
            return self.posterior_mean, self.posterior_covariance

        else:
            # KDE inference
            from scipy import interpolate

            for comp_n in range(self.cca.n_components):

                # If the relation is almost perfectly linear, it doesn't make sense to perform a
                # KDE estimation.
                corr = np.corrcoef(self.X_f.T[comp_n], self.Y_f.T[comp_n]).diagonal(offset=1)[0]
                # If the Pearson's correlation coefficient is > 0.999, linear regression is used instead of KDE.
                if corr >= 0.999:
                    kind = "linear"
                    fun = LinearRegression().fit(self.X_f.T[comp_n].reshape(-1, 1),
                                                 self.Y_f.T[comp_n].reshape(-1, 1))

                else:
                    # Conditional:
                    hp, sup = posterior_conditional(
                        X=self.X_f.T[comp_n],
                        Y=self.Y_f.T[comp_n],
                        X_obs=self.X_obs_f.T[comp_n],
                    )
                    hp[np.abs(hp) < 1e-12] = 0  # Set very small values to 0.
                    # hp = _normalize_distribution(hp, sup)
                    kind = "pdf"
                    fun = interpolate.interp1d(sup, hp, kind="cubic")

                # The KDE inference method can be hybrid - the returned functions are saved as a dictionary
                sample_fun = {"kind": kind, "function": fun}

                if comp_n > 0:
                    functions = np.concatenate((functions, [sample_fun]), axis=0)
                else:
                    functions = [sample_fun]

            self.kde_functions = functions

            return self.kde_functions

    def inverse_transform(
        self,
        Y_pred,
    ) -> np.array:
        """
        Back-transforms the sampled gaussian distributed posterior Y to their physical space.
        :param Y_pred:
        :return: forecast_posterior
        """
        check_is_fitted(self.cca)
        Y_pred = check_array(Y_pred)

        y_post = self.Y_post_processing.inverse_transform(
            Y_pred
        )  # Posterior CCA scores
        y_post = (
            np.matmul(y_post, self.cca.y_loadings_.T) * self.cca.y_std_
            + self.cca.y_mean_
        )  # Posterior PC scores

        # Back transform PC scores
        nc = self.Y_pc.shape[0]  # Number of components
        dummy = np.zeros((self.n_posts, nc))  # Create a dummy matrix filled with zeros
        dummy[
            :, : y_post.shape[1]
        ] = y_post  # Fill the dummy matrix with the posterior PC
        y_post = self.Y_pre_processing.inverse_transform(dummy)  # Inverse transform

        return y_post
