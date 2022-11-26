#  Copyright (c) 2022. Robin Thibaut, Ghent University

"""Bayesian Evidential Learning Framework.

Currently, the common practice is to first transform predictor and target variables
through PCA, and then apply CCA.

Alternative blueprints could be written in the same style as the BEL class implementing the classic scheme.
"""

import numpy as np
from scipy import interpolate, stats
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    MultiOutputMixin,
)
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.utils import check_array
from sklearn.utils.validation import (
    check_is_fitted,
)

from ..algorithms import mvn_inference, posterior_conditional, it_sampling, kde_params
from ..tmaps import TransportMap

__all__ = ["BEL"]


class BEL(TransformerMixin, MultiOutputMixin, BaseEstimator):
    """Heart of the framework.
    Inherits from scikit-learn base classes.
    BEL stands for Bayesian Evidential Learning.
    """

    def __init__(
        self,
        mode: str = "tm",
        copy: bool = True,
        *,
        X_pre_processing=None,
        Y_pre_processing=None,
        X_post_processing=None,
        Y_post_processing=None,
        regression_model=None,
        n_comp_cca=None,
        x_dim=None,
        y_dim=None,
        random_state=None,
    ):
        """Initialize the BEL class.

        :param mode: How to infer the posterior distribution (if CCA is used). "kde", "mvn" or "tm".
        :param copy: Whether to copy arrays or not (default is True).
        :param X_pre_processing: sklearn pipeline for pre-processing the predictor.
        :param Y_pre_processing: sklearn pipeline for pre-processing the target.
        :param X_post_processing: sklearn pipeline for post-processing the predictor.
        :param X_post_processing: sklearn pipeline for post-processing the target.
        :param regression_model: The regression model to use. Default is Canonical Correlation Analysis.
        :param n_comp_cca: Number of components to keep in CCA (only if CCA is used).
        :param x_dim: Predictor original dimensions.
        :param y_dim: Target original dimensions.
        :param random_state: Seed to reproduce the same samples.
        """
        self.copy = copy
        # How to infer the posterior parameters
        self.mode = mode

        # Processing pipelines
        if X_pre_processing is None:
            X_pre_processing = Pipeline([("nothing", "passthrough")])
        if Y_pre_processing is None:
            Y_pre_processing = Pipeline([("nothing", "passthrough")])
        if X_post_processing is None:
            X_post_processing = Pipeline([("nothing", "passthrough")])
        if Y_post_processing is None:
            Y_post_processing = Pipeline([("nothing", "passthrough")])
        if regression_model is None:
            regression_model = Pipeline([("nothing", "passthrough")])

        self.X_pre_processing = X_pre_processing
        self.Y_pre_processing = Y_pre_processing
        self.X_post_processing = X_post_processing
        self.Y_post_processing = Y_post_processing
        self.regression_model = regression_model
        self.n_comp_cca = n_comp_cca
        # Parameters for sampling
        self._seed = random_state

        # Original dataset dimensions
        self.x_dim, self.y_dim = x_dim, y_dim

        # Pre-processed predictor training data, possibly dimension reduced
        self._x_pre_processed, self._y_pre_processed = None, None
        # Pre-processed observation predictor, possibly dimension reduced
        self._x_obs_pre_processed = None

    # The following properties are central to the BEL framework
    @property
    def X_shape(self):
        """Predictor original shape."""
        return self._X_shape

    @X_shape.setter
    def X_shape(self, x_shape):
        self._X_shape = x_shape

    @property
    def Y_shape(self):
        """Target original shape."""
        return self._Y_shape

    @Y_shape.setter
    def Y_shape(self, y_shape):
        self._Y_shape = y_shape

    @property
    def n_posts(self):
        """Number of sample to extract from the posterior multivariate
        distribution after post-processing."""
        return self._n_posts

    @n_posts.setter
    def n_posts(self, n_p):
        self._n_posts = n_p

    @property
    def seed(self):
        """Seed a.k.a.

        random state to reproduce the same samples
        """
        return self._seed

    @seed.setter
    def seed(self, s):
        self._seed = s
        np.random.seed(self._seed)

    @property
    def random_state(self):
        """Seed a.k.a.

        random state to reproduce the same samples
        """
        return self._seed

    @random_state.setter
    def random_state(self, s):
        self._seed = s

    @property
    def x_pre_processed(self):
        """Pre-processed predictor."""
        return self._x_pre_processed

    @x_pre_processed.setter
    def x_pre_processed(self, x_pre_processed):
        self._x_pre_processed = x_pre_processed

    @property
    def y_pre_processed(self):
        """Pre-processed target."""
        return self._y_pre_processed

    @y_pre_processed.setter
    def y_pre_processed(self, y_pre_processed):
        self._y_pre_processed = y_pre_processed

    @property
    def x_observation(self):
        """Pre-processed observation."""
        return self._x_obs_pre_processed

    @x_observation.setter
    def x_observation(self, x_observation):
        self._x_obs_pre_processed = x_observation

    def fit(self, X: np.array = None, Y: np.array = None):
        """Fit all pipelines.

        :param X: Predictor array.
        :param Y: Target array.
        :return: self
        """

        if self.x_pre_processed is None:
            _xt = self.X_pre_processing.fit_transform(X)
        else:
            _xt = self.x_pre_processed

        if self.y_pre_processed is None:
            _yt = self.Y_pre_processing.fit_transform(Y)
        else:
            _yt = self.y_pre_processed

        # Regression
        try:
            if self.n_comp_cca is None:  # If not specified, use all components
                self.regression_model.n_components = min(_xt.shape[1], _yt.shape[1])
            else:
                self.regression_model.n_components = self.n_comp_cca
            _xc, _yc = self.regression_model.fit_transform(X=_xt, y=_yt)  # Learning
        except ValueError:  # If no CCA
            _xc, _yc = _xt, _yt

        # CV Normalized
        _xf, _yf = (
            self.X_post_processing.fit_transform(_xc),
            self.Y_post_processing.fit_transform(_yc),
        )  # Post-processing

        self.X_f, self.Y_f = _xf, _yf  # At the moment, we have to save these.

        return self

    def transform(self, X=None, Y=None) -> (np.array, np.array):
        """Transform data across all pipelines.

        :param X: Predictor array.
        :param Y: Target array.
        :return: Post-processed variables
        """

        if X is not None and Y is None:  # If only X is provided
            _xt = self.X_pre_processing.transform(X)  # Pre-processing
            _xc = self.regression_model.transform(X=_xt)  # CCA
            _xp = self.X_post_processing.transform(_xc)  # Post-processing

            return _xp

        elif Y is not None and X is None:  # If only Y is provided
            _yt = self.Y_pre_processing.transform(Y)
            dummy = np.zeros(
                (1, self.regression_model.x_loadings_.shape[0])
            )  # Dummy used for CCA
            _, _yc = self.regression_model.transform(
                X=dummy, Y=_yt
            )  # CCA. We only need the Y-loadings, so we pass dummy X
            _yp = self.Y_post_processing.transform(_yc)

            return _yp

        else:  # If both X and Y are provided
            _xt = self.X_pre_processing.transform(X)  # Pre-processing
            _yt = self.Y_pre_processing.transform(Y)

            _xc, _yc = self.regression_model.transform(X=_xt, Y=_yt)

            _xp, _yp = (
                self.X_post_processing.transform(_xc),
                self.Y_post_processing.transform(_yc),
            )

            return _xp, _yp

    def fit_transform(self, X, y=None, **fit_params):
        """Fit-Transform across all pipelines.

        :param X: Predictor array.
        :param y: Target array.
        :return: If mode == "mvn" - returns the posterior mean and covariance. If mode == "kde" - returns a dictionary
            of functions.
        """

        return self.fit(X, y).transform(X, y)

    def predict(
        self,
        X_obs: np.array = None,
        n_posts: int = None,
        mode: str = None,
        noise: float = None,
        return_samples: bool = True,
        inverse_transform: bool = True,
        precomputed_kde: np.array = None,
        dtype: str = "float64",
    ) -> np.array:
        """Predict the posterior distribution of the target variable.

        :param X_obs: The observed data.
        :param n_posts: The number of posterior samples to draw.
        :param mode: The mode of inference to use. Default is "tm".
        :param noise: The noise level of the model (only if mode == 'mvn').
        :param return_samples: Option to return samples or not. Default=True.
        :param inverse_transform: Option to return the samples in the original space. If the dimensionality of the
            original space is very high, this can be memory-consuming. It can be set to False to return the samples in the
            reduced space, which is much faster, so that the samples can be back-transformed later. Default=True.
        :param precomputed_kde: (if mode="kde) Precomputed KDE functions. Computing the KDEs can be time-consuming.
            If the KDEs are precomputed, they can be passed as an argument.
        :param dtype: The data type of the samples. Default=float64.
        :return: The posterior samples in the original space or in the transformed space.
        """
        if mode is not None:  # If mode is provided
            self.mode = mode

        if noise is None:
            self.noise = 0.01

        if n_posts is not None:  # If n_posts is provided
            self.n_posts = n_posts  # Set the number of posterior samples

        # Project observed data into canonical space.
        if self._x_obs_pre_processed is None:
            X_obs_pc = self.X_pre_processing.transform(X_obs)
        else:
            X_obs_pc = self._x_obs_pre_processed

        X_obs_c = self.regression_model.transform(
            X_obs_pc
        )  # Project observed data into Canonical space.
        X_obs_f = self.X_post_processing.transform(X_obs_c)
        self.X_obs_f = X_obs_f

        # Estimate the posterior mean and covariance
        n_obs = X_obs_f.shape[0]  # Number of observations
        n_cca = self.regression_model.n_components  # Number of canonical variables

        if self.mode == "mvn":  # If mode is mvn
            self.posterior_mean, self.posterior_covariance = (
                np.zeros((n_obs, n_cca)),
                np.zeros((n_obs, n_cca, n_cca)),
            )
            for n, dp in enumerate(X_obs_f):  # For each observation point
                # Evaluate the covariance in d (here we assume no data error, so C is identity times a given factor)
                # Number of PCA components for the curves
                x_dim = self.X_pre_processing[
                    "pca"
                ].n_components  # Number of PCA components
                # I matrix. (n_comp_PCA, n_comp_PCA)
                x_cov = (
                    np.eye(x_dim) * self.noise
                )  # Noise level. We assume that the data is noisy with a given level of noise.
                # (n_comp_CCA, n_comp_CCA)
                # Get the rotation matrices
                x_rotations = self.regression_model.x_rotations_
                x_cov = x_rotations.T @ x_cov @ x_rotations
                dict_args = {"x_cov": x_cov}

                X, Y = self.X_f, self.Y_f
                # mvn_inference is designed to accept 1 observation at a time.
                post_mean, post_cov = mvn_inference(
                    X=X,
                    Y=Y,
                    X_obs=dp.reshape(1, -1),
                    **dict_args,
                )  # Posterior mean and covariance
                self.posterior_mean[n] = post_mean
                self.posterior_covariance[n] = post_cov

        elif self.mode == "kde":  # KDE
            self.kde_functions = np.zeros(
                (n_obs, n_cca), dtype="object"
            )  # KDE functions

            if precomputed_kde is not None:  # If precomputed KDE functions are provided
                self.kde_functions = precomputed_kde

            if not np.all(self.kde_functions):  # If KDE functions are not provided
                # KDE inference
                for comp_n in range(n_cca):
                    # If the relation is almost perfectly linear, it doesn't make sense to perform a
                    # KDE estimation.
                    corr = np.corrcoef(self.X_f.T[comp_n], self.Y_f.T[comp_n]).diagonal(
                        offset=1
                    )[0]
                    # If the Pearson's correlation coefficient is > 0.999, linear regression is used instead of KDE.
                    if corr >= 0.999:  # If the relation is almost perfectly linear
                        kind = "linear"
                        fun = LinearRegression().fit(
                            self.X_f.T[comp_n].reshape(-1, 1),
                            self.Y_f.T[comp_n].reshape(-1, 1),
                        )  # Linear regression
                        bw = 0  # Bandwidth is not used in this case.
                        # The KDE inference method can be hybrid - the returned functions are saved as a dictionary
                        sample_fun = {"kind": kind, "function": fun, "bandwidth": bw}
                        functions = [sample_fun] * n_obs
                    else:  # If the relation is not perfectly linear
                        # Compute KDE
                        dens, support, bw = kde_params(
                            x=self.X_f.T[comp_n], y=self.Y_f.T[comp_n]
                        )
                        # Rule of thumb:
                        dens[dens < 1e-8] = 0  # Remove the small values
                        functions = []
                        for n, dp in enumerate(X_obs_f):  # For each observation point
                            # Conditional:
                            hp, sup = posterior_conditional(
                                X_obs=dp.T[comp_n],
                                dens=dens,
                                support=support,
                                k=2 ** 7 + 1,
                            )
                            hp[np.abs(hp) < 1e-8] = 0  # Set very small values to 0.
                            kind = "pdf"
                            fun = interpolate.interp1d(
                                sup, hp, kind="linear"
                            )  # Interpolate
                            # The KDE inference method can be hybrid - the returned functions are saved as a dictionary
                            sample_fun = {
                                "kind": kind,
                                "function": fun,
                                "bandwidth": bw,
                            }
                            functions.append(sample_fun)  # Save the function

                    # Shape = (n_obs, n_comp_CCA)
                    self.kde_functions[:, comp_n] = functions  # noqa

        elif self.mode == "tm":
            nonmonotone = [
                [
                    [],
                    [0],
                    [0, 0, "HF"],
                    [0, 0, 0, "HF"],
                    [0, 0, 0, 0, "HF"],
                    [0, 0, 0, 0, 0, "HF"],
                    [0, 0, 0, 0, 0, 0, "HF"],
                    [0, 0, 0, 0, 0, 0, 0, "HF"],
                ]
            ]

            monotone = [
                [
                    [1],
                    "iRBF 1",
                    "iRBF 1",
                    "iRBF 1",
                    "iRBF 1",
                    "iRBF 1",
                    "iRBF 1",
                    "iRBF 1",
                    "iRBF 1",
                ]
            ]

            self.tm_functions = np.zeros((n_obs, n_cca), dtype="object")
            for comp_n in range(n_cca):
                X = np.vstack((self.X_f.T[comp_n], self.Y_f.T[comp_n])).T
                # Initialize a transport map object
                tm = TransportMap(
                    monotone=monotone,
                    nonmonotone=nonmonotone,
                    X=X,
                    polynomial_type="probabilist's hermite",
                    monotonicity="separable monotonicity",
                    standardize_samples=True,
                    workers=1,
                )  # Number of workers for the parallel optimization; 1 means no parallelization

                tm.optimize()

                kind = "tm"
                sample_fun = {"kind": kind, "function": tm, "X": X}
                functions = [sample_fun] * n_obs

                # Shape = (n_obs, n_comp_CCA)
                self.tm_functions[:, comp_n] = functions  # noqa

        if return_samples:
            samples = self.random_sample(
                X_obs_f=X_obs_f,
                n_posts=n_posts,
                mode=mode,
                init_kde=precomputed_kde,
            )  # Samples from the posterior
            if inverse_transform:
                return self.inverse_transform(samples, dtype=dtype)  # Inverse transform
            else:
                return samples  # Return samples

    def random_sample(
        self,
        X_obs_f: None,
        obs_n: int = None,
        n_posts: int = None,
        mode: str = None,
        init_kde: np.array = None,
    ) -> np.array:
        """Random sample the inferred posterior distribution. It can be used to
        generate samples from the posterior.

        :param X_obs_f: Observed data points in the feature space. Shape = (n_obs, n_comp_CCA)
        :param obs_n: If we want to generate samples from the posterior of a specific observation point, obs_n is the
            index of the observation point.
        :param n_posts: Number of posterior samples
        :param mode: How to sample the posterior distribution
        :param init_kde: Initial KDE function. If None, the KDE function is computed from the observed data.
        :return: Samples from the posterior distribution (n_obs, n_posts, n_comp_CCA)
        """
        if mode is not None:
            self.mode = mode

        # Set the seed for later use
        if self.seed is None:
            self.seed = np.random.randint(2 ** 32 - 1, dtype="uint32")

        if X_obs_f is None:
            X_obs_f = self.X_obs_f

        check_is_fitted(self.regression_model)
        if n_posts is None:
            n_posts = self.n_posts
        else:
            self.n_posts = n_posts
        # Draw n_posts random samples from the multivariate normal distribution :
        # Pay attention to the transpose operator
        np.random.seed(self.seed)

        if self.mode == "mvn":  # Multivariate normal distribution
            if obs_n is not None:  # If we have a specific observation
                post_mn = self.posterior_mean[obs_n].reshape(1, -1)
                post_cv = self.posterior_covariance[obs_n].reshape(1, -1)
            else:
                post_mn = self.posterior_mean
                post_cv = self.posterior_covariance

            Y_samples = []  # Samples from the posterior
            for n, (mean, cov) in enumerate(zip(post_mn, post_cv)):
                Y_samples.append(np.random.multivariate_normal(mean, cov, n_posts))
                Y_samples.append(
                    np.random.multivariate_normal(mean=mean, cov=cov, size=n_posts)
                )  # Draw n_posts samples from the multivariate normal distribution

        if self.mode == "kde":  # Kernel density estimation
            n_obs = X_obs_f.shape[0]  # Number of observations
            Y_samples = np.zeros(
                (n_obs, self.n_posts, self.kde_functions.shape[1])
            )  # Shape = (n_obs, n_posts, n_comp_CCA)

            if obs_n is not None:  # If we have a specific observation
                kde_fn = self.kde_functions[obs_n].reshape(
                    1, -1
                )  # Shape = (1, n_comp_CCA)
            else:
                kde_fn = self.kde_functions  # Shape = (n_obs, n_comp_CCA)

            if init_kde is None:
                # Parses the functions dict
                for i, fun_per_comp in enumerate(kde_fn):
                    for j, fun in enumerate(fun_per_comp):
                        if fun["kind"] == "pdf":  # If the function is a pdf
                            pdf = fun["function"]
                            uniform_samples = it_sampling(  # Sample from the pdf
                                pdf=pdf,
                                num_samples=self.n_posts,
                                lower_bd=pdf.x.min(),
                                upper_bd=pdf.x.max(),
                                k=2 ** 7 + 1,
                            )
                        elif (
                            fun["kind"] == "linear"
                        ):  # If the function is a linear interpolation
                            rel1d = fun["function"]
                            uniform_samples = np.ones(self.n_posts) * rel1d.predict(
                                np.array(
                                    X_obs_f[i][j].reshape(1, -1)
                                )  # check this line
                            )  # Shape X_obs_f = (n_obs, n_components)

                        Y_samples[i, :, j] = uniform_samples  # noqa
            else:  # If the KDE is already initialized
                for i, fun_per_comp in enumerate(kde_fn):  # Parses the function dict
                    for j, fun in enumerate(fun_per_comp):
                        pv = init_kde[i, j]
                        if fun["kind"] == "pdf":
                            pdf = fun["function"]
                            uniform_samples = it_sampling(
                                pdf=pdf,
                                num_samples=self.n_posts,
                                lower_bd=pdf.x.min(),
                                upper_bd=pdf.x.max(),
                                k=2 ** 7 + 1,
                                cdf_y=pv,
                            )
                        elif fun["kind"] == "linear":
                            uniform_samples = np.ones(self.n_posts) * pv

                        Y_samples[i, :, j] = uniform_samples  # noqa

        if self.mode == "tm":
            n_obs = X_obs_f.shape[0]  # Number of observations
            Y_samples = np.zeros(
                (n_obs, self.n_posts, self.tm_functions.shape[1])
            )  # Shape = (n_obs, n_posts, n_comp_CCA)

            if obs_n is not None:  # If we have a specific observation
                tm_fn = self.tm_functions[obs_n].reshape(
                    1, -1
                )  # Shape = (1, n_comp_CCA)
            else:
                tm_fn = self.tm_functions  # Shape = (n_obs, n_comp_CCA)

            # Parses the functions dict
            for i, fun_per_comp in enumerate(tm_fn):
                for j, fun in enumerate(fun_per_comp):
                    tm = fun["function"]
                    X = fun["X"]
                    N = X.shape[0]
                    # Map the target samples to the reference distribution; we only get a N-by-1
                    # vector because we only defined the map for the second dimension/column of X
                    if self.n_posts == N:
                        norm_samples = tm.map(X)
                    else:
                        norm_samples = stats.norm.rvs(size=(self.n_posts, 1))
                        N = self.n_posts
                    # Now define the value we wish to condition on
                    x1_obs = X_obs_f[i][j].reshape(-1)[0]  # our 'observed' value

                    # In the inversion, we pretend that we have already inverted the first dimension
                    # of X and obtained x1_obs, so we create fake, pre-calculated values for it
                    X_precalc = np.ones((N, 1)) * x1_obs

                    # Now invert the map conditionally. X_star are the posterior samples.
                    X_star = tm.inverse_map(
                        X_precalc=X_precalc, Y=norm_samples
                    )  # Only necessary when heuristic is deactivated
                    Y_samples[i, :, j] = X_star.reshape(-1)  # noqa

        return Y_samples  # noqa

    def kde_init(self, X_obs_f: np.array, obs_n: int = None):
        """Initialize the KDEs, i.e. the functions that will be used to sample
        from the posterior distribution.

        :param X_obs_f: Observed data points
        :param obs_n: Observation number
        :return: The initialized KDEs
        """
        n_obs = X_obs_f.shape[0]  # Number of observations
        n_comp = X_obs_f.shape[1]  # Number of components
        init_samples = np.zeros(
            (n_obs, n_comp), dtype="object"
        )  # Shape = (n_obs, n_comp)

        if obs_n is not None:  # If we have a specific observation
            kde_fn = self.kde_functions[obs_n].reshape(1, -1)  # Shape = (1, n_comp_CCA)
        else:
            kde_fn = self.kde_functions  # Shape = (n_obs, n_comp_CCA)

        # Parses the functions dict
        for i, fun_per_comp in enumerate(kde_fn):
            for j, fun in enumerate(fun_per_comp):
                if fun["kind"] == "pdf":
                    pdf = fun["function"]
                    pv = it_sampling(  # Sample from the pdf
                        pdf=pdf,
                        lower_bd=pdf.x.min(),
                        upper_bd=pdf.x.max(),
                        k=2 ** 7
                        + 1,  # Number of samples. It is a power of 2 + 1 because Romberg integration will be used
                        return_cdf=True,
                    )
                elif fun["kind"] == "linear":
                    rel1d = fun["function"]
                    pv = rel1d.predict(  # Sample from the linear interpolation
                        np.array(X_obs_f[i][j].reshape(1, -1))
                    )

                init_samples[i, j] = pv  # noqa
        return init_samples

    def inverse_transform(
        self, Y_pred: np.array, dtype: str = "float64", get_PC=False
    ) -> np.array:
        """Back-transforms the posterior samples Y_pred to their physical
        space.

        :param Y_pred: The posterior samples (shape = (n_obs, n_components, n_samples))
        :param get_PC: If True, returns the canonical variates
        :param dtype: The dtype of the output array
        :return: The back-transformed samples
        """
        check_is_fitted(self.regression_model)

        Y_post = []

        for i, yp in enumerate(Y_pred):  # For each observed data

            if yp.ndim < 2:  # If we have only one component
                yp = yp.reshape(1, -1)

            yp = check_array(yp)

            y_post = self.Y_post_processing.inverse_transform(
                yp
            )  # Posterior CCA scores

            n_comp = self.regression_model.n_components  # Number of components

            if (
                y_post.shape[1] > n_comp
            ):  # If the number of components is smaller than the number of observations
                y_post = y_post[
                    :, :n_comp
                ]  # Truncate the posterior samples, because the number of components is smaller than the number of
                # observations

            # x_dummy to be used in the inverse_transform of CCA:
            x_dummy = np.zeros((y_post.shape[0], n_comp))
            try:
                x_post_dummy, y_post = self.regression_model.inverse_transform(
                    x_dummy, y_post
                )  # Inverse transform the posterior samples
            except TypeError:
                y_post = (
                    np.matmul(y_post, self.regression_model.y_loadings_.T)
                    * self.regression_model._y_std  # noqa
                    + self.regression_model._y_mean  # noqa
                )  # Posterior PC scores

            if get_PC:  # return the CV
                Y_post.append(y_post)

            else:  # continues back-transforming the posterior samples
                # Back transform PC scores
                y_post_raw = self.Y_pre_processing.inverse_transform(
                    y_post
                )  # Inverse transform

                if (
                    type(y_post_raw) == list
                ):  # If the target contains more than one variable
                    y_post_raw = np.concatenate(
                        [y_raw.reshape(-1) for y_raw in y_post_raw], axis=0
                    )

                Y_post.append(y_post_raw)

        return np.array(Y_post, dtype=dtype)

    def cca_pc_transform(self, X=None, Y=None) -> (np.array, np.array):
        """Transform PCs to CVs.

        :param X: Predictor array.
        :param Y: Target array.
        :return: CVs
        """
        if X is not None and Y is None:  # If only X is provided
            _xc = self.regression_model.transform(X=X)  # CCA
            return _xc

        elif Y is not None and X is None:  # If only Y is provided
            dummy = np.zeros(
                (1, self.regression_model.x_loadings_.shape[0])
            )  # Dummy used for CCA
            _, _yc = self.regression_model.transform(
                X=dummy, Y=Y
            )  # CCA. We only need the Y-loadings, so we pass dummy X
            return _yc
        else:
            _xc, _yc = self.regression_model.transform(X=X, Y=Y)  # CCA
            return _xc, _yc
