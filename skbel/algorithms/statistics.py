#  Copyright (c) 2021. Robin Thibaut, Ghent University

import math
import numpy as np
import pandas as pd
import warnings
from numpy.polynomial.chebyshev import chebfit, chebval, chebint
from numpy.random import uniform
from scipy import integrate, ndimage, interpolate
from scipy.optimize import root
from sklearn.neighbors import KernelDensity
from sklearn.utils import check_array
from sklearn.model_selection import GridSearchCV
from skbel.algorithms.extmath import get_block

__all__ = [
    "KDE",
    "kde_params",
    "posterior_conditional",
    "mvn_inference",
    "it_sampling",
    "normalize",
]


class KDE:
    """
    Bivariate kernel density estimator.
    This class is adapted from the class of the same name in the package Seaborn 0.11.1
    https://seaborn.pydata.org/generated/seaborn.kdeplot.html
    """

    def __init__(
        self,
        *,
        kernel_type: str = None,
        bandwidth: float = None,
        grid_search: bool = True,
        gridsize: int = 100,
        cut: float = 0.2,
        clip: list = None,
    ):
        """Initialize the estimator with its parameters.

        Parameters
        ----------
        gridsize : int, optional
            Number of points on each dimension of the evaluation grid.
        cut : number, optional
            Factor, multiplied by the smoothing bandwidth, that determines how
            far the evaluation grid extends past the extreme datapoints. When
            set to 0, truncate the curve at the data limits.
        clip : pair of numbers None, or a pair of such pairs
            Do not evaluate the density outside of these limits.

        """
        if clip is None:
            clip = None, None

        self.kernel_type = kernel_type
        if kernel_type is None:
            self.kernel_type = "gaussian"
        self.bw = bandwidth
        self.grid_search = grid_search
        self.gridsize = gridsize
        self.cut = cut
        self.clip = clip

        self.support = None

    @staticmethod
    def _define_support_grid(
        x: np.array, bandwidth: float, cut: float, clip: list, gridsize: int
    ):
        """Create the grid of evaluation points depending for vector x."""
        clip_lo = -np.inf if clip[0] is None else clip[0]
        clip_hi = +np.inf if clip[1] is None else clip[1]
        bw = 1 if bandwidth is None else bandwidth
        gridmin = max(x.min() - bw * cut, clip_lo)
        gridmax = min(x.max() + bw * cut, clip_hi)
        return np.linspace(gridmin, gridmax, gridsize)

    def _define_support_univariate(self, x: np.array):
        """Create a 1D grid of evaluation points."""
        grid = self._define_support_grid(x, self.bw, self.cut, self.clip, self.gridsize)
        return grid

    def _define_support_bivariate(self, x1: np.array, x2: np.array):
        """Create a 2D grid of evaluation points."""
        clip = self.clip
        if clip[0] is None or np.isscalar(clip[0]):
            clip = (clip, clip)
        grid1 = self._define_support_grid(x1, self.bw, self.cut, clip[0], self.gridsize)
        grid2 = self._define_support_grid(x2, self.bw, self.cut, clip[1], self.gridsize)

        return grid1, grid2

    def define_support(
        self,
        x1: np.array,
        x2: np.array = None,
        cache: bool = True,
    ):
        """Create the evaluation grid for a given data set."""
        if x2 is None:
            support = self._define_support_univariate(x1)
        else:
            support = self._define_support_bivariate(x1, x2)

        if cache:
            self.support = support

        return support

    def _fit(self, fit_data: np.array):
        """Fit the scikit-learn kde"""
        bw = 1 if self.bw is None else self.bw
        fit_kws = {
            "bandwidth": bw,
            "algorithm": "auto",
            "kernel": self.kernel_type,
            "metric": "euclidean",
            "atol": 1e-4,
            "rtol": 1e-4,
            "breadth_first": True,
            "leaf_size": 40,
            "metric_params": None,
        }
        kde = KernelDensity(**fit_kws)
        if self.grid_search and not self.bw:
            # GridSearchCV maximizes the total log probability density under the model.

            # The data X will be be divided into train-test splits based on folds defined in cv param

            # For each combination of parameters that you specified in param_grid, the model
            # will be trained on the train part from the step above and then scoring will be used on test part.

            # The scores for each parameter combination will be combined for all the folds and averaged.
            # Highest performing parameter combination will be selected.

            grid = GridSearchCV(kde, {"bandwidth": 10 ** np.linspace(-2, 1, 100)})
            grid.fit(fit_data)
            self.bw = grid.best_params_["bandwidth"]
            fit_kws["bandwidth"] = self.bw
            kde.set_params(**{"bandwidth": self.bw})

        kde.fit(fit_data)

        return kde

    def _eval_univariate(self, x: np.array):
        """Fit and evaluate on univariate data."""
        support = self.support
        if support is None:
            support = self.define_support(x, cache=True)

        kde = self._fit(x.reshape(-1, 1))

        density = np.exp(kde.score_samples(support.reshape(-1, 1)))

        return density, support

    def _eval_bivariate(self, x1: np.array, x2: np.array):
        """Fit and evaluate on bivariate data."""
        support = self.support
        if support is None:
            support = self.define_support(x1, x2, cache=False)

        X_train = np.vstack([x1, x2]).T

        kde = self._fit(X_train)

        # xx1, xx2 = np.meshgrid(*support)
        # grid1, grid2 = support
        # X, Y = np.meshgrid(grid1, grid2[::-1])
        X, Y = np.meshgrid(*support)
        # grid = np.vstack([Y.ravel(), X.ravel()]).T
        grid = np.vstack([X.ravel(), Y.ravel()]).T

        density = np.exp(kde.score_samples(grid))
        density = density.reshape(X.shape)
        # density = kde.score_samples([xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)

        return density, support

    def __call__(self, x1, x2=None):
        """Fit and evaluate on univariate or bivariate data."""
        if x2 is None:
            return self._eval_univariate(x1)
        else:
            return self._eval_bivariate(x1, x2)


def _univariate_density(
    data_variable: pd.DataFrame,
    estimate_kws: dict,
):
    # Initialize the estimator object
    estimator = KDE(**estimate_kws)

    all_data = data_variable.dropna()

    # Extract the data points from this sub set and remove nulls
    sub_data = all_data.dropna()
    observations = sub_data[data_variable]

    observation_variance = observations.var()
    if math.isclose(observation_variance, 0) or np.isnan(observation_variance):
        msg = "Dataset has 0 variance; skipping density estimate."
        warnings.warn(msg, UserWarning)

    # Estimate the density of observations at this level
    density, support = estimator(observations)

    return density, support, estimator.bw


def _bivariate_density(
    data: pd.DataFrame,
    estimate_kws: dict,
):
    """
    Estimate bivariate KDE
    :param data: DataFrame containing (x, y) data
    :param estimate_kws: KDE parameters
    :return:
    """

    estimator = KDE(**estimate_kws)

    # Extract the data points from this sub set and remove nulls
    sub_data = data.dropna()
    observations = sub_data[["x", "y"]]

    # Check that KDE will not error out
    variance = observations[["x", "y"]].var()
    if any(math.isclose(x, 0) for x in variance) or variance.isna().any():
        msg = "Dataset has 0 variance; skipping density estimate."
        warnings.warn(msg, UserWarning)

    # Estimate the density of observations at this level
    observations = observations["x"], observations["y"]
    density, support = estimator(*observations)

    # # Transform the support grid back to the original scale
    # xx, yy = support
    #
    # support = xx, yy

    return density, support, estimator.bw


def kde_params(
    x: np.array = None,
    y: np.array = None,
    bw: float = None,
    gridsize: int = 200,
    cut: float = 0.2,
    clip=None,
):
    """
    Computes the kernel density estimate (KDE) of one or two data sets.

    Parameters
    ----------
    x : array_like
        The x-coordinates of the input data.
    y : array_like, optional
        The y-coordinates of the input data.
    gridsize : int, optional
        Number of discrete points in the evaluation grid.
    cut : scalar, optional
        Draw the estimate to cut * bw from the extreme data points.
    clip : pair of scalars, or pair of pair of scalars, optional
        Lower and upper bounds for datapoints used to fit KDE. Can provide
        a pair of (low, high) bounds for bivariate plots.
    Returns
    -------
    density : ndarray
        The estimated probability density function evaluated at the support.
    support : ndarray
        The support of the density function, the x-axis of the KDE.

    """

    data = {"x": x, "y": y}
    frame = pd.DataFrame(data=data)

    # Pack the kwargs for KDE
    estimate_kws = dict(
        bandwidth=bw,
        gridsize=gridsize,
        cut=cut,
        clip=clip,
    )

    if y is None:
        density, support, bw = _univariate_density(
            data_variable=frame, estimate_kws=estimate_kws
        )

    else:
        density, support, bw = _bivariate_density(
            data=frame,
            estimate_kws=estimate_kws,
        )

    return density, support, bw


def _pixel_coordinate(line: list, x_1d: np.array, y_1d: np.array):
    """
    Gets the pixel coordinate of the value x or y, in order to get posterior conditional probability given a KDE.
    :param line: Coordinates of the line we'd like to sample along [(x1, y1), (x2, y2)]
    :param x_1d: List of x coordinates along the axis
    :param y_1d: List of y coordinates along the axis
    :return:
    """
    # https://stackoverflow.com/questions/18920614/plot-cross-section-through-heat-map
    # Convert the line to pixel/index coordinates
    x_world, y_world = np.array(list(zip(*line)))
    col = y_1d.shape * (x_world - min(x_1d)) / x_1d.ptp()
    row = x_1d.shape * (y_world - min(y_1d)) / y_1d.ptp()

    # Interpolate the line at "num" points...
    num = 200
    row, col = [np.linspace(item[0], item[1], num) for item in [row, col]]

    return row, col


def _conditional_distribution(
    kde_array: np.array,
    x_array: np.array,
    y_array: np.array,
    x: float = None,
    y: float = None,
):
    """
    Compute the conditional posterior distribution p(x_array|y_array) given x or y.
    Provide only one observation ! Either x or y.
    Perform a cross-section in the KDE along the y axis.
    :param x: Observed data (horizontal axis)
    :param y: Observed data (vertical axis)
    :param kde_array: KDE of the prediction
    :param x_array: X grid (1D)
    :param y_array: Y grid (1D)
    :return:
    """

    # Coordinates of the line we'd like to sample along
    if x is not None:
        line = [(x, min(y_array)), (x, max(y_array))]
    elif y is not None:
        line = [(min(x_array), y), (max(x_array), y)]
    else:
        msg = "No observation point included."
        warnings.warn(msg, UserWarning)
        return 0

    # Convert line to row/column
    row, col = _pixel_coordinate(line=line, x_1d=x_array, y_1d=y_array)

    # Extract the values along the line, using cubic interpolation
    zi = ndimage.map_coordinates(kde_array, np.vstack((row, col)))

    return zi


def _normalize_distribution(post: np.array, support: np.array):
    """
    When a cross-section is performed along a bivariate KDE, the integral might not = 1.
    This function normalizes such functions so that their integral = 1.
    :param post: Values of the KDE cross-section
    :param support: Corresponding support
    :return:
    """
    a = integrate.simps(y=np.abs(post), x=support)

    if np.abs(a - 1) > 1e-4 and a > 0:  # Rule of thumb
        try:
            post *= 1 / a
        except RuntimeWarning:  # Division by zero
            pass

    return post


def posterior_conditional(
    X_obs: float = None,
    Y_obs: float = None,
    dens: np.array = None,
    support: np.array = None,
) -> (np.array, np.array):
    """
    Computes the posterior distribution p(y|x_obs) or p(x|y_obs) by doing a cross section of the KDE of (d, h).
    :param X_obs: Observation (predictor, x-axis)
    :param Y_obs: Observation (target, y-axis)
    :param dens: The density values of the KDE of (X, Y).
    :param support: The support grid of the KDE of (X, Y).
    :return:
    """
    # Grid parameters
    xg, yg = support

    if X_obs is not None:
        support = yg
        # Extract the density values along the line, using cubic interpolation
        if type(X_obs) is list or type(X_obs) is tuple:
            X_obs = X_obs[0]
        post = _conditional_distribution(
            x=X_obs, x_array=xg, y_array=yg, kde_array=dens
        )
    elif Y_obs is not None:
        support = xg
        # Extract the density values along the line, using cubic interpolation
        if type(Y_obs) is list or tuple:
            Y_obs = X_obs[0]
        post = _conditional_distribution(
            y=Y_obs, x_array=xg, y_array=yg, kde_array=dens
        )

    else:
        msg = "No observation point included."
        warnings.warn(msg, UserWarning)
        return 0

    post = _normalize_distribution(post, support)

    return post, support


def mvn_inference(
    X: np.array, Y: np.array, X_obs: np.array, **kwargs
) -> (np.array, np.array):
    """
    Estimating posterior mean and covariance of the target.
    .. [1] A. Tarantola. Inverse Problem Theory and Methods for Model Parameter Estimation.
           SIAM, 2005. Pages: 70-71
    :param X: Canonical Variate of the training data
    :param Y: Canonical Variate of the training target, gaussian-distributed
    :param X_obs: Canonical Variate of the observation (n_samples, n_features).
    Note that in this implementation, n_samples must be = 1.
    :return: y_posterior_mean, y_posterior_covariance
    """

    Y = check_array(Y, copy=True, ensure_2d=False)
    X = check_array(X, copy=True, ensure_2d=False)
    X_obs = check_array(X_obs, copy=True, ensure_2d=False)

    # Size of the set
    n_training = X.shape[0]

    # Computation of the posterior mean in Canonical space
    y_mean = np.mean(Y, axis=0)  # (n_comp_CCA, 1)
    # Mean is 0, as expected.
    y_mean = np.where(np.abs(y_mean) < 1e-12, 0, y_mean)

    # Evaluate the covariance in h (in Canonical space)
    # Very close to the Identity matrix
    # (n_comp_CCA, n_comp_CCA)
    y_cov = np.cov(Y.T)

    if "x_cov" in kwargs.keys():
        x_cov = kwargs["x_cov"]
    else:
        x_cov = np.zeros(shape=y_cov.shape)

    # Linear modeling h to d (in canonical space) with least-square criterion.
    # Pay attention to the transpose operator.
    # Computes the vector g that approximately solves the equation y @ g = x.
    g = np.linalg.lstsq(Y, X, rcond=None)[0].T
    # Replace values below threshold by 0.
    g = np.where(np.abs(g) < 1e-12, 0, g)  # (n_comp_CCA, n_comp_CCA)

    # Modeling error due to deviations from theory
    # (n_components_CCA, n_training)
    x_ls_predicted = np.matmul(Y, g.T)
    x_modeling_mean_error = np.mean(X - x_ls_predicted, axis=0)  # (n_comp_CCA, 1)
    x_modeling_error = (
        X - x_ls_predicted - np.tile(x_modeling_mean_error, (n_training, 1))
    )
    # (n_comp_CCA, n_training)

    # Information about the covariance of the posterior distribution in Canonical space.
    x_modeling_covariance = np.cov(x_modeling_error.T)  # (n_comp_CCA, n_comp_CCA)

    # Build block matrix
    s11 = y_cov
    if y_cov.ndim == 0:
        y_cov = [y_cov]
    s12 = y_cov @ g.T
    s21 = g @ y_cov
    s22 = g @ y_cov @ g.T + x_cov + x_modeling_covariance
    block = np.block([[s11, s12], [s21, s22]])

    # Inverse
    delta = np.linalg.pinv(block)
    # Partition block
    d11 = get_block(delta, 1)
    d12 = get_block(delta, 2)

    # Observe that posterior covariance does not depend on observed x.
    y_posterior_covariance = np.linalg.pinv(d11)  # (n_comp_CCA, n_comp_CCA)
    # Computing the posterior mean is simply a linear operation, given precomputed posterior covariance.
    y_posterior_mean = y_posterior_covariance @ (
        d11 @ y_mean - d12 @ (X_obs[0] - x_modeling_mean_error - y_mean @ g.T)
    )  # (n_comp_CCA,)

    return y_posterior_mean, y_posterior_covariance


def _convergent(quadrature):
    msg = "The integral is probably divergent, or slowly convergent."
    if len(quadrature) > 3 and quadrature[3] == msg:
        return False
    else:
        return True


def normalize(pdf, lower_bd=-np.inf, upper_bd=np.inf, vectorize=False):
    """Normalize a non-normalized PDF.

    Parameters
    ----------
    pdf : function, float -> float
        The probability density function (not necessarily normalized). Must take
        floats or ints as input, and return floats as an output.
    lower_bd : float
        Lower bound of the support of the pdf. This parameter allows one to
        manually establish cutoffs for the density.
    upper_bd : float
        Upper bound of the support of the pdf.
    vectorize: boolean
        Vectorize the function. This slows down function calls, and so is
        generally set to False.

    Returns
    -------
    pdf_norm : function
        Function with same signature as pdf, but normalized so that the integral
        between lower_bd and upper_bd is close to 1. Maps nicely over iterables.
    """
    if lower_bd >= upper_bd:
        raise ValueError("Lower bound must be less than upper bound.")
    quadrature = integrate.quad(pdf, lower_bd, upper_bd, full_output=1)
    if not _convergent(quadrature):
        raise ValueError("PDF integral likely divergent.")
    A = quadrature[0]

    def pdf_normed(x):
        if lower_bd <= x <= upper_bd:
            return pdf(x) / A
        else:
            return 0

    if vectorize:

        def pdf_vectorized(x):
            try:
                return pdf_normed(x)
            except ValueError:
                return np.array([pdf_normed(xi) for xi in x])

        return pdf_vectorized
    else:
        return pdf_normed


def get_cdf(pdf, lower_bd=-np.inf, upper_bd=np.inf):
    """Generate a CDF from a (possibly not normalized) pdf.

    Parameters
    ----------
    pdf : function, float -> float
        The probability density function (not necessarily normalized). Must take
        floats or ints as input, and return floats as an output.
    lower_bd : float
        Lower bound of the support of the pdf. This parameter allows one to
        manually establish cutoffs for the density.
    upper_bd : float
        Upper bound of the support of the pdf.

    Returns
    -------
    cdf : function
        The cumulative density function of the (normalized version of the)
        provided pdf. Will return a float if provided with a float or int; will
        return a numpy array if provided with an iterable.

    """
    pdf_norm = normalize(pdf, lower_bd, upper_bd)

    def cdf_number(x):
        """Numerical cdf"""
        if x < lower_bd:
            return 0.0
        elif x > upper_bd:
            return 1.0
        else:
            return integrate.quad(
                pdf_norm, lower_bd, x, epsabs=1e-3, epsrel=1e-3, limit=50
            )[0]

    def cdf_vector(x):
        try:
            return np.array([cdf_number(xi) for xi in x])
        except AttributeError:
            return cdf_number(x)

    return cdf_vector


def it_sampling(pdf, num_samples, lower_bd=-np.inf, upper_bd=np.inf):
    """Sample from an arbitrary, unnormalized PDF.
    Copyright (c) 2017 Peter Wills

    Parameters
    ----------
    pdf : function, float -> float
        The probability density function (not necessarily normalized). Must take
        floats or ints as input, and return floats as an output.
    num_samples : int
        The number of samples to be generated.
    lower_bd : float
        Lower bound of the support of the pdf. This parameter allows one to
        manually establish cutoffs for the density.
    upper_bd : float
        Upper bound of the support of the pdf.

    Returns
    -------
    samples : numpy array
        An array of samples from the provided PDF, with support between lower_bd
        and upper_bd.

    Notes
    -----
    For a unimodal distribution, the mode is a good choice for the parameter
    guess. Any number for which the CDF is not extremely close to 0 or 1 should
    be acceptable. If the cdf(guess) is near 1 or 0, then its derivative is near 0,
    and so the numerical root finder will be very slow to converge.

    This sampling technique is slow (~3 ms/sample for a unit normal with initial
    guess of 0), since we re-integrate to get the CDF at every iteration of the
    numerical root-finder. This is improved somewhat by using Chebyshev
    approximations of the CDF, but the sampling rate is still prohibitively slow
    for >100k samples.

    """
    seeds = uniform(0, 1, num_samples)
    cdf = get_cdf(pdf, lower_bd, upper_bd)
    cdf_y = cdf(np.linspace(lower_bd, upper_bd, 200))
    inverse_cdf = interpolate.interp1d(
        cdf_y, pdf.x, kind="linear", fill_value="extrapolate"
    )
    simple_samples = inverse_cdf(seeds)
    return simple_samples
