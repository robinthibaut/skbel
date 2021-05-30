#  Copyright (c) 2021. Robin Thibaut, Ghent University

import math
import warnings

import numpy as np
import pandas as pd
from scipy import integrate, ndimage, stats
from sklearn.utils import check_array

from skbel.algorithms.extmath import get_block

__all__ = ["KDE", "kde_params", "posterior_conditional", "mvn_inference"]


class KDE:
    """
    Bivariate kernel density estimator.
    This class is adapted from the class of the same name in the package Seaborn 0.11.1
    https://seaborn.pydata.org/generated/seaborn.kdeplot.html
    """

    def __init__(
        self,
        *,
        bw_method: str = None,
        bw_adjust: float = 1,
        gridsize: int = 400,
        cut: float = 3,
        clip: list = None,
        cumulative: bool = False,
    ):
        """Initialize the estimator with its parameters.

        Parameters
        ----------
        bw_method : string, scalar, or callable, optional
            Method for determining the smoothing bandwidth to use; passed to
            :class:`scipy.stats.gaussian_kde`.
        bw_adjust : number, optional
            Factor that multiplicatively scales the value chosen using
            ``bw_method``. Increasing will make the curve smoother. See Notes.
        gridsize : int, optional
            Number of points on each dimension of the evaluation grid.
        cut : number, optional
            Factor, multiplied by the smoothing bandwidth, that determines how
            far the evaluation grid extends past the extreme datapoints. When
            set to 0, truncate the curve at the data limits.
        clip : pair of numbers None, or a pair of such pairs
            Do not evaluate the density outside of these limits.
        cumulative : bool, optional
            If True, estimate a cumulative distribution function.

        """
        if clip is None:
            clip = None, None

        self.bw_method = bw_method
        self.bw_adjust = bw_adjust
        self.gridsize = gridsize
        self.cut = cut
        self.clip = clip
        self.cumulative = cumulative

        self.support = None

    @staticmethod
    def _define_support_grid(
        x: np.array, bw: float, cut: float, clip: list, gridsize: int
    ):
        """Create the grid of evaluation points depending for vector x."""
        clip_lo = -np.inf if clip[0] is None else clip[0]
        clip_hi = +np.inf if clip[1] is None else clip[1]
        gridmin = max(x.min() - bw * cut, clip_lo)
        gridmax = min(x.max() + bw * cut, clip_hi)
        return np.linspace(gridmin, gridmax, gridsize)

    def _define_support_univariate(self, x: np.array, weights: np.array):
        """Create a 1D grid of evaluation points."""
        kde = self._fit(x, weights)
        bw = np.sqrt(kde.covariance.squeeze())
        grid = self._define_support_grid(x, bw, self.cut, self.clip, self.gridsize)
        return grid

    def _define_support_bivariate(self, x1: np.array, x2: np.array, weights: np.array):
        """Create a 2D grid of evaluation points."""
        clip = self.clip
        if clip[0] is None or np.isscalar(clip[0]):
            clip = (clip, clip)

        kde = self._fit([x1, x2], weights)
        bw = np.sqrt(np.diag(kde.covariance).squeeze())

        grid1 = self._define_support_grid(x1, bw[0], self.cut, clip[0], self.gridsize)
        grid2 = self._define_support_grid(x2, bw[1], self.cut, clip[1], self.gridsize)

        return grid1, grid2

    def define_support(
        self,
        x1: np.array,
        x2: np.array = None,
        weights: np.array = None,
        cache: bool = True,
    ):
        """Create the evaluation grid for a given data set."""
        if x2 is None:
            support = self._define_support_univariate(x1, weights)
        else:
            support = self._define_support_bivariate(x1, x2, weights)

        if cache:
            self.support = support

        return support

    def _fit(self, fit_data: np.array, weights: np.array = None):
        """Fit the scipy kde"""
        fit_kws = {"bw_method": self.bw_method}
        if weights is not None:
            fit_kws["weights"] = weights

        kde = stats.gaussian_kde(fit_data, **fit_kws)
        kde.set_bandwidth(kde.factor * self.bw_adjust)

        return kde

    def _eval_univariate(self, x: np.array, weights=None):
        """Fit and evaluate on univariate data."""
        support = self.support
        if support is None:
            support = self.define_support(x, cache=True)

        kde = self._fit(x, weights)

        if self.cumulative:
            s_0 = support[0]
            density = np.array([kde.integrate_box_1d(s_0, s_i) for s_i in support])
        else:
            density = kde(support)

        return density, support

    def _eval_bivariate(self, x1: np.array, x2: np.array, weights: np.array = None):
        """Fit and evaluate on bivariate data."""
        support = self.support
        if support is None:
            support = self.define_support(x1, x2, cache=False)

        kde = self._fit([x1, x2], weights)

        if self.cumulative:
            grid1, grid2 = support
            density = np.zeros((grid1.size, grid2.size))
            p0 = min(grid1), min(grid2)
            for i, xi in enumerate(grid1):
                for j, xj in enumerate(grid2):
                    density[i, j] = kde.integrate_box(p0, (xi, xj))

        else:
            xx1, xx2 = np.meshgrid(*support)
            density = kde([xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)

        return density, support

    def __call__(self, x1, x2=None, weights=None):
        """Fit and evaluate on univariate or bivariate data."""
        if x2 is None:
            return self._eval_univariate(x1, weights)
        else:
            return self._eval_bivariate(x1, x2, weights)


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
    density, support = estimator(observations, weights=None)

    return density, support


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

    all_data = data.dropna()

    # Extract the data points from this sub set and remove nulls
    sub_data = all_data.dropna()
    observations = sub_data[["x", "y"]]

    # Check that KDE will not error out
    variance = observations[["x", "y"]].var()
    if any(math.isclose(x, 0) for x in variance) or variance.isna().any():
        msg = "Dataset has 0 variance; skipping density estimate."
        warnings.warn(msg, UserWarning)

    # Estimate the density of observations at this level
    observations = observations["x"], observations["y"]
    density, support = estimator(*observations, weights=None)

    # Transform the support grid back to the original scale
    xx, yy = support

    support = xx, yy

    return density, support


def kde_params(
    x: np.array = None,
    y: np.array = None,
    bw: float = None,
    gridsize: int = 400,
    cut: float = 3,
    clip=None,
    cumulative: bool = False,
    bw_method: str = "scott",
    bw_adjust: int = 1,
):
    """
    Obtain density and support (grid) of the bivariate KDE
    :param x:
    :param y:
    :param bw:
    :param gridsize:
    :param cut:
    :param clip:
    :param cumulative:
    :param bw_method:
    :param bw_adjust:
    :return:
    """

    data = {"x": x, "y": y}
    frame = pd.DataFrame(data=data)

    # Handle deprecation of `bw`
    if bw is not None:
        bw_method = bw

    # Pack the kwargs for statistics.KDE
    estimate_kws = dict(
        bw_method=bw_method,
        bw_adjust=bw_adjust,
        gridsize=gridsize,
        cut=cut,
        clip=clip,
        cumulative=cumulative,
    )

    if y is None:
        density, support = _univariate_density(
            data_variable=frame, estimate_kws=estimate_kws
        )

    else:
        density, support = _bivariate_density(
            data=frame,
            estimate_kws=estimate_kws,
        )

    return density, support


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
    num = 400
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

    if np.abs(a - 1) > 1e-4:  # Rule of thumb
        post *= 1 / a

    return post


def posterior_conditional(
    X: np.array, Y: np.array, X_obs: float = None, Y_obs: float = None
):
    """
    Computes the posterior distribution p(y|x_obs) or p(x|y_obs) by doing a cross section of the KDE of (d, h).
    :param X: Predictor (x-axis)
    :param Y: Target (y-axis)
    :param X_obs: Observation (predictor, x-axis)
    :param Y_obs: Observation (target, y-axis)
    :return:
    """

    # Compute KDE
    dens, support = kde_params(x=X, y=Y)
    # Grid parameters
    xg, yg = support

    if X_obs is not None:
        support = yg
        # Extract the density values along the line, using cubic interpolation
        if type(X_obs) is list or tuple:
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
    :param X_obs: Canonical Variate of the observation
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
