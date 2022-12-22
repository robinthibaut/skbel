#  Copyright (c) 2021. Robin Thibaut, Ghent University

import math
import warnings

import numpy as np
import pandas as pd
from numpy.random import uniform
from scipy import ndimage, integrate
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.utils import check_array

from skbel.algorithms.extmath import get_block

__all__ = [
    "KDE",
    "kde_params",
    "posterior_conditional",
    "mvn_inference",
    "it_sampling",
    "normalize",
    "remove_outliers",
]


def tupleset(t, i, value):
    """Set the `i`th element of a tuple to `value`.

    :param t: tuple
    :param i: index
    :param value: value
    """
    l = list(t)
    l[i] = value
    return tuple(l)


def romb(y: np.array, dx: float = 1.0) -> np.array:
    """Romberg's integration using samples of a function.

    :param y: A vector of ``2**k + 1`` equally-spaced samples of a function.
    :param dx: The sample spacing. Default is 1.
    :return: The integral of the function.
    """
    y = np.asarray(y)
    nd = len(y.shape)
    axis = -1
    Nsamps = y.shape[axis]
    Ninterv = Nsamps - 1
    n = 1
    k = 0
    while n < Ninterv:
        n <<= 1
        k += 1
    if n != Ninterv:
        raise ValueError(
            "Number of samples must be one plus a " "non-negative power of 2."
        )

    R = {}
    slice_all = (slice(None),) * nd
    slice0 = tupleset(slice_all, axis, 0)
    slicem1 = tupleset(slice_all, axis, -1)
    h = Ninterv * np.asarray(dx, dtype=float)
    R[(0, 0)] = (y[slice0] + y[slicem1]) / 2.0 * h
    slice_R = slice_all
    start = stop = step = Ninterv
    for i in range(1, k + 1):
        start >>= 1
        slice_R = tupleset(slice_R, axis, slice(start, stop, step))
        step >>= 1
        R[(i, 0)] = 0.5 * (R[(i - 1, 0)] + h * y[slice_R].sum(axis=axis))
        for j in range(1, i + 1):
            prev = R[(i, j - 1)]
            R[(i, j)] = prev + (prev - R[(i - 1, j - 1)]) / ((1 << (2 * j)) - 1)
        h /= 2.0
    return R[(k, k)]


class KDE:
    """Uni/Bi-variate kernel density estimator.

    This class is adapted from the class of the same name in the package Seaborn 0.11.1
    https://seaborn.pydata.org/generated/seaborn.kdeplot.html
    """

    def __init__(
            self,
            *,
            kernel_type: str = None,
            bandwidth: float = None,
            grid_search: bool = True,
            bandwidth_space: np.array = None,
            gridsize: int = 200,
            cut: float = 1,
            clip: list = None,
    ):
        """Initialize the estimator with its parameters.

        :param kernel_type: kernel type, one of 'gaussian', 'tophat', 'epanechnikov',
            'exponential', 'linear', 'cosine'
        :param bandwidth: bandwidth
        :param grid_search: perform a grid search for the bandwidth
        :param bandwidth_space: array of bandwidths to try
        :param gridsize: number of points on each dimension of the evaluation grid.
        :param cut: Factor, multiplied by the smoothing bandwidth, that determines how
            far the evaluation grid extends past the extreme datapoints. When
            set to 0, truncate the curve at the data limits.
        :param clip: A list of two elements, the lower and upper bounds for the
            support of the density. If None, the support is the range of the data.
        """
        if clip is None:
            clip = None, None

        self.kernel_type = kernel_type
        if kernel_type is None:
            self.kernel_type = "gaussian"  # default
        self.bw = bandwidth
        self.grid_search = grid_search
        if bandwidth_space is None:
            self.bandwidth_space = np.logspace(-2, 2, 50)  # default bandwidths
        else:
            self.bandwidth_space = bandwidth_space
        self.gridsize = gridsize
        self.cut = cut
        self.clip = clip

        self.support = None

    @staticmethod
    def _define_support_grid(
            x: np.array, bandwidth: float, cut: float, clip: list, gridsize: int
    ):
        """Create the grid of evaluation points depending for vector x.

        :param x: vector of values
        :param bandwidth: bandwidth
        :param cut: factor, multiplied by the smoothing bandwidth, that determines how
            far the evaluation grid extends past the extreme datapoints. When
            set to 0, truncate the curve at the data limits.
        :param clip: pair of numbers None, or a pair of such pairs
            Do not evaluate the density outside of these limits.
        :param gridsize: number of points on each dimension of the evaluation grid.
        :return: evaluation grid
        """
        clip_lo = -np.inf if clip[0] is None else clip[0]
        clip_hi = +np.inf if clip[1] is None else clip[1]
        bw = 1 if bandwidth is None else bandwidth
        gridmin = max(x.min() - bw * cut, clip_lo)
        gridmax = min(x.max() + bw * cut, clip_hi)
        return np.linspace(gridmin, gridmax, gridsize)

    def _define_support_univariate(self, x: np.array):
        """Create a 1D grid of evaluation points.

        :param x: 1D array of data
        :return: 1D array of evaluation points
        """
        grid = self._define_support_grid(
            x, self.bw, self.cut, self.clip, self.gridsize
        )  # define grid
        return grid

    def _define_support_bivariate(self, x1: np.array, x2: np.array):
        """Create a 2D grid of evaluation points.

        :param x1: 1st dimension of the evaluation grid
        :param x2: 2nd dimension of the evaluation grid
        :return: 2D grid of evaluation points
        """
        clip = self.clip
        if clip[0] is None or np.isscalar(clip[0]):  # if clip is a single number
            clip = (clip, clip)
        grid1 = self._define_support_grid(
            x1, self.bw, self.cut, clip[0], self.gridsize
        )  # define grid for x1
        grid2 = self._define_support_grid(
            x2, self.bw, self.cut, clip[1], self.gridsize
        )  # define grid for x2

        return grid1, grid2

    def define_support(
            self,
            x1: np.array,
            x2: np.array = None,
            cache: bool = True,
    ):
        """Create the evaluation grid for a given data set.

        :param x1: 1D array of data
        :param x2: 2D array of data
        :param cache: if True, cache the support grid
        :return: grid of evaluation points
        """
        if x2 is None:
            support = self._define_support_univariate(x1)  # 1D
        else:
            support = self._define_support_bivariate(x1, x2)  # 2D

        if cache:
            self.support = support  # cache the support grid

        return support

    def _fit(self, fit_data: np.array):
        """Fit the scikit-learn KDE.

        :param fit_data: Data to fit the KDE to.
        :return: fitted KDE object
        """
        bw = 1 if self.bw is None else self.bw  # bandwidth
        fit_kws = {
            "bandwidth": bw,
            "algorithm": "auto",  # kdtree or ball_tree
            "kernel": self.kernel_type,
            "metric": "euclidean",  # default
            "atol": 1e-4,  # tolerance for convergence
            "rtol": 0,  #
            "breadth_first": True,  #
            "leaf_size": 40,
            "metric_params": None,
        }  # define the kernel density estimator parameters
        kde = KernelDensity(**fit_kws)  # initiate the estimator
        if self.grid_search and not self.bw:
            # GridSearchCV maximizes the total log probability density under the model.
            # The data X will be divided into train-test splits based on folds defined in cv param
            # For each combination of parameters that you specified in param_grid, the model
            # will be trained on the train part from the step above and then scoring will be used on test part.
            # The scores for each parameter combination will be combined for all the folds and averaged.
            # Highest performing parameter will be selected.

            grid = GridSearchCV(
                kde, {"bandwidth": self.bandwidth_space}
            )  # Grid search on bandwidth
            grid.fit(fit_data)  # Fit the grid search
            self.bw = grid.best_params_[
                "bandwidth"
            ]  # Set the bandwidth to the best bandwidth
            fit_kws["bandwidth"] = self.bw  # Update the bandwidth in the fit_kws
            kde.set_params(
                **{"bandwidth": self.bw}
            )  # Update the bandwidth in the scikit-learn model

        kde.fit(fit_data)  # Fit the KDE

        return kde

    def _eval_univariate(self, x: np.array):
        """Fit and evaluate on univariate data.

        :param x: Data to evaluate.
        :return: (density, support)
        """
        support = self.support
        if support is None:
            support = self.define_support(x, cache=True)

        kde = self._fit(x.reshape(-1, 1))

        density = np.exp(kde.score_samples(support.reshape(-1, 1)))  # evaluate the KDE

        return density, support

    def _eval_bivariate(self, x1: np.array, x2: np.array):
        """Fit and evaluate on bivariate data.

        :param x1: First data set.
        :param x2: Second data set.
        :return: (density, support)
        """
        support = self.support
        if support is None:
            support = self.define_support(x1, x2, cache=False)

        X_train = np.vstack([x1, x2]).T

        kde = self._fit(X_train)

        X, Y = np.meshgrid(*support)
        grid = np.vstack([X.ravel(), Y.ravel()]).T

        density = np.exp(kde.score_samples(grid))  # evaluate the KDE
        density = density.reshape(X.shape)

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
    """Estimate the density of a single variable.

    :param data_variable: DataFrame with a single variable.
    :param estimate_kws: Keyword arguments for the density estimator.
    :return: (density, support, bandwidth)
    """
    # Initialize the estimator object
    estimator = KDE(**estimate_kws)

    sub_data = data_variable.dropna()

    # # Extract the data points from this sub set and remove nulls
    observations = sub_data["x"].to_numpy()

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
    """Estimate bivariate KDE.

    :param data: DataFrame containing (x, y) data
    :param estimate_kws: KDE parameters
    :return: (density, support, bandwidth)
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

    return density, support, estimator.bw


def kde_params(
        x: np.array = None,
        y: np.array = None,
        bw: float = None,
        bandwidth_space=None,
        gridsize: int = 200,
        cut: float = 1,
        clip=None,
):
    """Computes the kernel density estimate (KDE) of one or two data sets.

    :param x: The x-coordinates of the input data.
    :param y: The y-coordinates of the input data.
    :param gridsize: Number of discrete points in the evaluation grid.
    :param bw: The bandwidth of the kernel.
    :param bandwidth_space: The space to search for the bandwidth.
    :param cut: Draw the estimate to cut * bw from the extreme data points.
    :param clip: Lower and upper bounds for datapoints used to fit KDE. Can provide
        a pair of (low, high) bounds for bivariate plots.
    :return: (density: The estimated probability density function evaluated at the support,
             support: The support of the density function, the x-axis of the KDE.)
    """

    # Pack the kwargs for KDE
    estimate_kws = dict(
        bandwidth=bw,
        bandwidth_space=bandwidth_space,
        gridsize=gridsize,
        cut=cut,
        clip=clip,
    )

    if y is None:
        data = {"x": x}
        frame = pd.DataFrame(data=data)
        density, support, bw = _univariate_density(
            data_variable=frame, estimate_kws=estimate_kws
        )

    else:
        data = {"x": x, "y": y}
        frame = pd.DataFrame(data=data)
        density, support, bw = _bivariate_density(
            data=frame,
            estimate_kws=estimate_kws,
        )

    return density, support, bw


def _pixel_coordinate(line: list, x_1d: np.array, y_1d: np.array, k: int = None):
    """Gets the pixel coordinate of the value x or y, in order to get posterior
    conditional probability given a KDE.

    :param line: Coordinates of the line we'd like to sample along [(x1, y1), (x2, y2)]
    :param x_1d: List of x coordinates along the axis
    :param y_1d: List of y coordinates along the axis
    :param k: Used to set number of rows/columns
    :return: (rows, columns)
    """
    if k is None:
        num = 200
    else:
        num = k

    # https://stackoverflow.com/questions/18920614/plot-cross-section-through-heat-map
    # Convert the line to pixel/index coordinates
    x_world, y_world = np.array(list(zip(*line)))
    col = y_1d.shape * (x_world - min(x_1d)) / x_1d.ptp()
    row = x_1d.shape * (y_world - min(y_1d)) / y_1d.ptp()

    # Interpolate the line at "num" points...
    row, col = [np.linspace(item[0], item[1], num) for item in [row, col]]

    return row, col


def _conditional_distribution(
        kde_array: np.array,
        x_array: np.array,
        y_array: np.array,
        x: float = None,
        y: float = None,
        k: int = None,
):
    """Compute the conditional posterior distribution p(x_array|y_array) given
    x or y. Provide only one observation ! Either x or y. Perform a cross-
    section in the KDE along the y axis.

    :param kde_array: KDE of the prediction
    :param x_array: X grid (1D)
    :param y_array: Y grid (1D)
    :param x: Observed data (horizontal axis)
    :param y: Observed data (vertical axis)
    :param k: Used to set number of rows/columns
    :return: (cross_section: The cross-section of the KDE, line: The line of the KDE)
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
    row, col = _pixel_coordinate(line=line, x_1d=x_array, y_1d=y_array, k=k)

    # Extract the values along the line, using cubic interpolation
    zi = ndimage.map_coordinates(kde_array, np.vstack((row, col)))

    if x is not None:
        line = np.linspace(min(y_array), max(y_array), k)
    elif y is not None:
        line = np.linspace(min(x_array), max(x_array), k)

    return zi, line


def _scale_distribution(post: np.array, support: np.array) -> np.array:
    """Scale the distribution to have a maximum of 1, and a minimum of 0.

    :param post: Values of the KDE cross-section
    :param support: Support of the KDE cross-section
    :return: The scaled distribution
    """

    post[np.abs(post) < 1e-8] = 0  # Rule of thumb

    if post.any():  # If there is any value
        a = integrate.simps(y=np.abs(post), x=support)  # Integrate the absolute values
        post *= 1 / a  # Scale the distribution

    return post


def posterior_conditional(
        X_obs: float = None,
        Y_obs: float = None,
        dens: np.array = None,
        support: np.array = None,
        k: int = None,
) -> (np.array, np.array):
    """Computes the posterior distribution p(y|x_obs) or p(x|y_obs) by doing a
    cross-section of the KDE of (d, h).

    :param X_obs: Observation (predictor, x-axis)
    :param Y_obs: Observation (target, y-axis)
    :param dens: The density values of the KDE of (X, Y).
    :param support: The support grid of the KDE of (X, Y).
    :param k: Used to set number of rows/columns
    :return: The posterior distribution p(y|x_obs) or p(x|y_obs) and the support grid of the cross-section.
    """
    # Grid parameters
    xg, yg = support

    if X_obs is not None:
        # Extract the density values along the line, using cubic interpolation
        if isinstance(X_obs, list) or isinstance(X_obs, np.ndarray):
            X_obs = X_obs[0]
        post, line = _conditional_distribution(
            x=X_obs, x_array=xg, y_array=yg, kde_array=dens, k=k
        )
    elif Y_obs is not None:
        # Extract the density values along the line, using cubic interpolation
        if isinstance(Y_obs, list) or isinstance(Y_obs, np.ndarray):
            Y_obs = Y_obs[0]
        post, line = _conditional_distribution(
            y=Y_obs, x_array=xg, y_array=yg, kde_array=dens, k=k
        )
    else:
        msg = "No observation point included."
        warnings.warn(msg, UserWarning)
        return 0

    post = _scale_distribution(post, line)

    return post, line


def mvn_inference(
        X: np.array, Y: np.array, X_obs: np.array, **kwargs
) -> (np.array, np.array):
    """Estimates the posterior mean and covariance of the target.
       Note that in this implementation, n_samples must be = 1.

    .. [1] A. Tarantola. Inverse Problem Theory and Methods for Model Parameter Estimation.
           SIAM, 2005. Pages: 70-71

    :param X: Canonical Variate of the training data
    :param Y: Canonical Variate of the training target, gaussian-distributed
    :param X_obs: Canonical Variate of the observation (n_samples, n_features).
    :return: y_posterior_mean, y_posterior_covariance
    """

    Y = check_array(Y, copy=True, ensure_2d=False)
    X = check_array(X, copy=True, ensure_2d=False)
    X_obs = check_array(X_obs, copy=True, ensure_2d=False)

    # Size of the set
    n_training = X.shape[0]

    # Computation of the posterior mean in Canonical space
    y_mean = np.mean(Y, axis=0)  # (n_comp_CCA, 1)  # noqa
    # Mean is 0, as expected.
    y_mean = np.where(np.abs(y_mean) < 1e-8, 0, y_mean)

    # Evaluate the covariance in h (in Canonical space)
    # Very close to the Identity matrix
    # (n_comp_CCA, n_comp_CCA)
    y_cov = np.cov(Y.T)  # noqa

    if "x_cov" in kwargs.keys():
        x_cov = kwargs["x_cov"]
    else:
        x_cov = np.zeros(shape=y_cov.shape)

    # Linear modeling h to d (in canonical space) with least-square criterion.
    # Pay attention to the transpose operator.
    # Computes the vector g that approximately solves the equation y @ g = x.
    g = np.linalg.lstsq(Y, X, rcond=None)[0].T
    # Replace values below threshold by 0.
    g = np.where(np.abs(g) < 1e-8, 0, g)  # (n_comp_CCA, n_comp_CCA)

    # Modeling error due to deviations from theory
    # (n_components_CCA, n_training)
    x_ls_predicted = np.matmul(Y, g.T)  # noqa
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
            d11 @ y_mean - d12 @ (X_obs[0] - x_modeling_mean_error - y_mean @ g.T)  # noqa
    )  # (n_comp_CCA,)

    return y_posterior_mean, y_posterior_covariance


def normalize(pdf):
    """Normalize a non-normalized PDF.

    :param pdf: The probability density function (not necessarily normalized). Must take
        floats or ints as input, and return floats as an output.
    :return: pdf_norm: Function with same signature as pdf, but normalized so that the integral
        between lower_bd and upper_bd is close to 1. Maps nicely over iterables.
    """

    dx = np.abs(pdf.x[1] - pdf.x[0])  # Assume uniform spacing
    quadrature = romb(pdf.y, dx)  # Integrate using Romberg's method
    A = quadrature  # Normalization constant

    def pdf_normed(x):
        """Normalized PDF.

        :param x: Input to the pdf.
        :return: pdf(x) / A.
        """
        b = np.interp(x=x, xp=pdf.x, fp=pdf.y)  # Evaluate the PDF at x
        if A < 1e-3:  # Rule of thumb
            return 0
        if b / A < 1e-3:  # If the PDF is very small, return 0
            return 0
        else:
            return b / A

    return pdf_normed


def get_cdf(pdf):
    """Generate a CDF from a (possibly not normalized) pdf.

    :param pdf: The probability density function (not necessarily normalized). Must take
        floats or ints as input, and return floats as an output.
    :return: cdf: The cumulative density function of the (normalized version of the)
        provided pdf. Will return a float if provided with a float or int; will
        return a numpy array if provided with an iterable.
    """
    pdf_norm = normalize(pdf)  # Calculate the normalized pdf
    lower_bound = np.min(pdf.x)
    upper_bound = np.max(pdf.x)

    def cdf_number(x):
        """Numerical cdf.

        :param x: The value to evaluate the cdf at.
        :return: The value of the cdf at x.
        """
        if x <= lower_bound:
            return 0
        elif x >= upper_bound:
            return 1
        else:
            d = np.abs(x - lower_bound)
            if d > 1e-4:  # Check that spacing isn't too small
                samples = np.linspace(lower_bound, x, 2 ** 7 + 1)
                dx = np.abs(samples[1] - samples[0])
                y = np.array([pdf_norm(s) for s in samples])
                return romb(y, dx)
            else:
                return 0

    def cdf_vector(x):
        """Vectorized cdf.

        :param x: The values to evaluate the cdf at.
        :return: The values of the cdf at x.
        """
        try:
            return np.array([cdf_number(xi) for xi in x])
        except AttributeError:
            return cdf_number(x)

    return cdf_vector


def it_sampling(
        pdf,
        num_samples: int = 1,
        lower_bd=-np.inf,
        upper_bd=np.inf,
        k: int = None,
        cdf_y: np.array = None,
        return_cdf: bool = False,
):
    """Sample from an arbitrary, un-normalized PDF.

    :param pdf: function, float -> float The probability density function (not necessarily normalized). Must take floats
     or ints as input, and return floats as an output.
    :param num_samples: The number of samples to be generated.
    :param lower_bd: Lower bound of the support of the pdf. This parameter allows one to manually establish cutoffs for
     the density.
    :param upper_bd: Upper bound of the support of the pdf.
    :param k: Step number between lower_bd and upper_bd
    :param cdf_y: precomputed values of the CDF
    :param return_cdf: Option to return the computed CDF values
    :return: samples: An array of samples from the provided PDF, with support between lower_bd and upper_bd.
    """
    if k is None:
        k = 200  # Default step size

    if cdf_y is None:
        cdf = get_cdf(pdf)  # CDF of the pdf
        cdf_y = cdf(np.linspace(lower_bd, upper_bd, k))  # CDF values

    if return_cdf:
        return cdf_y

    else:
        if cdf_y.any():
            seeds = uniform(0, 1, num_samples)  # Uniformly distributed seeds
            simple_samples = np.interp(x=seeds, xp=cdf_y, fp=pdf.x)  # Samples
        else:
            simple_samples = np.zeros(num_samples)  # Samples

        return simple_samples


def remove_outliers(data):
    """Removes outliers from the data.

    :param data: array-like
    :return: data without outliers
    """
    q25 = np.quantile(data, 0.25, axis=0)
    q75 = np.quantile(data, 0.75, axis=0)
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    # if one element is out of bounds, delete its row
    data = data[(data > lower_bound).all(axis=1) & (data < upper_bound).all(axis=1)]
    return data
