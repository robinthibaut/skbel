"""pytest unit tests for skbel."""

#  Copyright (c) 2022. Robin Thibaut, Ghent University

import os
from os.path import join as jp

import numpy as np
import scipy
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer

from skbel import BEL
from skbel.tmaps import TransportMap

# Get file path
my_path = os.path.dirname(os.path.abspath(__file__))


def init_bel():
    """Set all BEL pipelines.
    This is the blueprint of the framework.
    """
    # Pipeline before CCA
    X_pre_processing = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("pca", PCA(n_components=50)),
        ]
    )
    Y_pre_processing = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("pca", PCA(n_components=30)),
        ]
    )

    # Canonical Correlation Analysis
    cca = CCA(n_components=30)

    # Pipeline after CCA
    X_post_processing = Pipeline(
        [("normalizer", PowerTransformer(method="yeo-johnson", standardize=True))]
    )
    Y_post_processing = Pipeline(
        [("normalizer", PowerTransformer(method="yeo-johnson", standardize=True))]
    )

    # Initiate BEL object
    bel_model = BEL(
        X_pre_processing=X_pre_processing,
        X_post_processing=X_post_processing,
        Y_pre_processing=Y_pre_processing,
        Y_post_processing=Y_post_processing,
        regression_model=cca,
    )

    return bel_model


def test_mvn():
    """Compare posterior mean and covariance with reference default values."""

    # Initiate BEL object
    bel = init_bel()
    # Set seed
    seed = 123456
    bel.seed = seed
    bel.mode = "mvn"
    bel.n_posts = 10

    X_train = np.load(jp(my_path, "X_train.npy"))
    y_train = np.load(jp(my_path, "y_train.npy"))

    X_test = np.load(jp(my_path, "X_test.npy"))

    # Fit
    bel.fit(X=X_train, Y=y_train)
    # Predict posterior mean and covariance
    bel.predict(X_obs=X_test)

    # Compare with reference
    ref_mean = np.load(jp(my_path, "ref_mean.npy"))
    ref_covariance = np.load(jp(my_path, "ref_covariance.npy"))

    msg1 = "The posterior means are different"
    np.testing.assert_array_almost_equal(bel.posterior_mean[0], ref_mean, err_msg=msg1)

    msg2 = "The posterior covariances are different"
    np.testing.assert_allclose(
        bel.posterior_covariance[0], ref_covariance, atol=1e-3, err_msg=msg2
    )


def test_kde():
    """Compare posterior samples with reference default values."""
    # %% Initiate BEL model
    # Initiate BEL object
    model = init_bel()
    # Set seed
    seed = 123456
    model.seed = seed

    X_train = np.load(jp(my_path, "X_train.npy"))
    y_train = np.load(jp(my_path, "y_train.npy"))

    X_test = np.load(jp(my_path, "X_test.npy"))
    # %% Set model parameters
    model.mode = "kde"  # How to compute the posterior conditional distribution
    # Number of samples to be extracted from the posterior distribution
    model.n_posts = 10

    # %% Train the model
    # Fit BEL model
    model.fit(X=X_train, Y=y_train)

    # Sample for the observation
    # Extract n random sample (target CV's).
    # The posterior distribution is computed within the method below.
    y_samples = model.predict(X_test)
    # np.save(jp(my_path, "y_samples_kde.npy"), y_samples)
    y_samples_ref = np.load(jp(my_path, "y_samples_kde.npy"))

    msg1 = "The posterior samples are different"
    np.testing.assert_allclose(y_samples, y_samples_ref, atol=2, err_msg=msg1)


def test_tm():
    """Compare posterior samples with reference default values."""
    N = 1000
    X = scipy.stats.norm.rvs(size=(N, 2), random_state=123456)
    b = 1  # Twist factor
    X[:, 1] += b * X[:, 0] ** 2

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

    tm = TransportMap(
        monotone=monotone,
        nonmonotone=nonmonotone,
        X=X,
        polynomial_type="probabilist's hermite",
        monotonicity="separable monotonicity",
        standardize_samples=True,
        workers=1,
    )

    tm.optimize()

    norm_samples = scipy.stats.norm.rvs(size=(500, 1), random_state=42)  # noqa

    x1_obs = 2.32

    X_precalc = np.ones((500, 1)) * x1_obs

    # Now invert the map conditionally. X_star are the posterior samples.
    X_star = tm.inverse_map(X_precalc=X_precalc, Y=norm_samples)

    X_star_ref = np.load(jp(my_path, "x_star_tm.npy"))

    msg1 = "The inverted samples are different"
    np.testing.assert_allclose(X_star, X_star_ref, atol=1e-3, err_msg=msg1)
