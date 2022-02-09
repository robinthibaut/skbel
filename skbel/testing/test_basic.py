"""pytest unit tests for skbel.basic."""

#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os
from os.path import join as jp

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer

from skbel.learning.bel import BEL


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
        cca=cca,
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

    # Get file path
    my_path = os.path.dirname(os.path.abspath(__file__))

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
    # np.testing.assert_allclose(bel.posterior_mean[0], ref_mean, atol=1e-3, err_msg=msg1)

    msg2 = "The posterior covariances are different"
    np.testing.assert_allclose(
        bel.posterior_covariance[0], ref_covariance, atol=1e-3, err_msg=msg2
    )


def test_kde():
    # %% Initiate BEL model
    # Initiate BEL object
    model = init_bel()
    # Set seed
    seed = 123456
    model.seed = seed

    # Get file path
    my_path = os.path.dirname(os.path.abspath(__file__))

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
