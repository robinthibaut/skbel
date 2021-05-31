"""pytest unit"""

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
    """
    Set all BEL pipelines
    """
    # Pipeline before CCA
    X_pre_processing = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("pca", PCA()),
        ]
    )
    Y_pre_processing = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("pca", PCA()),
        ]
    )

    # Canonical Correlation Analysis
    # Number of CCA components is chosen as the min number of PC
    n_pc_pred, n_pc_targ = (
        50,
        30,
    )
    cca = CCA(n_components=min(n_pc_targ, n_pc_pred), max_iter=500 * 20, tol=1e-6)

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

    # Set PC cut
    bel_model.X_n_pc = n_pc_pred
    bel_model.Y_n_pc = n_pc_targ

    return bel_model


def test_posterior():
    """Compare posterior mean and covariance with reference default values"""

    # Initiate BEL object
    bel = init_bel()
    # Set seed
    seed = 123456
    bel.seed = seed

    # Get file path
    my_path = os.path.dirname(os.path.abspath(__file__))

    X_train = np.load(jp(my_path, "X_train.npy"))
    y_train = np.load(jp(my_path, "y_train.npy"))

    X_test = np.load(jp(my_path, "X_test.npy"))

    # Fit
    bel.fit(X=X_train, Y=y_train)
    # Predict posterior mean and covariance
    post_mean, post_cov = bel.predict(X_test)

    # Compare with reference
    ref_mean = np.load(jp(my_path, "ref_mean.npy"))
    ref_covariance = np.load(jp(my_path, "ref_covariance.npy"))

    msg1 = "The posterior means are different"
    np.testing.assert_allclose(post_mean, ref_mean, atol=1e-6, err_msg=msg1)

    msg2 = "The posterior covariances are different"
    np.testing.assert_allclose(post_cov, ref_covariance, atol=1e-6, err_msg=msg2)
