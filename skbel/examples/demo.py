#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os
from os.path import join as jp

import joblib
from loguru import logger

import numpy as np
import pandas as pd

from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer

from skbel.learning.bel import BEL

from skbel import utils


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
    n_pc_pred, n_pc_targ = 50, 30

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


def bel_training(bel, *, X_train_, x_test_, y_train_, y_test_, directory):
    """
    :param bel:
    :param X_train_:
    :param x_test_:
    :param y_train_:
    :param y_test_:
    :param directory:
    :return:
    """
    # Directories
    # Directory in which to load forecasts
    sub_dir = jp(directory, "results")

    # %% Folders
    obj_dir = jp(sub_dir, "obj")
    fig_data_dir = jp(sub_dir, "data")
    fig_pca_dir = jp(sub_dir, "pca")
    fig_cca_dir = jp(sub_dir, "cca")
    fig_pred_dir = jp(sub_dir, "uq")

    # %% Creates directories
    [
        utils.dirmaker(f, erase=True)
        for f in [
            obj_dir,
            fig_data_dir,
            fig_pca_dir,
            fig_cca_dir,
            fig_pred_dir,
        ]
    ]

    # %% Fit
    bel.Y_obs = y_test_
    bel.fit(X=X_train_, Y=y_train_)

    # %% Sample
    # Extract n random sample (target pc's).
    # The posterior distribution is computed within the method below.
    bel.predict(x_test_)

    # Save the fitted BEL model
    joblib.dump(bel, jp(obj_dir, "bel.pkl"))
    msg = f"model trained and saved in {obj_dir}"
    logger.info(msg)


if __name__ == "__main__":
    # Initiate BEL model
    model = init_bel()

    # Set directories
    data_dir = jp(os.getcwd(), "dataset")
    output_dir = jp(os.getcwd(), "results")

    # Load dataset
    X_train = pd.read_pickle(jp(data_dir, "X_train.pkl"))
    X_test = pd.read_pickle(jp(data_dir, "X_test.pkl"))
    y_train = pd.read_pickle(jp(data_dir, "y_train.pkl"))
    y_test = pd.read_pickle(jp(data_dir, "y_test.pkl"))

    # Train model
    bel_training(
        bel=model,
        X_train_=X_train,
        x_test_=X_test,
        y_train_=y_train,
        y_test_=y_test,
        directory=output_dir,
    )

    # Plot the results
    bel = joblib.load(jp(output_dir, "obj", "bel.pkl"))

    myvis.plot_results(
        bel,
        base_dir=base_dir,
        root=sample,
        folder=w,
        annotation=annotation,
        d=False,
    )

    myvis.pca_vision(
        bel,
        base_dir=base_dir,
        w=w,
        root=sample,
        d=True,
        h=True,
        exvar=True,
        before_after=True,
        labels=True,
        scores=True,
    )

    myvis.cca_vision(base_dir=base_dir, root=sample, folders=wells)
