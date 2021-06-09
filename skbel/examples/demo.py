#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os
from os.path import join as jp

import joblib
import pandas as pd
from loguru import logger
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer

import demo_visualization as myvis
from skbel.goggles import pca_vision, cca_vision

from skbel import utils
from skbel.learning.bel import BEL


def init_bel():
    """
    Set all BEL pipelines. This is the blueprint of the framework.
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
    cca = CCA()

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


if __name__ == "__main__":

    # %% Set directories
    data_dir = jp(os.getcwd(), "dataset")
    # Directory in which to unload forecasts
    sub_dir = jp(os.getcwd(), "results")

    # Folders
    obj_dir = jp(sub_dir, "obj")  # Location to save the BEL model
    fig_data_dir = jp(sub_dir, "data")  # Location to save the raw data figures
    fig_pca_dir = jp(sub_dir, "pca")  # Location to save the PCA figures
    fig_cca_dir = jp(sub_dir, "cca")  # Location to save the CCA figures
    fig_pred_dir = jp(sub_dir, "uq")  # Location to save the prediction figures

    # Creates directories
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

    # %% Load dataset
    X_train = pd.read_pickle(jp(data_dir, "X_train.pkl"))
    X_test = pd.read_pickle(jp(data_dir, "X_test.pkl"))
    y_train = pd.read_pickle(jp(data_dir, "y_train.pkl"))
    y_test = pd.read_pickle(jp(data_dir, "y_test.pkl"))

    # %% Initiate BEL model
    model = init_bel()

    # %% Set model parameters
    model.mode = "mvn"  # How to compute the posterior conditional distribution
    # Set PC cut
    model.X_n_pc = 50
    model.Y_n_pc = 30
    # Save original dimensions of both predictor and target
    model.X_shape = (6, 200)  # Six curves with 200 time steps each
    model.Y_shape = (1, 100, 87)  # One matrix with 100 rows and 87 columns
    # Number of CCA components is chosen as the min number of PC
    n_cca = min(model.X_n_pc, model.Y_n_pc)
    model.cca.n_components = n_cca
    # Number of samples to be extracted from the posterior distribution
    model.n_posts = 400

    # %% Train the model
    # Fit BEL model
    model.Y_obs = y_test
    model.fit(X=X_train, Y=y_train)

    # Sample for the observation
    # Extract n random sample (target CV's).
    # The posterior distribution is computed within the method below.
    model.predict(X_test)

    # Save the fitted BEL model
    joblib.dump(model, jp(obj_dir, "bel.pkl"))
    msg = f"model trained and saved in {obj_dir}"
    logger.info(msg)

    # %% Visualization

    # Plot raw data
    myvis.plot_results(model, base_dir=sub_dir)

    # Plot PCA
    pca_vision(
        model,
        fig_dir=fig_pca_dir,
    )

    # Plot CCA
    cca_vision(bel=model, fig_dir=fig_cca_dir)
