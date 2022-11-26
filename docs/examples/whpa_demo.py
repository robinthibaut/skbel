#  Copyright (c) 2022. Robin Thibaut, Ghent University

import os
from os.path import join as jp

import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer

import demo_visualization as myvis

from skbel import utils
from skbel import BEL


def init_bel():
    """Set all BEL pipelines.

    This is the blueprint of the framework.
    """
    # Pipeline before CCA

    # Data preprocessing
    X_pre_processing = Pipeline(
        [
            ("pca", PCA(n_components=50)),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    # Target preprocessing
    Y_pre_processing = Pipeline(
        [
            ("pca", PCA(n_components=30)),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    # Canonical Correlation Analysis
    cca = CCA(n_components=30)

    # Pipeline after CCA

    # Data postprocessing
    X_post_processing = Pipeline(
        [("normalizer", PowerTransformer(method="yeo-johnson", standardize=True))]
    )

    # Target postprocessing
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


if __name__ == "__main__":

    # %% Set directories
    data_dir = jp(os.getcwd(), "dataset")
    # Directory in which to unload forecasts
    sub_dir = jp(os.getcwd(), "results")

    # Folders
    obj_dir = jp(sub_dir, "obj")  # Location to save the BEL model
    fig_data_dir = jp(sub_dir, "data")  # Location to save the raw data figures
    fig_pca_dir = jp(sub_dir, "pca")  # Location to save the PCA figures
    fig_cca_dir = jp(sub_dir, "regression_model")  # Location to save the CCA figures
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
    model.mode = "tm"  # How to compute the posterior conditional distribution
    # Save original dimensions of both predictor and target
    model.X_shape = (6, 200)  # Six curves with 200 time steps each
    model.Y_shape = (100, 87)  # 100 rows and 87 columns
    # Number of samples to be extracted from the posterior distribution
    model.n_posts = 400

    # %% Train the model
    # Fit BEL model
    model.fit(X=X_train, Y=y_train)

    # Sample for the observation
    # Extract n random sample (target CV's).
    # The posterior distribution is computed within the method below.
    y_predicted = model.predict(X_test.array.reshape(1, -1))

    # %% Visualization

    # Plot raw data
    myvis.plot_results(
        model,
        y_predicted=y_predicted,
        X=X_train,
        X_obs=X_test,
        Y=y_train,
        Y_obs=y_test,
        base_dir=sub_dir,
    )
