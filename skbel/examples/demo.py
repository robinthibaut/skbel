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
from demo_config import Setup
from skbel import utils
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
        mode="kde",
    )

    # Set PC cut
    bel_model.X_n_pc = n_pc_pred
    bel_model.Y_n_pc = n_pc_targ

    return bel_model


def bel_training(
    bel_,
    *,
    X_train_: pd.DataFrame,
    x_test_: pd.DataFrame,
    y_train_: pd.DataFrame,
    y_test_: pd.DataFrame = None,
    directory: str = None,
):
    """
    :param bel_: BEL model
    :param X_train_: Predictor set for training
    :param x_test_: Predictor "test"
    :param y_train_: Target set for training
    :param y_test_: "True" target (optional)
    :param directory: Path to the directory in which to unload the results
    :return:
    """
    #%% Directory in which to load forecasts
    if directory is None:
        sub_dir = os.getcwd()
    else:
        sub_dir = directory

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

    # %% Fit BEL model
    bel_.Y_obs = y_test_
    bel_.fit(X=X_train_, Y=y_train_)

    # %% Sample for the observation
    # Extract n random sample (target pc's).
    # The posterior distribution is computed within the method below.
    bel_.predict(x_test_)

    # Save the fitted BEL model
    joblib.dump(bel_, jp(obj_dir, "bel.pkl"))
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
        bel_=model,
        X_train_=X_train,
        x_test_=X_test,
        y_train_=y_train,
        y_test_=y_test,
        directory=output_dir,
    )

    # Plot the results
    bel = joblib.load(jp(output_dir, "obj", "bel.pkl"))
    bel.n_posts = Setup.HyperParameters.n_posts

    # Plot raw data
    myvis.plot_results(bel, base_dir=output_dir)

    # Plot PCA
    # myvis.pca_vision(
    #     bel,
    #     base_dir=output_dir,
    # )

    # Plot CCA
    myvis.cca_vision(bel=bel, base_dir=output_dir)
