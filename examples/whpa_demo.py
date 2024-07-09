#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os
from os.path import join as jp
import joblib

from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer

from demo_visualization import plot_results

from skbel import utils
from skbel import BEL


def init_bel():
    """Set all BEL pipelines.

    This is the blueprint of the framework.
    """
    # Pipeline before CCA
    X_pre_processing = Pipeline(
        [
            ("pca", PCA(n_components=50)),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    Y_pre_processing = Pipeline(
        [
            ("pca", PCA(n_components=30)),
            ("scaler", StandardScaler(with_mean=False)),
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


if __name__ == "__main__":

    # %% Set directories
    data_dir = jp(os.getcwd(), "dataset")
    # Directory in which to unload forecasts
    sub_dir = jp(os.getcwd(), "results")

    # Folders
    fig_data_dir = jp(sub_dir, "data")  # Location to save the raw data figures
    fig_pred_dir = jp(sub_dir, "uq")  # Location to save the prediction figures

    # Creates directories
    [
        utils.dirmaker(f, erase=True)
        for f in [
            fig_data_dir,
            fig_pred_dir,
        ]
    ]

    # %% Load dataset
    X_train = joblib.load(jp(data_dir, "X_train.pkl")).to_numpy()
    X_test = joblib.load(jp(data_dir, "X_test.pkl")).to_numpy().reshape(1, -1)
    y_train = joblib.load(jp(data_dir, "y_train.pkl")).to_numpy()
    y_test = joblib.load(jp(data_dir, "y_test.pkl")).to_numpy().reshape(1, -1)

    # %% Initiate BEL model
    model = init_bel()

    # %% Set model parameters
    model.mode = "tm"  # How to compute the posterior conditional distribution
    # Save original dimensions of both predictor and target
    model.X_shape = (6, 200)  # Six curves with 200 time steps each
    model.Y_shape = (100, 87)  # 100 rows and 87 columns
    # Number of samples to be extracted from the posterior distribution
    model.n_posts = 400

    model.seed = 42  # Set seed for reproducibility

    # %% Train the model
    # Fit BEL model
    model.fit(X=X_train, Y=y_train)

    # Sample for the observation
    # Extract n random sample (target CV's).
    # The posterior distribution is computed within the method below.
    y_predicted = model.predict(X_test)

    # %% Visualization

    # Plot raw data
    plot_results(
        model,
        y_predicted=y_predicted,
        X=X_train,
        X_obs=X_test,
        Y=y_train,
        Y_obs=y_test,
        base_dir=sub_dir,
    )

