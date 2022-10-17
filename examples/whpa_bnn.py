#  Copyright (c) 2021. Robin Thibaut, Ghent University

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
