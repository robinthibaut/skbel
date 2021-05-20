#  Copyright (c) 2021. Robin Thibaut, Ghent University
"""
Boilerplate
-----------
Grid geometry parameters need to be passed around modules.
Defining data classes allows to avoid declaring those parameters several times.
"""

import os
import platform
from dataclasses import dataclass
from os.path import dirname, join

import numpy as np


class Machine(object):
    computer: str = platform.node()


class Setup:
    @dataclass
    class Directories:
        """Define main directories and file names"""

        # Content directory
        main_dir: str = dirname(os.path.abspath(__file__))
        storage_dir: str = join(main_dir, "storage")
        data_dir: str = join(main_dir, "datasets")
        hydro_res_dir: str = join(data_dir, "forwards")
        forecasts_dir: str = join(storage_dir, "forecasts")
        forecasts_base_dir: str = join(forecasts_dir, "base")
        grid_dir: str = join(main_dir, "spatial", "parameters")
        test_dir: str = join(main_dir, "testing")
        ref_dir: str = join(test_dir, "reference")

    @dataclass
    class Files:
        """Class to keep track of important file names"""

        # Output file names
        project_name: str = "whpa"
        predictor_name: str = "d"
        target_name: str = "h"

        # Files stored in the raw, forward dataset folder
        hk_file: str = "hk0.npy"
        predictor_file: str = "bkt.npy"
        target_file: str = "pz.npy"

        output_files = [hk_file, predictor_file, target_file]

        sgems_file: str = "hd.sgems"
        command_file: str = "sgsim_commands.py"

        sgems_family = [sgems_file, command_file, hk_file]

        # Files that are stored after preprocessing
        predictor_pickle: str = "d_pca.pkl"
        target_pickle: str = "h_pca.pkl"
        model_pickle: str = "cca.pkl"
        target_post_bin: str = "post.npy"
        target_pc_bin: str = "target_pc.npy"
        predictor_training_bin: str = "training_curves.npy"
        predictor_test_bin: str = "test_curves.npy"

    @dataclass
    class GridDimensions:
        """Class for keeping track of grid dimensions"""

        x_lim: float = 1500.0
        y_lim: float = 1000.0
        z_lim: float = 1.0

        dx: float = 10.0  # Block x-dimension
        dy: float = 10.0  # Block y-dimension
        dz: float = 10.0  # Block z-dimension

        xo: float = 0.0
        yo: float = 0.0
        zo: float = 0.0

        nrow: int = y_lim // dy  # Number of rows
        ncol: int = x_lim // dx  # Number of columns
        nlay: int = 1  # Number of layers

        # Refinement parameters around the pumping well.
        # 150 meters from the pumping well coordinates, grid cells will have dimensions 9x9 and so on...
        r_params = np.array(
            [
                [9, 150],
                [8, 100],
                [7, 90],
                [6, 80],
                [5, 70],
                [4, 60],
                [3, 50],
                [2.5, 40],
                [2, 30],
                [1.5, 20],
                [1, 10],
            ]
        )  # ...10 meters from the pumping well coordinates, grid cells will have dimensions 1x1.

    @dataclass
    class Focus:
        """Geometry of the focused area on the main grid, enclosing all wells, as to reduce computation time"""

        x_range = [800, 1150]
        y_range = [300, 700]
        # Defines cell dimensions for the signed distance computation.
        cell_dim: float = 4

    @dataclass
    class Wells:
        """Wells coordinates"""

        wells_data = {
            "pumping0": {
                "coordinates": [1000, 500],
                "rates": [-1000, -1000, -1000],
                "color": "w",
            },
            "injection0": {
                "coordinates": [950, 450],
                "rates": [0, 24, 0],
                "color": "b",
            },
            "injection1": {
                "coordinates": [930, 560],
                "rates": [0, 24, 0],
                "color": "g",
            },
            "injection2": {
                "coordinates": [900, 505],
                "rates": [0, 24, 0],
                "color": "r",
            },
            "injection3": {
                "coordinates": [1068, 515],
                "rates": [0, 24, 0],
                "color": "c",
            },
            "injection4": {
                "coordinates": [1030, 580],
                "rates": [0, 24, 0],
                "color": "m",
            },
            "injection5": {
                "coordinates": [1050, 470],
                "rates": [0, 24, 0],
                "color": "y",
            },
        }

        # Injection wells in use for prediction (default: all)
        combination = np.arange(1, len(wells_data))
        colors = ["b", "g", "r", "c", "m", "y"]

    @dataclass
    class HyperParameters:
        """Learning hyper parameters"""

        # Predictor
        n_pc_predictor: int = 50
        n_tstp: int = 200

        # Target
        n_pc_target: int = 30

        # Posterior
        # Size of data set
        n_total: int = 500  # Parameter to optimize
        n_training: int = int(n_total * 0.8)
        n_test: int = int(n_total * 0.2)
        # Sample size
        n_posts: int = n_training

    @dataclass
    class ModelParameters:
        """Model hyper parameters"""

        # Prior K
        k_min: float = 1.4
        k_max: float = 2
        k_std: float = 0.4

    @dataclass
    class ED:
        """Experimental Design"""

        metric = None
