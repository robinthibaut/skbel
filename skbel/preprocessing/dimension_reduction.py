"""Collection of functions to use the PCA class from scikit-learn."""

#  Copyright (c) 2021. Robin Thibaut, Ghent University


import numpy as np
import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from skbel import utils

__all__ = ["PC"]


class PC:
    def __init__(
        self,
        name: str,
        training_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None,
        directory: str = None,
    ):
        """
        Given a set of training data and one observation (optional), performs necessary dimension reduction
        and transformations.
        :param name: str: name of the dataset (e.g. 'data', 'target'...)
        :param training_df: pd.DataFrame Training set
        :param test_df: pd.DataFrame: Test set
        :param directory: str: Path to the folder in which to save the pickle
        """
        logger.info("Initiating dimension reduction")
        self.directory = directory
        self.name = name  # str, name of the object

        self.training_df = training_df
        logger.info(f"Training set {self.training_df.shape}")
        self.test_df = test_df

        try:
            logger.info(f"Test set {self.test_df.shape}")
        except AttributeError:
            logger.info("No test set")

        self.training_pc_df = None
        self.test_pc_df = None

        # Divide the sample by their standard deviation.
        self.scaler = StandardScaler(with_mean=False)
        # The samples are automatically scaled by scikit-learn PCA()
        self.operator = PCA()  # PCA operator (scikit-learn instance)
        self.pipe = make_pipeline(self.scaler, self.operator, verbose=False)

        self.n_samples = self.training_df.shape[0]
        self.n_pc_cut = None  # Number of components to keep

        # Number of training samples
        self.training_pc = None  # Training PCA scores

    def training_fit_transform(self):
        """
        Instantiate the PCA object and transforms training data to scores.
        :return: numpy.array: PC training
        """

        logger.info("Fitting and transforming training data")
        self.pipe.fit(self.training_df)
        training_pc = self.pipe.transform(self.training_df.to_numpy())

        # Store PC dataframe
        self.training_pc_df = utils.i_am_framed(
            array=training_pc, ids=self.training_df.index
        )

        return training_pc

    def test_transform(self, test_roots: list = None):
        """
        Transforms observation to PC scores.
        :param test_roots: list: List containing observation id (str)
        :return: numpy.array: Observation PC
        """

        if test_roots:
            selection = self.test_df.loc[test_roots]
            ids = test_roots
        else:
            selection = self.test_df.to_numpy()
            ids = self.test_df.index

        logger.info("Fitting and transforming test data")
        # Transform prediction data into principal components
        pc_prediction = self.pipe.transform(selection)

        # Store PC dataframe
        self.test_pc_df = utils.i_am_framed(array=pc_prediction, ids=ids)

        return pc_prediction

    def perc_2_comp(self, perc: float) -> int:
        """
        Given an explained variance percentage, returns the number of components
        necessary to obtain that level.
        :param perc: float: Percentage between 0 and 1
        """
        evr = np.cumsum(self.operator.explained_variance_ratio_)
        self.n_pc_cut = len(np.where(evr <= perc)[0])

        return self.n_pc_cut

    def perc_comp(self, n_c: int) -> float:
        """
        Returns the explained variance percentage given a number of components n_c.
        :param n_c: int: Number of components to keep
        """
        evr = np.cumsum(self.operator.explained_variance_ratio_)

        return evr[n_c - 1]

    def comp_refresh(self, n_comp: int = None) -> (np.array, np.array):
        """
        Given a number of components to keep, returns the PC array with the corresponding shape.
        :param n_comp: int: Number of components
        :return:
        """

        if n_comp is not None:
            self.n_pc_cut = (
                n_comp  # Assign the number of components in the class for later use
            )

        # Reloads the original training components
        pc_training = self.training_pc_df.to_numpy()
        pc_training = pc_training[:, : self.n_pc_cut]  # Cut

        pc_prediction = self.test_pc_df

        if self.test_pc_df is not None:
            pc_prediction = (
                self.test_pc_df.to_numpy()
            )  # Reloads the original test components
            pc_prediction = pc_prediction[:, : self.n_pc_cut]  # Cut

        return pc_training, pc_prediction

    def random_pc(self, n_rand: int):
        """
        Randomly selects PC components from the original training matrix.
        :param n_rand: int: Number of random PC to use
        :return numpy.array: Random PC scores
        """
        rand_rows = np.random.choice(
            self.n_samples, n_rand
        )  # Selects n_posts rows from the training array
        # Extracts those rows, from the number of
        score_selection = self.training_pc_df[rand_rows][self.n_pc_cut :]
        # components used until the end of the array.

        # For each column of shape n_samples, n_components, selects a random PC component to add.
        test = [
            np.random.choice(score_selection[:][i])
            for i in range(score_selection.shape[1])
        ]

        return np.array(test)

    def custom_inverse_transform(self, pc_to_invert: np.array, n_comp: int = None):
        """
        Inverse fit_transform PC based on the desired number of PC (stored in the shape of the argument).
        The self.operator.components contains all components.
        :param pc_to_invert: np.array: (n_samples, n_components) PC array to back-fit_transform to physical space
        :param n_comp: int: Number of components to back-fit_transform with
        :return: numpy.array: Back transformed array
        """
        if n_comp is None:
            n_comp = self.n_pc_cut

        # TODO: (optimization) only fit after dimension check
        op_cut = make_pipeline(self.scaler, PCA(n_components=n_comp))
        op_cut.fit(self.training_df)

        inv = op_cut.inverse_transform(pc_to_invert[:, :n_comp])

        return inv
