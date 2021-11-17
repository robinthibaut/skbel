#  Copyright (c) 2021. Robin Thibaut, Ghent University

import itertools
import numpy as np
import os
import warnings
import shutil
import types
from typing import List

__all__ = [
    "FLOAT_DTYPES",
    "Combination",
    "Function",
    "flatten_array",
    "data_read",
    "folder_reset",
    "dirmaker",
    "combinator",
]

FLOAT_DTYPES = (np.float64, np.float32, np.float16)
Combination = List[List[int]]
Function = types.FunctionType


def flatten_array(arr: np.array) -> np.array:
    """Flattens a numpy array
    :param arr: np.array: Numpy array.
    :return: np.array: Flattened array.
    """
    arr_flat = np.array([item for sublist in arr for item in sublist])  # Flatten
    return arr_flat.reshape(1, -1)  # Reshape


def data_read(
    file: str = None,
    start: int = 0,
    end: int = None,
    step: int = 1,
    delimiter: str = None,
):
    """
    Reads data from a file. It needs to be a text file and the data needs to be separated by a space or tab (default)
    or by a delimiter specified by the user.
    :param file: str: File path, such as 'data.txt'.
    :param start: int: Starting line, default is 0.
    :param end: int: Ending line, default is None (last line).
    :param step: int: Step, default is 1 (every line).
    :param delimiter: str: Delimiter, default is None (space).
    :return: Data contained in file. np.array if data can be converted to float, else list.
    """
    with open(file, "r") as fr:  # Open file
        lines = fr.readlines()[
            start:end:step
        ]  # Read lines. The content is sliced according to the parameters.
        try:  # Try to convert to float
            op = np.array(
                [list(map(float, line.split(delimiter))) for line in lines],
                dtype=object,
            )  # The map function converts to float. The split function splits the line according to the delimiter.
        except ValueError:  # If not, keep as string.
            op = [line.split(delimiter) for line in lines]
    return op  # Return data


def folder_reset(folder: str, exceptions: list = None):
    """Deletes files in folder
    :param folder: str: Folder path.
    :param exceptions: list: List of files to keep.
    """
    if not isinstance(exceptions, (list, tuple)):
        exceptions = [exceptions]
    try:
        for filename in os.listdir(folder):
            if filename not in exceptions:
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    warnings.warn("Failed to delete %s. Reason: %s" % (file_path, e))
    except FileNotFoundError:
        pass


def dirmaker(dird: str, erase: bool = False):
    """
    Given a folder path, check if it exists, and if not, creates it.
    :param dird: str: Directory path.
    :param erase: bool: Whether to delete existing folder or not.
    :return: None
    """
    try:
        if not os.path.exists(dird):
            os.makedirs(dird)
            return 0
        else:
            if erase:
                shutil.rmtree(dird)
                os.makedirs(dird)
            return 1
    except Exception as e:
        warnings.warn(e)
        return 0


def combinator(combi):
    """Given a n-sized 1D array, generates all possible configurations, from size 1 to n-1.
    'None' will indicate to use the original combination.
    :param combi: list: List of size n.
    :return: list: List of combinations.
    """
    cb = [
        list(itertools.combinations(combi, i)) for i in range(1, combi[-1] + 1)
    ]  # Get all possible wel combinations
    # Flatten
    cb = [item for sublist in cb for item in sublist]
    return cb
