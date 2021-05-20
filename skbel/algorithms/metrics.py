#  Copyright (c) 2021. Robin Thibaut, Ghent University

import numpy as np
from scipy.spatial.distance import cdist
from skimage.metrics import structural_similarity


__all__ = ["modified_hausdorff", "structural_similarity"]


def modified_hausdorff(a: np.array, b: np.array) -> float:
    """
    Compute the modified Hausdorff distance between two N-D arrays.
    Distances between pairs are calculated using an Euclidean metric.

    .. [1] M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object
           matching. In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.
           http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=576361

    :param a : (M,N) ndarray. Input array.
    :param b : (O,N) ndarray. Input array.

    :return: d : double. The modified Hausdorff distance between arrays `a` and `b`.

    :raise ValueError: An exception is thrown if `a` and `b` do not have the same number of columns.

    Another Python implementation:
    https://github.com/sapphire008/Python/blob/master/generic/HausdorffDistance.py

    """

    a = np.asarray(a, dtype=np.float64, order="c")
    b = np.asarray(b, dtype=np.float64, order="c")

    if a.shape[1] != b.shape[1]:
        raise ValueError("a and b must have the same number of columns")

    # Compute distance between each pair of the two collections of inputs.
    d = cdist(a, b)
    # dim(d) = (M, O)
    fhd = np.mean(np.min(d, axis=0))  # Mean of minimum values along rows
    rhd = np.mean(np.min(d, axis=1))  # Mean of minimum values along columns

    return max(fhd, rhd)
