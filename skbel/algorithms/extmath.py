import numpy as np

__all__ = [
    "get_block",
    "matrix_paste",
]

from scipy.spatial import distance_matrix


def get_block(pm: np.array, i: int) -> np.array:
    """Extracts block from a 2x2 partitioned matrix.

    :param pm: Partitioned matrix
    :param i: Block index [[1,2], [3,4]]
    :return: Block
    """

    b = pm.shape[0] // 2

    if i == 1:
        return pm[:b, :b]
    if i == 2:
        return pm[:b, b:]
    if i == 3:
        return pm[b:, :b]
    if i == 4:
        return pm[b:, b:]
    else:
        return 0


def matrix_paste(c_big: np.array, c_small: np.array) -> list:
    """Pastes a small matrix into a big matrix.

    :param c_big: Big matrix
    :param c_small: Small matrix
    """
    # Compute distance matrix between refined and dummy grid.
    dm = distance_matrix(c_big, c_small)
    inds = [
        np.unravel_index(np.argmin(dm[i], axis=None), dm[i].shape)[0]
        for i in range(dm.shape[0])
    ]
    return inds
