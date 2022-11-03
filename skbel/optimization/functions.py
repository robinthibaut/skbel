#  Copyright (c) 2022. Robin Thibaut, Ghent University

import numpy as np
from tensorflow_probability import distributions as tfd

__all__ = ["kl_objective"]


def kl_objective(model, y_true, x_test, std=1e-2, it=1):
    """KL objective function

    :param model: keras model
    :param y_true: true value
    :param x_test: test data
    :param std: standard deviation of ideal distribution
    :param it: number of iterations
    :return: KL loss
    """
    scale_tril = [[std] for _ in range(model.output_shape[1])]
    det_dist = tfd.MultivariateNormalTriL(
        loc=y_true.astype(np.float32), scale_tril=scale_tril
    )
    qdis = model(x_test.reshape(1, -1))
    diff = 0
    for i in range(it):
        diff += tfd.kl_divergence(det_dist, qdis).numpy()[
            0
        ]  # kl divergence between "true" and predicted
    diff /= it
    return qdis, diff
