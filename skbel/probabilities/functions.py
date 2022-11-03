#  Copyright (c) 2022. Robin Thibaut, Ghent University

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

tfd = tfp.distributions

__all__ = ["prior_fn", "posterior_fn", "prior_trainable", "posterior_mean_field"]


def prior_fn(kernel_size, bias_size, dummy_input):
    n = kernel_size + bias_size  # number of weights
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model


# Define variational posterior weight distribution as multivariate Gaussian.
# Note that the learnable parameters for this distribution are the means,
# variances, and covariances.
def posterior_fn(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n),
                dtype=dtype,
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.0))
    return tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(2 * n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfd.Independent(
                    tfd.Normal(
                        loc=t[..., :n], scale=1e-5 + tf.nn.softplus(c + t[..., n:])
                    ),
                    reinterpreted_batch_ndims=1,
                )
            ),
        ]
    )


def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfd.Independent(
                    tfd.Normal(loc=t, scale=1), reinterpreted_batch_ndims=1
                )
            ),
        ]
    )


# model = tfk.Sequential(
#     [
#         tfk.layers.InputLayer(input_shape=(len(inputs),), name="input"),
#         tfk.layers.Dense(32, activation="relu", name="dense_1"),
#         tfk.layers.Dense(32, activation="relu", name="dense_2"),
#         tfk.layers.Dense(
#             tfp.layers.MultivariateNormalTriL.params_size(len(outputs)),
#             activation=None,
#             name="distribution_weights",
#         ),
#         tfp.layers.MultivariateNormalTriL(
#             len(outputs),
#             activity_regularizer=tfp.layers.KLDivergenceRegularizer(
#                 prior, weight=1 / n_batches
#             ),
#             name="output",
#         ),
#     ],
#     name="model",
# )
