#  Copyright (c) 2022. Robin Thibaut, Ghent University

import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

__all__ = [
    "neg_log_likelihood",
    "MonteCarloDropout",
    "prior_trainable",
    "posterior_mean_field",
    "prior_fn",
    "posterior_fn",
    "prior_regularize",
    "random_gaussian_initializer",
]


class MonteCarloDropout(tfk.layers.Dropout):
    """Monte Carlo Dropout layer."""

    def call(self, inputs_):
        return super().call(inputs_, training=True)


def neg_log_likelihood(x, rv_x):
    """Negative log likelihood of the data under the distribution."""
    return -rv_x.log_prob(x)


# Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.0))
    return tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(
                2 * n,
                dtype=dtype,
                initializer=lambda shape, dtype: random_gaussian_initializer(
                    shape, dtype
                ),
                trainable=True,
            ),
            # tfp.layers.VariableLayer(2 * n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfd.Independent(
                    tfd.Normal(
                        loc=t[..., :n],
                        scale=1e-5 + 1e-2 * tf.nn.softplus(c + t[..., n:]),  # softplus ensures positivity and avoids numerical instability
                    ),
                    reinterpreted_batch_ndims=1, # each weight is independent
                )  # reinterpreted_batch_ndims=1 means that the last dimension is the event dimension
            ),
        ]
    )


# Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
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


def prior_fn(kernel_size, bias_size, dummy_input):
    n = kernel_size + bias_size
    prior_model = tfk.Sequential(
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
    posterior_model = tfk.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


def random_gaussian_initializer(shape, dtype="float32"):
    n = int(shape / 2)
    loc_norm = tf.random_normal_initializer(mean=0.0, stddev=0.1)
    loc = tf.Variable(initial_value=loc_norm(shape=(n,), dtype=dtype))
    scale_norm = tf.random_normal_initializer(mean=-3.0, stddev=0.1)
    scale = tf.Variable(initial_value=scale_norm(shape=(n,), dtype=dtype))
    return tf.concat([loc, scale], 0)


def prior_regularize(output_shape):
    prior = tfd.Independent(
        tfd.Normal(loc=tf.zeros(len(output_shape)), scale=1.0),
        reinterpreted_batch_ndims=1,
    )
    return prior


class RBFKernelFn(tf.keras.layers.Layer):
    """RBF kernel function.
    https://www.tensorflow.org/probability/examples/Probabilistic_Layers_Regression#case_5_functional_uncertainty
    """

    def __init__(self, **kwargs):
        super(RBFKernelFn, self).__init__(**kwargs)
        dtype = kwargs.get("dtype", None)

        self._amplitude = self.add_variable(
            initializer=tf.constant_initializer(0), dtype=dtype, name="amplitude"
        )

        self._length_scale = self.add_variable(
            initializer=tf.constant_initializer(0), dtype=dtype, name="length_scale"
        )

    def call(self, x):
        # Never called -- this is just a layer, so it can hold variables
        # in a way Keras understands.
        return x

    @property
    def kernel(self):
        return tfp.math.psd_kernels.ExponentiatedQuadratic(
            amplitude=tf.nn.softplus(0.1 * self._amplitude),
            length_scale=tf.nn.softplus(5.0 * self._length_scale),
        )
