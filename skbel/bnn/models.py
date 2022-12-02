#  Copyright (c) 2022. Robin Thibaut, Ghent University

# EXPERIMENTAL FUNCTIONALITY -- USE AT YOUR OWN RISK

import tensorflow as tf
from sklearn.base import TransformerMixin, MultiOutputMixin, BaseEstimator
from tensorflow import keras as tfk
import tensorflow_probability as tfp

from skbel.nn_utilities import (
    prior_regularize,
    prior_trainable,
    posterior_mean_field,
    neg_log_likelihood,
    MonteCarloDropout,
)

__all__ = [
    "probabilistic_variational_model",
    "PBNN",
    "probabilistic_mcd_model",
    "variational_model",
]


def probabilistic_variational_model(
    input_shape: int,
    output_shape: int,
    n_hidden: int or list,
    kl_weight: float,
    n_layers: int = 2,
    loss=None,
):
    """Define variational model with 2 hidden layers.

    Example:
        # Define model instance

        model = probabilistic_variational_model(
            input_shape=X_train.shape[1],
            output_shape=Y_train.shape[1],
            n_hidden=n_hidden,
            kl_weight=kl_weight,
        )
        model.summary()

        # Run training session

        history = model.fit(
            X_train,
            Y_train,
            epochs=epochs,
            validation_split=0.2,
            callbacks=[
                tfk.callbacks.EarlyStopping(
                    monitor="val_loss", patience=12, restore_best_weights=True
                )
            ],
            verbose=1,
        )

        # plot loss

        plt.plot(history.history["loss"], label="train")
        plt.plot(history.history["val_loss"], label="test")
        plt.legend()
        plt.show()

        # To make a prediction given new data (x_test)

        posterior_distribution = model(x_test.reshape(1, -1))

        # you can then access its mean and variance/covariance matrix
        posterior_mean = posterior_distribution.mean()
        posterior_covariance = posterior_distribution.covariance()

        # you can generate samples from the posterior distribution
        samples = posterior_distribution.sample(100)

    :param input_shape: shape of input
    :param output_shape: shape of output
    :param n_hidden: number of hidden units
    :param kl_weight: weight of KL term
    :param n_layers: number of hidden layers
    :param loss: loss function
    :return: variational model
    """
    if loss is None:
        loss = neg_log_likelihood
    variational_layers = [
        tfp.layers.DenseVariational(
            n_hidden[i],
            make_prior_fn=prior_trainable,
            make_posterior_fn=posterior_mean_field,
            kl_weight=kl_weight,
            activation="relu",
            name=f"dense{i + 1}",
        )
        for i in range(n_layers)
    ]

    model = tfk.Sequential(
        [
            tfk.layers.InputLayer(input_shape=(input_shape,), name="input"),
            *variational_layers,
            tfp.layers.DenseVariational(
                tfp.layers.MultivariateNormalTriL.params_size(output_shape),
                make_prior_fn=prior_trainable,
                make_posterior_fn=posterior_mean_field,
                kl_weight=kl_weight,
                activation="linear",
                name=f"dense_output",
            ),
            tfp.layers.MultivariateNormalTriL(
                output_shape,
                activity_regularizer=tfp.layers.KLDivergenceRegularizer(
                    prior_regularize((output_shape,)), weight=kl_weight
                ),
                name="output",
            ),
        ],
        name="model",
    )

    # Compile model
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=loss)

    return model


# we need to wrap a class around the model to make it compatible with sklearn
class PBNN(TransformerMixin, MultiOutputMixin, BaseEstimator):
    def __init__(
        self,
        input_shape,
        output_shape,
        n_hidden,
        kl_weight,
        n_layers=2,
        epochs=100,
        batch_size=32,
        verbose=0,
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_hidden = n_hidden
        self.kl_weight = kl_weight
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

        self.model = probabilistic_variational_model(
            input_shape, output_shape, n_hidden, kl_weight, n_layers
        )

    def fit(self, X, y):
        self.model.fit(
            X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose
        )
        return self

    def predict(self, X, n_samples=100):
        return self.model(X).sample(n_samples)

    def predict_std(self, X):
        return self.model(X).stddev()

    def score(self, X, y):
        return self.model.evaluate(X, y, verbose=self.verbose)

    # it needs a 'transform' method that does the same as 'predict'
    def transform(self, X):
        return self.predict(X)

    # it needs a 'fit_transform' method that does the same as 'fit' and 'predict'
    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    # it needs a 'inverse_transform' method that does nothing
    def inverse_transform(self, y):
        return y


def variational_model(
    input_shape: int,
    output_shape: int,
    n_hidden: int or list,
    kl_weight: float,
    n_layers: int = 2,
):
    """Define variational model with 2 hidden layers.

    :param input_shape: shape of input
    :param output_shape: shape of output
    :param n_hidden: number of hidden units
    :param kl_weight: weight of KL term
    :param n_layers: number of hidden layers
    :return: variational model
    """

    if isinstance(n_hidden, list):
        # check if list is of correct length
        if len(n_hidden) != n_layers:
            raise ValueError("n_hidden must be a list of length n_layers")

    else:
        # if not a list, make it a list of length n_layers
        n_hidden = [n_hidden] * n_layers

    variational_layers = [
        tfp.layers.DenseVariational(
            n_hidden[i],
            make_prior_fn=prior_trainable,
            make_posterior_fn=posterior_mean_field,
            kl_weight=kl_weight,
            activation="relu",
            name=f"dense{i + 1}",
        )
        for i in range(n_layers)
    ]

    model = tfk.Sequential(
        [
            tfk.layers.InputLayer(input_shape=(input_shape,), name="input"),
            *variational_layers,
            tfk.layers.Dense(
                output_shape,
                activation="linear",
                name="output",
            ),
        ],
        name="model",
    )

    # Compile model
    optimizer = tf.keras.optimizers.Adam()
    # classic MSE loss:
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss)

    return model


# extra models for later
def probabilistic_mcd_model(input_shape, output_shape, n_hidden, kl_weight, rate=0.25):
    """Define mcd model.

    :param input_shape: shape of input
    :param output_shape: shape of output
    :param n_hidden: number of hidden units
    :param kl_weight: weight of KL term
    :param rate: dropout rate
    :return: variational model
    """
    model = tfk.Sequential(
        [
            tfk.layers.InputLayer(input_shape=(input_shape,), name="input"),
            # MC dropout instead of DenseVariational:
            tfk.layers.Dense(n_hidden, activation="relu", name="dense_1"),
            MonteCarloDropout(rate, name="dropout_1"),
            tfk.layers.Dense(n_hidden, activation="relu", name="dense_2"),
            MonteCarloDropout(rate, name="dropout_2"),
            tfk.layers.Dense(
                tfp.layers.MultivariateNormalTriL.params_size(output_shape),
                activation="linear",
                name="distribution_weights",
            ),
            tfp.layers.MultivariateNormalTriL(
                output_shape,
                # activity_regularizer=tfp.layers.KLDivergenceRegularizer(
                #     prior_regularize((output_shape,)), weight=kl_weight
                # ),
                name="output",
            ),
            # if only modeling epistemic uncertainty:
            # tfk.layers.Dense(
            #     len(outputs),
            #     activation="linear",
            #     name="output",
            # ),
        ],
        name="model",
    )

    # Compile model
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=neg_log_likelihood)

    return model


def epistemic_mcd_model(input_shape, output_shape, n_hidden, kl_weight, rate=0.25):
    """Define mcd model with 2 hidden layers.

    :param input_shape: shape of input
    :param output_shape: shape of output
    :param n_hidden: number of hidden units
    :param kl_weight: weight of KL term
    :param rate: dropout rate
    :return: variational model
    """
    model = tfk.Sequential(
        [
            tfk.layers.InputLayer(input_shape=(input_shape,), name="input"),
            tfk.layers.Dense(n_hidden, activation="relu", name="dense_1"),
            MonteCarloDropout(rate, name="dropout_1"),
            tfk.layers.Dense(n_hidden, activation="relu", name="dense_2"),
            MonteCarloDropout(rate, name="dropout_2"),
            tfk.layers.Dense(
                output_shape,
                activation="linear",
                name="output",
            ),
        ],
        name="model",
    )

    # Compile model
    optimizer = tf.keras.optimizers.Adam()
    # classic MSE loss:
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss)

    return model
