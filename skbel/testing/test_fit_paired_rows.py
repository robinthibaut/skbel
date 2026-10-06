#  Copyright (c) 2022. Robin Thibaut, Ghent University

"""Regression tests for paired-row handling in ``BEL.fit``."""

import numpy as np
import pytest
from sklearn.cross_decomposition import CCA

from skbel.learning.bel import BEL


def _data(n_rows, n_x=3, n_y=2, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_rows, n_x)), rng.normal(size=(n_rows, n_y))


@pytest.mark.parametrize("n_rows", [1, 2, 3])
def test_default_passthrough_preserves_pairs(n_rows):
    X, Y = _data(n_rows)
    bel = BEL().fit(X, Y)
    assert bel.X_f.shape == X.shape
    assert bel.Y_f.shape == Y.shape
    np.testing.assert_array_equal(bel.X_f, X)
    np.testing.assert_array_equal(bel.Y_f, Y)


def test_raw_unequal_rows_fail():
    X, _ = _data(4)
    _, Y = _data(3)
    with pytest.raises(ValueError, match="same number of rows"):
        BEL().fit(X, Y)


def test_cached_and_mixed_unequal_rows_fail():
    X, Y = _data(4)
    bel = BEL()
    bel.x_pre_processed = X
    bel.y_pre_processed = Y[:3]
    with pytest.raises(ValueError, match="same number of rows"):
        bel.fit()

    mixed = BEL()
    mixed.x_pre_processed = X
    with pytest.raises(ValueError, match="same number of rows"):
        mixed.fit(X, Y[:2])


def test_valid_paired_cca_unchanged():
    X, Y = _data(20)
    bel = BEL(regression_model=CCA(n_components=2)).fit(X, Y)
    xc, yc = CCA(n_components=2).fit_transform(X, Y)
    np.testing.assert_allclose(bel.X_f, xc)
    np.testing.assert_allclose(bel.Y_f, yc)


def test_no_cca_fallback_unchanged():
    # More components than allowed makes CCA raise ValueError -> intentional no-CCA path.
    X, Y = _data(10)
    bel = BEL(regression_model=CCA(), n_comp_cca=0).fit(X, Y)
    np.testing.assert_array_equal(bel.X_f, X)
    np.testing.assert_array_equal(bel.Y_f, Y)
