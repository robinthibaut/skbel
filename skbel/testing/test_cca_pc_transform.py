"""Finite regression tests: BEL.cca_pc_transform with the installed sklearn CCA."""

import numpy as np
import pytest
from sklearn.cross_decomposition import CCA

from skbel.learning.bel import BEL


@pytest.fixture
def fitted():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 4))
    y = x[:, :3] @ rng.normal(size=(3, 3)) + 0.1 * rng.normal(size=(12, 3))
    cca = CCA(n_components=2).fit(x, y)
    bel = BEL(regression_model=cca)
    xt = rng.normal(size=(5, 4))
    yt = rng.normal(size=(5, 3))
    return bel, cca, xt, yt


@pytest.mark.parametrize("rows", [slice(None), slice(0, 1)])
def test_paired_matches_positional(fitted, rows):
    bel, cca, xt, yt = fitted
    x, y = xt[rows], yt[rows]
    xc, yc = bel.cca_pc_transform(X=x, Y=y)
    xr, yr = cca.transform(x, y)
    assert xc.shape == (x.shape[0], 2) and yc.shape == (x.shape[0], 2)
    np.testing.assert_allclose(xc, xr)
    np.testing.assert_allclose(yc, yr)


@pytest.mark.parametrize("rows", [slice(None), slice(0, 1)])
def test_x_only_matches_positional(fitted, rows):
    bel, cca, xt, _ = fitted
    x = xt[rows]
    np.testing.assert_allclose(bel.cca_pc_transform(X=x), cca.transform(x))


@pytest.mark.parametrize("rows", [slice(None), slice(0, 1)])
def test_y_only_matches_positional_and_algebra(fitted, rows):
    bel, cca, xt, yt = fitted
    x, y = xt[rows], yt[rows]
    yc = bel.cca_pc_transform(Y=y)
    assert yc.shape == (y.shape[0], 2)
    np.testing.assert_allclose(yc, cca.transform(x, y)[1])
    # Direct algebra: standardized target times y_rotations_
    expected = ((y - cca._y_mean) / cca._y_std) @ cca.y_rotations_
    np.testing.assert_allclose(yc, expected, atol=1e-12)


def test_callers_arrays_unchanged(fitted):
    bel, _, xt, yt = fitted
    x0, y0 = xt.copy(), yt.copy()
    bel.cca_pc_transform(X=xt)
    bel.cca_pc_transform(Y=yt)
    bel.cca_pc_transform(X=xt, Y=yt)
    np.testing.assert_array_equal(xt, x0)
    np.testing.assert_array_equal(yt, y0)


@pytest.fixture
def public(fitted):
    bel, _, xt, yt = fitted
    rng = np.random.default_rng(1)
    x = rng.normal(size=(12, 4))
    y = x[:, :3] @ rng.normal(size=(3, 3)) + 0.1 * rng.normal(size=(12, 3))
    bel = BEL(regression_model=CCA(n_components=2), n_comp_cca=2).fit(x, y)
    return bel, bel.regression_model, xt, yt


@pytest.mark.parametrize("rows", [slice(None), slice(0, 1)])
def test_public_transform_matches_positional(public, rows):
    bel, cca, xt, yt = public
    x, y = xt[rows], yt[rows]
    xr, yr = cca.transform(x, y)
    xp, yp = bel.transform(X=x, Y=y)
    np.testing.assert_allclose(xp, bel.X_post_processing.transform(xr))
    np.testing.assert_allclose(yp, bel.Y_post_processing.transform(yr))
    np.testing.assert_allclose(bel.transform(X=x), bel.X_post_processing.transform(xr))
    yo = bel.transform(X=None, Y=y)  # sklearn output wrapper requires X to be passed
    assert yo.shape == (y.shape[0], 2)
    np.testing.assert_allclose(yo, bel.Y_post_processing.transform(yr))


def test_public_fit_transform_and_inputs_unchanged(public):
    bel, _, xt, yt = public
    x0, y0 = xt.copy(), yt.copy()
    xp, yp = bel.fit_transform(xt, yt)
    assert xp.shape == (5, 2) and yp.shape == (5, 2)
    np.testing.assert_array_equal(xt, x0)
    np.testing.assert_array_equal(yt, y0)


def test_unrelated_errors_not_swallowed(fitted):
    bel, _, xt, _ = fitted
    with pytest.raises(ValueError):
        bel.cca_pc_transform(X=xt[:, :2])  # wrong feature count
