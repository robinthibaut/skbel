"""Finite tests: forward ``BEL.transform`` with the default no-op regression."""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from skbel.learning.bel import BEL


def _data(n_rows, n_x, n_y):
    x = np.arange(n_rows * n_x, dtype=float).reshape(n_rows, n_x) ** 2 + 1.0
    y = 10.0 - np.arange(n_rows * n_y, dtype=float).reshape(n_rows, n_y) * 1.5
    return x, y


CASES = [(n, nx, ny) for n in (1, 2, 3) for nx, ny in ((1, 1), (2, 2), (1, 2), (2, 1), (2, 3))]


@pytest.mark.parametrize(("n_rows", "n_x", "n_y"), CASES)
def test_default_forward_transform_exact(n_rows, n_x, n_y):
    x, y = _data(n_rows, n_x, n_y)
    x0, y0 = x.copy(), y.copy()
    bel = BEL().fit(x, y)

    xp, yp = bel.transform(X=x, Y=y)
    assert xp.shape == (n_rows, n_x) and yp.shape == (n_rows, n_y)
    np.testing.assert_array_equal(xp, x)
    np.testing.assert_array_equal(yp, y)

    np.testing.assert_array_equal(bel.transform(X=x), x)
    yo = bel.transform(X=None, Y=y)
    assert yo.shape == (n_rows, n_y)
    np.testing.assert_array_equal(yo, y)

    xf, yf = BEL().fit_transform(x, y)
    np.testing.assert_array_equal(xf, bel.X_f)
    np.testing.assert_array_equal(yf, bel.Y_f)
    np.testing.assert_array_equal(xf, xp)
    np.testing.assert_array_equal(yf, yp)

    np.testing.assert_array_equal(x, x0)
    np.testing.assert_array_equal(y, y0)


def test_paired_order_is_preserved():
    x, y = _data(3, 2, 3)
    bel = BEL().fit(x, y)
    xp, yp = bel.transform(X=x[::-1], Y=y[::-1])
    np.testing.assert_array_equal(xp, x[::-1])
    np.testing.assert_array_equal(yp, y[::-1])


def test_pre_and_post_processing_still_applied():
    x, y = _data(3, 2, 3)
    x0, y0 = x.copy(), y.copy()
    bel = BEL(
        X_pre_processing=Pipeline([("f", FunctionTransformer(lambda a: 2.0 * a + 1.0))]),
        Y_pre_processing=Pipeline([("f", FunctionTransformer(lambda a: 3.0 * a - 2.0))]),
        X_post_processing=Pipeline([("s", StandardScaler())]),
        Y_post_processing=Pipeline([("s", StandardScaler())]),
    ).fit(x, y)

    xt, yt = 2.0 * x + 1.0, 3.0 * y - 2.0
    exp_x = (xt - xt.mean(axis=0)) / xt.std(axis=0)
    exp_y = (yt - yt.mean(axis=0)) / yt.std(axis=0)
    assert not np.allclose(exp_x, x) and not np.allclose(exp_y, y)

    xp, yp = bel.transform(X=x, Y=y)
    np.testing.assert_allclose(xp, exp_x)
    np.testing.assert_allclose(yp, exp_y)
    np.testing.assert_allclose(bel.transform(X=x), exp_x)
    np.testing.assert_allclose(bel.transform(X=None, Y=y), exp_y)

    # Row subset uses the training statistics of each independent pipeline.
    xs, ys = bel.transform(X=x[1:2], Y=y[1:2])
    np.testing.assert_allclose(xs, exp_x[1:2])
    np.testing.assert_allclose(ys, exp_y[1:2])

    xf, yf = bel.fit_transform(x, y)
    np.testing.assert_allclose(xf, exp_x)
    np.testing.assert_allclose(yf, exp_y)
    np.testing.assert_array_equal(x, x0)
    np.testing.assert_array_equal(y, y0)


class _ActiveRegression:
    """Non-noop regression sentinel: shifts X by 100 and Y by 200 on transform."""

    def __init__(self):
        self.calls = []

    def fit_transform(self, X, y):
        self.x_loadings_ = np.zeros((X.shape[1], 1))
        return X, y

    def transform(self, X, Y=None):
        self.calls.append((np.shape(X), None if Y is None else np.shape(Y)))
        if Y is None:
            return X + 100.0
        return X + 100.0, Y + 200.0


def test_active_regression_is_not_bypassed():
    x, y = _data(3, 2, 3)
    reg = _ActiveRegression()
    bel = BEL(regression_model=reg).fit(x, y)

    xp, yp = bel.transform(X=x, Y=y)
    np.testing.assert_array_equal(xp, x + 100.0)
    np.testing.assert_array_equal(yp, y + 200.0)
    np.testing.assert_array_equal(bel.transform(X=x), x + 100.0)
    np.testing.assert_array_equal(bel.transform(X=None, Y=y), y + 200.0)
    assert reg.calls == [((3, 2), (3, 3)), ((3, 2), None), ((3, 2), (3, 3))]


def test_active_pipeline_is_not_treated_as_passthrough():
    x, y = _data(3, 2, 2)
    bel = BEL(regression_model=Pipeline([("s", StandardScaler())])).fit(x, y)
    # Single-argument forward path is the pipeline's own transform, not identity.
    np.testing.assert_allclose(bel.transform(X=x), (x - x.mean(axis=0)) / x.std(axis=0))
    # Paired call reaches the active pipeline (not silently passed through).
    with pytest.raises(TypeError):
        bel.transform(X=x, Y=y)
