"""Focused tests for DCT unsupervised fit contract."""

import numpy as np

from skbel.preprocessing.dct import DiscreteCosineTransform2D, dct2, idct2


def test_fit_X_only():
    """fit(X) should work without y argument."""
    X = np.arange(8.0).reshape(2, 2, 2)
    est = DiscreteCosineTransform2D()
    result = est.fit(X)
    assert result is est


def test_fit_X_none():
    """fit(X, None) should work."""
    X = np.arange(8.0).reshape(2, 2, 2)
    est = DiscreteCosineTransform2D()
    result = est.fit(X, None)
    assert result is est


def test_fit_X_y():
    """fit(X, y) should work (y is ignored)."""
    X = np.arange(8.0).reshape(2, 2, 2)
    y = np.array([1, 2])
    est = DiscreteCosineTransform2D()
    result = est.fit(X, y)
    assert result is est


def test_fit_no_mutation():
    """fit() should not mutate caller X or learned state."""
    X = np.arange(8.0).reshape(2, 2, 2).copy()
    X_orig = X.copy()

    est = DiscreteCosineTransform2D()
    est.fit(X)

    assert np.array_equal(X, X_orig), "fit() mutated caller X"
    assert est.n_rows is None, "fit() should not set n_rows before transform"
    assert est.n_cols is None, "fit() should not set n_cols before transform"


def test_fit_transform_agrees():
    """fit_transform(X) should agree with fit(X).transform(X)."""
    X = np.arange(8.0).reshape(2, 2, 2)

    est1 = DiscreteCosineTransform2D()
    result1 = est1.fit_transform(X)

    est2 = DiscreteCosineTransform2D()
    result2 = est2.fit(X).transform(X)

    np.testing.assert_allclose(result1, result2, rtol=1e-10)


def test_full_cutoff_roundtrip():
    """Full-cutoff transform/inverse should preserve shape and values."""
    X = np.arange(8.0).reshape(2, 2, 2)

    est = DiscreteCosineTransform2D()
    X_transformed = est.fit(X).transform(X)
    X_recovered = est.inverse_transform(X_transformed)

    assert X_recovered.shape == X.shape, "Shape not preserved"
    np.testing.assert_allclose(X_recovered, X, atol=1e-13)


def test_truncated_roundtrip_with_oracle():
    """Truncated transform/inverse should agree with DCT oracle with zero-padding."""
    X = np.arange(8.0).reshape(2, 2, 2)
    m_cut, n_cut = 2, 1

    est = DiscreteCosineTransform2D(m_cut=m_cut, n_cut=n_cut)
    X_transformed = est.fit(X).transform(X)
    X_recovered = est.inverse_transform(X_transformed)

    # Manual oracle: DCT with zero-padding recovery
    expected_recovered = []
    for i in range(X.shape[0]):
        dct_slice = dct2(X[i])
        truncated = dct_slice[:m_cut, :n_cut]
        padded = np.zeros((X.shape[1], X.shape[2]))
        padded[:m_cut, :n_cut] = truncated
        recovered_slice = idct2(padded)
        expected_recovered.append(recovered_slice)
    expected_recovered = np.array(expected_recovered)

    assert X_recovered.shape == expected_recovered.shape
    np.testing.assert_allclose(X_recovered, expected_recovered, atol=1e-13)
