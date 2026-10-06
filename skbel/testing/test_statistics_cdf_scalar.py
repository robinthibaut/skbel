#  Copyright (c) 2021. Robin Thibaut, Ghent University

import numpy as np
import pytest

from skbel.algorithms.statistics import get_cdf


class SimplePDF:
    """Minimal PDF fixture for testing CDF contract."""

    def __init__(self, x_vals, y_vals):
        self.x = np.array(x_vals)
        self.y = np.array(y_vals)

    def __call__(self, x):
        return np.interp(x, self.x, self.y)


@pytest.fixture
def uniform_pdf():
    """Uniform density on [0, 1] with 129 points."""
    x = np.linspace(0, 1, 129)
    y = np.ones_like(x)
    return SimplePDF(x, y)


@pytest.fixture
def nonuniform_pdf():
    """Nonuniform (triangular) density on [0, 1]."""
    x = np.linspace(0, 1, 129)
    y = 2 * (1 - np.abs(x - 0.5))  # Peak at 0.5
    return SimplePDF(x, y)


class TestScalarInputs:
    """Test Python and NumPy scalar inputs."""

    def test_float_scalar_boundary_zero(self, uniform_pdf):
        cdf = get_cdf(uniform_pdf)
        result = cdf(0.0)
        assert isinstance(result, (float, np.floating))
        assert result == 0.0

    def test_float_scalar_boundary_one(self, uniform_pdf):
        cdf = get_cdf(uniform_pdf)
        result = cdf(1.0)
        assert isinstance(result, (float, np.floating))
        assert result == 1.0

    def test_float_scalar_interior(self, uniform_pdf):
        cdf = get_cdf(uniform_pdf)
        result = cdf(0.5)
        assert isinstance(result, (float, np.floating))
        assert 0.45 < result < 0.55

    def test_int_scalar(self, uniform_pdf):
        cdf = get_cdf(uniform_pdf)
        result = cdf(0)
        assert isinstance(result, (float, np.floating, int, np.integer))
        assert result == 0.0

    def test_numpy_float64_scalar(self, uniform_pdf):
        cdf = get_cdf(uniform_pdf)
        result = cdf(np.float64(0.5))
        assert isinstance(result, (float, np.floating))
        assert 0.45 < result < 0.55

    def test_numpy_int32_scalar(self, uniform_pdf):
        cdf = get_cdf(uniform_pdf)
        result = cdf(np.int32(0))
        assert isinstance(result, (float, np.floating, int, np.integer))
        assert result == 0.0

    def test_scalar_below_lower_bound(self, uniform_pdf):
        cdf = get_cdf(uniform_pdf)
        result = cdf(-0.5)
        assert isinstance(result, (float, np.floating))
        assert result == 0.0

    def test_scalar_above_upper_bound(self, uniform_pdf):
        cdf = get_cdf(uniform_pdf)
        result = cdf(1.5)
        assert isinstance(result, (float, np.floating))
        assert result == 1.0


class TestIterableInputs:
    """Test various iterable input types."""

    def test_list_input(self, uniform_pdf):
        cdf = get_cdf(uniform_pdf)
        result = cdf([0.0, 0.5, 1.0])
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0], decimal=1)

    def test_numpy_array_input(self, uniform_pdf):
        cdf = get_cdf(uniform_pdf)
        result = cdf(np.array([0.0, 0.5, 1.0]))
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0], decimal=1)

    def test_tuple_input(self, uniform_pdf):
        cdf = get_cdf(uniform_pdf)
        result = cdf((0.0, 0.5, 1.0))
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0], decimal=1)

    def test_generator_input(self, uniform_pdf):
        cdf = get_cdf(uniform_pdf)
        gen = (x for x in [0.0, 0.5, 1.0])
        result = cdf(gen)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0], decimal=1)

    def test_singleton_iterable(self, uniform_pdf):
        cdf = get_cdf(uniform_pdf)
        result = cdf([0.5])
        assert isinstance(result, np.ndarray)
        assert len(result) == 1
        assert 0.45 < result[0] < 0.55

    def test_empty_iterable(self, uniform_pdf):
        cdf = get_cdf(uniform_pdf)
        result = cdf([])
        assert isinstance(result, np.ndarray)
        assert len(result) == 0


class TestScalarVectorAgreement:
    """Test that scalar and 1-element iterable give same result."""

    def test_scalar_vs_singleton_list(self, uniform_pdf):
        cdf = get_cdf(uniform_pdf)
        scalar_result = cdf(0.5)
        vector_result = cdf([0.5])
        assert np.isclose(scalar_result, vector_result[0], atol=1e-10)

    def test_scalar_vs_singleton_array(self, uniform_pdf):
        cdf = get_cdf(uniform_pdf)
        scalar_result = cdf(0.5)
        vector_result = cdf(np.array([0.5]))
        assert np.isclose(scalar_result, vector_result[0], atol=1e-10)

    def test_scalar_vs_singleton_tuple(self, uniform_pdf):
        cdf = get_cdf(uniform_pdf)
        scalar_result = cdf(0.25)
        vector_result = cdf((0.25,))
        assert np.isclose(scalar_result, vector_result[0], atol=1e-10)


class TestNonuniformDensity:
    """Test scalar inputs with nonuniform density."""

    def test_nonuniform_scalar_boundary(self, nonuniform_pdf):
        cdf = get_cdf(nonuniform_pdf)
        result = cdf(0.0)
        assert isinstance(result, (float, np.floating))
        assert result == 0.0

    def test_nonuniform_scalar_interior_peak(self, nonuniform_pdf):
        cdf = get_cdf(nonuniform_pdf)
        result = cdf(0.5)
        assert isinstance(result, (float, np.floating))
        assert 0.4 < result < 0.6

    def test_nonuniform_scalar_list(self, nonuniform_pdf):
        cdf = get_cdf(nonuniform_pdf)
        result = cdf([0.0, 0.5, 1.0])
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        assert result[0] == 0.0 and result[2] == 1.0
