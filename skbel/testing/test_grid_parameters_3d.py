#  Copyright (c) 2021. Robin Thibaut, Ghent University

import copy

import numpy as np
import pytest

from skbel.spatial import grid_parameters


def loop_oracle(x0, y0, z0, ncol, nrow, nlay, grf):
    """Explicit-loop midpoints in layer-major, row-major, column-major order."""
    out = []
    for k in range(nlay):
        for i in range(nrow):
            for j in range(ncol):
                out.append([x0 + (j + 0.5) * grf, y0 + (i + 0.5) * grf, z0 + (k + 0.5) * grf])
    return np.array(out)


def test_single_layer_baseline_2x2x1():
    xys, nrow, ncol, nlay = grid_parameters(x_lim=[0, 2], y_lim=[0, 2], z_lim=[0, 1], grf=1)
    assert (nrow, ncol, nlay) == (2, 2, 1)
    expected = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]])
    assert xys.shape == (4, 2)
    np.testing.assert_allclose(xys, expected)


def test_single_layer_default_z_with_origin():
    xys, nrow, ncol, nlay = grid_parameters(x_lim=[10, 12], y_lim=[-4, -2], grf=1)
    assert (nrow, ncol, nlay) == (2, 2, 1)
    expected = np.array([[10.5, -3.5], [11.5, -3.5], [10.5, -2.5], [11.5, -2.5]])
    np.testing.assert_allclose(xys, expected)


@pytest.mark.parametrize("nlay", [2, 3])
def test_multi_layer_unit_cells(nlay):
    xys, nrow, ncol, nl = grid_parameters(x_lim=[0, 2], y_lim=[0, 2], z_lim=[0, nlay], grf=1)
    assert (nrow, ncol, nl) == (2, 2, nlay)
    assert xys.shape == (nrow * ncol * nlay, 3)
    np.testing.assert_allclose(xys, loop_oracle(0, 0, 0, 2, 2, nlay, 1))


def test_multi_layer_ordering_explicit():
    xys, *_ = grid_parameters(x_lim=[0, 2], y_lim=[0, 2], z_lim=[0, 2], grf=1)
    expected = np.array(
        [
            [0.5, 0.5, 0.5],
            [1.5, 0.5, 0.5],
            [0.5, 1.5, 0.5],
            [1.5, 1.5, 0.5],
            [0.5, 0.5, 1.5],
            [1.5, 0.5, 1.5],
            [0.5, 1.5, 1.5],
            [1.5, 1.5, 1.5],
        ]
    )
    np.testing.assert_allclose(xys, expected)


@pytest.mark.parametrize(
    "x_lim, y_lim, z_lim",
    [
        ([5, 8], [-3, 0], [2, 4]),
        ([-10, -6], [-7, -5], [-3, 0]),
        ([-2, 1], [100, 102], [-1, 2]),
    ],
)
def test_multi_layer_nonzero_and_negative_origins(x_lim, y_lim, z_lim):
    xys, nrow, ncol, nlay = grid_parameters(x_lim=x_lim, y_lim=y_lim, z_lim=z_lim, grf=1)
    assert (ncol, nrow, nlay) == (x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0])
    np.testing.assert_allclose(
        xys, loop_oracle(min(x_lim), min(y_lim), min(z_lim), ncol, nrow, nlay, 1)
    )


def test_multi_layer_unequal_axis_counts():
    xys, nrow, ncol, nlay = grid_parameters(x_lim=[0, 4], y_lim=[0, 2], z_lim=[0, 3], grf=1)
    assert (nrow, ncol, nlay) == (2, 4, 3)
    assert len(xys) == 2 * 4 * 3
    np.testing.assert_allclose(xys, loop_oracle(0, 0, 0, 4, 2, 3, 1))


def test_multi_layer_nonunit_cell_size():
    grf = 2.5
    xys, nrow, ncol, nlay = grid_parameters(x_lim=[1, 11], y_lim=[-5, 0], z_lim=[3, 10.5], grf=grf)
    assert (nrow, ncol, nlay) == (2, 4, 3)
    assert xys.shape == (nrow * ncol * nlay, 3)
    np.testing.assert_allclose(xys, loop_oracle(1, -5, 3, ncol, nrow, nlay, grf))


def test_inputs_not_mutated():
    x_lim, y_lim, z_lim = [-1, 3], [0, 2], [4, 6]
    before = copy.deepcopy((x_lim, y_lim, z_lim))
    grid_parameters(x_lim=x_lim, y_lim=y_lim, z_lim=z_lim, grf=1)
    assert (x_lim, y_lim, z_lim) == before
    x_arr, y_arr, z_arr = np.array([-1, 3]), np.array([0, 2]), np.array([4, 6])
    snapshot = [a.copy() for a in (x_arr, y_arr, z_arr)]
    grid_parameters(x_lim=x_arr, y_lim=y_arr, z_lim=z_arr, grf=1)
    for a, s in zip((x_arr, y_arr, z_arr), snapshot, strict=True):
        np.testing.assert_array_equal(a, s)
