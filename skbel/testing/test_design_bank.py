#  Copyright (c) 2026. Robin Thibaut, Ghent University

"""Tests for the shared simulation bank and exact design selections."""

import numpy as np
import pytest

from skbel.design import DesignSelection, SimulationBank, cadence_window


def _bank_arrays(n_rows=6, n_time=10, n_sensors=4):
    # Each value encodes its own (row, time, sensor) so selections can be checked exactly.
    r, t, s = np.meshgrid(np.arange(n_rows), np.arange(n_time), np.arange(n_sensors), indexing="ij")
    obs = 1000.0 * r + 10.0 * t + s
    targets = np.column_stack([np.arange(n_rows), -np.arange(n_rows)]).astype(float)
    return obs, targets


# ---------------------------------------------------------------------------
# Bank construction
# ---------------------------------------------------------------------------


def test_bank_copies_and_is_read_only():
    obs, targets = _bank_arrays()
    obs_before, targets_before = obs.copy(), targets.copy()
    bank = SimulationBank(obs, targets)
    obs[0, 0, 0] = -1.0
    targets[0, 0] = -1.0
    assert bank.observations[0, 0, 0] == obs_before[0, 0, 0]
    assert bank.targets[0, 0] == targets_before[0, 0]
    with pytest.raises(ValueError):
        bank.observations[0, 0, 0] = 5.0
    with pytest.raises(ValueError):
        bank.targets[0, 0] = 5.0
    assert (bank.n_rows, bank.n_time, bank.n_sensors, bank.n_targets) == (6, 10, 4, 2)


@pytest.mark.parametrize(
    "obs_shape, targets_shape",
    [
        ((6, 10), (6, 2)),  # observations not 3-D
        ((6, 10, 4), (6,)),  # targets not 2-D
        ((6, 10, 4), (5, 2)),  # unpaired rows
        ((0, 10, 4), (0, 2)),  # empty rows
        ((6, 0, 4), (6, 2)),  # empty time
        ((6, 10, 4), (6, 0)),  # empty targets
    ],
)
def test_bank_rejects_bad_shapes(obs_shape, targets_shape):
    with pytest.raises(ValueError):
        SimulationBank(np.zeros(obs_shape), np.zeros(targets_shape))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_bank_rejects_non_finite(bad):
    obs, targets = _bank_arrays()
    obs[1, 2, 3] = bad
    with pytest.raises(ValueError, match="finite"):
        SimulationBank(obs, targets)
    obs, targets = _bank_arrays()
    targets[2, 1] = bad
    with pytest.raises(ValueError, match="finite"):
        SimulationBank(obs, targets)


def test_bank_rejects_non_numeric():
    obs, targets = _bank_arrays()
    with pytest.raises(TypeError):
        SimulationBank(obs > 0, targets)
    with pytest.raises(TypeError):
        SimulationBank(obs.astype(complex), targets)
    with pytest.raises(TypeError):
        SimulationBank(obs, targets.astype(object))


def test_bank_times_and_sensor_ids_validation():
    obs, targets = _bank_arrays()
    SimulationBank(obs, targets, times=np.arange(10) * 900.0, sensor_ids=[0.1, 0.2, 0.3, 0.5])
    with pytest.raises(ValueError, match="length"):
        SimulationBank(obs, targets, times=np.arange(9.0))
    with pytest.raises(ValueError, match="increasing"):
        SimulationBank(obs, targets, times=np.r_[0.0, 2.0, 1.0, np.arange(3.0, 10.0)])
    with pytest.raises(ValueError, match="length"):
        SimulationBank(obs, targets, sensor_ids=["a", "b"])
    with pytest.raises(ValueError, match="unique"):
        SimulationBank(obs, targets, sensor_ids=["a", "b", "b", "c"])


# ---------------------------------------------------------------------------
# Selections
# ---------------------------------------------------------------------------


def test_feature_order_is_time_major_with_caller_sensor_order():
    obs, targets = _bank_arrays()
    bank = SimulationBank(obs, targets)
    sel = bank.select([3, 0, 2], time_indices=[1, 4, 7])
    X, Y = bank.features(sel, rows=[4, 1])
    np.testing.assert_array_equal(
        sel.feature_index, [[1, 3], [1, 0], [1, 2], [4, 3], [4, 0], [4, 2], [7, 3], [7, 0], [7, 2]]
    )
    for k, row in enumerate([4, 1]):
        expected = [1000.0 * row + 10.0 * t + s for t, s in sel.feature_index]
        np.testing.assert_array_equal(X[k], expected)
        np.testing.assert_array_equal(Y[k], targets[row])
    assert X.shape == (2, sel.n_features) == (2, 9)


def test_cadence_window_half_open_and_alignment():
    assert cadence_window(20, 3, [0], start=2, stop=11, step=3).time_indices == (2, 5, 8)
    # stop is exclusive, even when it falls on the cadence
    assert cadence_window(20, 3, [0], start=2, stop=8, step=3).time_indices == (2, 5)
    # anchor=0 aligns samples to multiples of step regardless of start
    assert cadence_window(20, 3, [0], start=2, stop=11, step=3, anchor=0).time_indices == (3, 6, 9)
    # anchor outside the window sets only the phase
    assert cadence_window(20, 3, [0], start=5, stop=12, step=4, anchor=-1).time_indices == (7, 11)
    # defaults: whole record at native cadence
    assert cadence_window(5, 2, [1, 0]).time_indices == (0, 1, 2, 3, 4)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"start": 5, "stop": 5}, ValueError),  # empty window
        ({"start": 6, "stop": 5}, ValueError),  # reversed window
        ({"start": 0, "stop": 21}, ValueError),  # beyond record
        ({"start": 20}, ValueError),  # start at end
        ({"start": -1}, ValueError),  # negative start
        ({"step": 0}, ValueError),
        ({"step": -2}, ValueError),
        ({"step": 1.5}, TypeError),  # fractional cadence
        ({"step": 2.0}, TypeError),  # float cadence, even if integral
        ({"step": True}, TypeError),  # boolean cadence
        ({"start": 1.0}, TypeError),
        ({"stop": np.float64(10)}, TypeError),
        ({"start": 2, "stop": 4, "step": 5, "anchor": 0}, ValueError),  # no sample of that phase
    ],
)
def test_cadence_window_rejects_impossible(kwargs, error):
    with pytest.raises(error):
        cadence_window(20, 3, [0], **kwargs)


@pytest.mark.parametrize(
    "sensors, error",
    [
        ([], ValueError),
        ([0, 0], ValueError),  # duplicate
        ([4], ValueError),  # out of range
        ([-1], ValueError),  # negative
        ([True, False], TypeError),  # boolean list
        (np.array([True, False, True, False]), TypeError),  # boolean mask
        ([0, True], TypeError),  # mixed boolean
        ([0.0, 1.0], TypeError),  # float, even if integral
        ([0.5], TypeError),
        (2, TypeError),  # scalar
        ("01", TypeError),
        ([[0, 1]], TypeError),  # nested
        (np.array([[0, 1]]), ValueError),
    ],
)
def test_selection_rejects_bad_sensors(sensors, error):
    with pytest.raises(error):
        cadence_window(10, 4, sensors)


@pytest.mark.parametrize(
    "time_indices, error",
    [
        ([], ValueError),
        ([1, 1], ValueError),  # duplicated time
        ([3, 1], ValueError),  # reordered time
        ([0, 10], ValueError),  # out of range
        ([-1, 2], ValueError),
        ([0.0, 1.0], TypeError),
        (np.array([True] * 10), TypeError),
    ],
)
def test_selection_rejects_bad_time_indices(time_indices, error):
    bank = SimulationBank(*_bank_arrays())
    with pytest.raises(error):
        bank.select([0], time_indices=time_indices)


def test_select_rejects_mixed_window_and_indices():
    bank = SimulationBank(*_bank_arrays())
    with pytest.raises(ValueError, match="not both"):
        bank.select([0], step=2, time_indices=[0, 2])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"step": True},
        {"step": np.True_},
        {"step": 1.0},
        {"step": np.float64(1)},
        {"start": False},
        {"start": 0.0},
        {"stop": 10.0},
        {"anchor": 0.0},
        {"anchor": False},
    ],
)
def test_select_rejects_non_integer_window_args_with_time_indices(kwargs):
    # These compare equal to the defaults, so they must be rejected by type, not value.
    bank = SimulationBank(*_bank_arrays())
    with pytest.raises(TypeError):
        bank.select([0], time_indices=[0, 2], **kwargs)
    with pytest.raises(TypeError):
        bank.select([0], **kwargs)


def test_select_with_time_indices_accepts_explicit_integer_defaults():
    bank = SimulationBank(*_bank_arrays())
    expected = bank.select([1, 0], time_indices=[0, 2, 7])
    for kwargs in ({"start": 0, "step": 1}, {"start": np.int64(0), "step": np.int32(1)}):
        sel = bank.select([1, 0], time_indices=[0, 2, 7], **kwargs)
        assert sel == expected
        assert sel.time_indices == (0, 2, 7)
        assert sel.sensors == (1, 0)


def test_selection_is_frozen_and_validated_directly():
    sel = DesignSelection(n_time=10, n_sensors=4, sensors=(2, 1), time_indices=(0, 3))
    with pytest.raises(AttributeError):
        sel.sensors = (0,)
    with pytest.raises(ValueError, match="strictly increasing"):
        DesignSelection(n_time=10, n_sensors=4, sensors=(0,), time_indices=(3, 0))
    with pytest.raises(TypeError):
        DesignSelection(n_time=10.0, n_sensors=4, sensors=(0,), time_indices=(0,))


def test_apply_rejects_mismatched_grid_and_broadcasting():
    obs, targets = _bank_arrays()
    bank = SimulationBank(obs, targets)
    sel = bank.select([0, 1], start=0, stop=10, step=2)
    with pytest.raises(ValueError, match="3 dimensions"):
        sel.apply(obs[0])  # a single record must keep its case axis
    with pytest.raises(ValueError, match="shape"):
        sel.apply(obs[:, :9])  # different native time length
    with pytest.raises(ValueError, match="shape"):
        sel.apply(obs[:, :, :3])  # different native sensor count
    other = cadence_window(9, 4, [0], step=2)
    with pytest.raises(ValueError, match="grid"):
        bank.features(other, rows=[0])
    bad = obs[:2].copy()
    bad[0, 4, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        sel.apply(bad)


def test_apply_does_not_modify_input_and_matches_bank_rows():
    obs, targets = _bank_arrays()
    bank = SimulationBank(obs, targets)
    sel = bank.select([2, 3], start=1, stop=9, step=3)
    record = obs[3:4].copy()
    before = record.copy()
    out = sel.apply(record)
    out[:] = 0.0
    np.testing.assert_array_equal(record, before)
    X, _ = bank.features(sel, rows=[3])
    np.testing.assert_array_equal(sel.apply(obs[3:4]), X)


@pytest.mark.parametrize(
    "rows, error",
    [
        ([], ValueError),
        ([0, 0], ValueError),
        ([6], ValueError),
        ([-1], ValueError),
        ([True, False, True, False, True, False], TypeError),
        ([0.0], TypeError),
    ],
)
def test_features_reject_bad_rows(rows, error):
    bank = SimulationBank(*_bank_arrays())
    sel = bank.select([0])
    with pytest.raises(error):
        bank.features(sel, rows=rows)


def test_irregular_times_refuse_sample_cadence_but_allow_exact_indices():
    obs, targets = _bank_arrays()
    times = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    bank = SimulationBank(obs, targets, times=times)
    assert not bank.is_regular
    with pytest.raises(ValueError, match="irregular"):
        bank.select([0], step=2)
    sel = bank.select([0], time_indices=[0, 2, 4])
    np.testing.assert_array_equal(bank.times[list(sel.time_indices)], [0.0, 2.0, 5.0])
    assert bank.select([0], start=3, stop=6).time_indices == (3, 4, 5)

    regular = SimulationBank(obs, targets, times=np.arange(10) * 0.1)
    assert regular.is_regular
    assert regular.select([0], step=3).time_indices == (0, 3, 6, 9)


def test_sensor_positions_by_label():
    obs, targets = _bank_arrays()
    bank = SimulationBank(obs, targets, sensor_ids=["top", "a", "b", "bottom"])
    assert bank.sensor_positions(["bottom", "top"]) == (3, 0)
    with pytest.raises(ValueError, match="unknown"):
        bank.sensor_positions(["middle"])
    with pytest.raises(ValueError, match="duplicates"):
        bank.sensor_positions(["a", "a"])
    with pytest.raises(ValueError, match="no sensor_ids"):
        SimulationBank(obs, targets).sensor_positions(["a"])
