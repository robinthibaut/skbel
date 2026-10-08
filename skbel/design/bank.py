#  Copyright (c) 2026. Robin Thibaut, Ghent University

"""Shared simulation bank and exact sensor/cadence/window selections.

A :class:`SimulationBank` holds one set of simulated observations, shaped
``(rows, time, sensor)``, and the targets that generated them, shaped
``(rows, targets)``. Row ``i`` of the observations and row ``i`` of the
targets belong to the same simulation; every operation here keeps that
pairing.

A :class:`DesignSelection` names a subset of sensors and native time indices
of that bank. Applying it to observations returns a flat feature matrix in a
documented order, so the same selection can be applied to the bank, to
held-out simulations and to observed records on the same native grid.

Conventions:

- Shapes are explicit and checked exactly; mismatches raise ``ValueError``.
  Non-numeric, boolean or complex arrays raise ``TypeError``.
- Values must be finite; there is no clipping, interpolation or NaN hiding.
- Indices are non-negative integers in range. Booleans, floats (even
  integral ones), duplicates and negative indices are rejected.
- Time is addressed by native sample index. Selections never resample,
  interpolate, repeat or reorder time samples.

NumPy only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np

__all__ = ["DesignSelection", "SimulationBank", "cadence_window"]

# Relative tolerance on time steps when deciding whether ``times`` is regular.
_REGULAR_RTOL = 1e-9


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_int(value, name: str, *, minimum: int | None = None) -> int:
    """Return ``value`` as a Python int; booleans and non-integers are rejected."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _index_vector(values, size: int, name: str) -> np.ndarray:
    """Validated one-dimensional vector of unique integer indices in ``[0, size)``.

    The caller's order is kept. Empty, boolean, fractional, duplicate,
    negative and out-of-range entries raise.
    """
    if isinstance(values, (str, bytes)) or not isinstance(values, (Sequence, np.ndarray)):
        raise TypeError(f"{name} must be a one-dimensional sequence of integers")
    if not isinstance(values, np.ndarray) and any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in values
    ):
        raise TypeError(f"{name} must contain integers only (no booleans or floats)")
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got ndim={arr.ndim}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{name} must have an integer dtype, got {arr.dtype}")
    arr = arr.astype(np.int64)
    if np.any(arr < 0) or np.any(arr >= size):
        raise ValueError(f"{name} entries must lie in [0, {size}), got {arr.tolist()}")
    if np.unique(arr).shape[0] != arr.shape[0]:
        raise ValueError(f"{name} must not contain duplicates, got {arr.tolist()}")
    return arr


def _finite_float_array(values, name: str, ndim: int) -> np.ndarray:
    """Copy of ``values`` as float64 with exactly ``ndim`` non-empty, finite axes."""
    arr = np.asarray(values)
    if arr.dtype == np.bool_ or not (
        np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)
    ):
        raise TypeError(f"{name} must be a real numeric array, got dtype {arr.dtype}")
    if arr.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got shape {arr.shape}")
    if 0 in arr.shape:
        raise ValueError(f"{name} must have no empty axes, got shape {arr.shape}")
    arr = np.array(arr, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite (no NaN or infinity)")
    return arr


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DesignSelection:
    """Exact sensor and time-index subset of a ``(rows, n_time, n_sensors)`` grid.

    :param n_time: native number of time samples the selection refers to.
    :param n_sensors: native number of sensors the selection refers to.
    :param sensors: native sensor positions, in the order the features use.
        Unique, non-negative and below ``n_sensors``; not sorted.
    :param time_indices: native time indices, strictly increasing and below
        ``n_time``. They are never interpolated, repeated or reordered.

    Feature order is time-major: feature ``k`` is the value at native time
    ``time_indices[k // len(sensors)]`` and native sensor
    ``sensors[k % len(sensors)]`` (see :attr:`feature_index`).
    """

    n_time: int
    n_sensors: int
    sensors: tuple[int, ...]
    time_indices: tuple[int, ...]

    def __post_init__(self):
        n_time = _check_int(self.n_time, "n_time", minimum=1)
        n_sensors = _check_int(self.n_sensors, "n_sensors", minimum=1)
        sensors = _index_vector(self.sensors, n_sensors, "sensors")
        times = _index_vector(self.time_indices, n_time, "time_indices")
        if times.shape[0] > 1 and np.any(np.diff(times) <= 0):
            raise ValueError(f"time_indices must be strictly increasing, got {times.tolist()}")
        object.__setattr__(self, "n_time", n_time)
        object.__setattr__(self, "n_sensors", n_sensors)
        object.__setattr__(self, "sensors", tuple(int(s) for s in sensors))
        object.__setattr__(self, "time_indices", tuple(int(t) for t in times))

    @property
    def n_features(self) -> int:
        """Number of features, ``len(time_indices) * len(sensors)``."""
        return len(self.time_indices) * len(self.sensors)

    @property
    def feature_index(self) -> np.ndarray:
        """Native ``(time_index, sensor)`` of each feature, shape ``(n_features, 2)``."""
        t = np.repeat(np.asarray(self.time_indices, dtype=np.int64), len(self.sensors))
        s = np.tile(np.asarray(self.sensors, dtype=np.int64), len(self.time_indices))
        return np.column_stack([t, s])

    def apply(self, observations) -> np.ndarray:
        """Select and flatten observations on the native grid.

        :param observations: array shaped exactly ``(cases, n_time, n_sensors)``,
            finite and real. A single record must be passed as ``(1, n_time,
            n_sensors)``; nothing is broadcast.
        :return: new float64 array of shape ``(cases, n_features)`` in
            :attr:`feature_index` order. The input is not modified.
        """
        arr = _finite_float_array(observations, "observations", 3)
        expected = (self.n_time, self.n_sensors)
        if arr.shape[1:] != expected:
            raise ValueError(
                f"observations must have shape (cases, {expected[0]}, {expected[1]}), "
                f"got {arr.shape}"
            )
        picked = arr[:, np.asarray(self.time_indices)][:, :, np.asarray(self.sensors)]
        return picked.reshape(arr.shape[0], self.n_features)


def cadence_window(
    n_time: int,
    n_sensors: int,
    sensors,
    *,
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    anchor: int | None = None,
) -> DesignSelection:
    """Selection of every ``step``-th native sample in the half-open window ``[start, stop)``.

    The selected time indices are the integers ``i`` with ``start <= i < stop``
    and ``(i - anchor) % step == 0``. With the default ``anchor=start`` the
    first sample is ``start``; ``anchor=0`` aligns samples to multiples of
    ``step`` on the native index, so windows starting at different offsets
    share the same sampling phase.

    Indices count native samples. On a regular grid with spacing ``dt`` the
    cadence is ``step * dt`` and the window spans ``(stop - start) * dt``. This
    function knows nothing about physical time; see
    :meth:`SimulationBank.select` for the check on irregular timestamps.

    :param n_time: native number of time samples.
    :param n_sensors: native number of sensors.
    :param sensors: native sensor positions, in feature order.
    :param start: first index of the window (inclusive), ``0 <= start < n_time``.
    :param stop: end of the window (exclusive), ``start < stop <= n_time``;
        ``None`` means ``n_time``.
    :param step: positive integer cadence in native samples.
    :param anchor: integer index setting the sampling phase; ``None`` means
        ``start``. It may lie outside the window.
    :return: a :class:`DesignSelection`.
    :raises ValueError: if the window is empty, out of range, or contains no
        sample of the requested phase.
    """
    n_time = _check_int(n_time, "n_time", minimum=1)
    n_sensors = _check_int(n_sensors, "n_sensors", minimum=1)
    start = _check_int(start, "start", minimum=0)
    stop = n_time if stop is None else _check_int(stop, "stop")
    step = _check_int(step, "step", minimum=1)
    anchor = start if anchor is None else _check_int(anchor, "anchor")
    if start >= n_time:
        raise ValueError(f"start={start} must be below n_time={n_time}")
    if not start < stop <= n_time:
        raise ValueError(
            f"window [start, stop) = [{start}, {stop}) must satisfy 0 <= start < stop <= {n_time}"
        )
    first = start + (anchor - start) % step
    indices = np.arange(first, stop, step, dtype=np.int64)
    if indices.shape[0] == 0:
        raise ValueError(
            f"window [{start}, {stop}) contains no sample with step={step} and anchor={anchor}"
        )
    return DesignSelection(
        n_time=n_time,
        n_sensors=n_sensors,
        sensors=tuple(_index_vector(sensors, n_sensors, "sensors").tolist()),
        time_indices=tuple(indices.tolist()),
    )


# ---------------------------------------------------------------------------
# Bank
# ---------------------------------------------------------------------------


class SimulationBank:
    """Paired simulated observations and targets, shared by many designs.

    :param observations: real, finite array shaped ``(rows, n_time, n_sensors)``.
    :param targets: real, finite array shaped ``(rows, n_targets)``; row ``i``
        is the target that produced ``observations[i]``.
    :param times: optional one-dimensional, finite, strictly increasing array of
        length ``n_time`` giving the physical time of each native sample.
    :param sensor_ids: optional sequence of ``n_sensors`` unique labels (for
        example depths or names) for :meth:`sensor_positions`.

    Both arrays are copied to float64, so later changes to the caller's arrays
    do not affect the bank. The stored arrays have NumPy's ``writeable`` flag
    cleared, which catches accidental in-place assignment through the
    attributes. This is read-only by convention, not immutability: a caller
    can set the flag back and change the bank's storage, and nothing records
    or detects that. Callers that share a bank are responsible for not
    mutating it.
    """

    def __init__(self, observations, targets, *, times=None, sensor_ids=None):
        obs = _finite_float_array(observations, "observations", 3)
        tgt = _finite_float_array(targets, "targets", 2)
        if tgt.shape[0] != obs.shape[0]:
            raise ValueError(
                "observations and targets must have the same number of rows, "
                f"got {obs.shape[0]} and {tgt.shape[0]}"
            )
        obs.flags.writeable = False
        tgt.flags.writeable = False
        self._observations = obs
        self._targets = tgt

        n_time, n_sensors = obs.shape[1], obs.shape[2]
        if times is None:
            self._times = None
        else:
            t = _finite_float_array(times, "times", 1)
            if t.shape[0] != n_time:
                raise ValueError(f"times must have length {n_time}, got {t.shape[0]}")
            if t.shape[0] > 1 and np.any(np.diff(t) <= 0):
                raise ValueError("times must be strictly increasing")
            t.flags.writeable = False
            self._times = t

        if sensor_ids is None:
            self._sensor_ids = None
        else:
            if isinstance(sensor_ids, (str, bytes)) or not isinstance(
                sensor_ids, (Sequence, np.ndarray)
            ):
                raise TypeError("sensor_ids must be a sequence of labels")
            ids = list(sensor_ids.tolist() if isinstance(sensor_ids, np.ndarray) else sensor_ids)
            if len(ids) != n_sensors:
                raise ValueError(f"sensor_ids must have length {n_sensors}, got {len(ids)}")
            if len(set(ids)) != n_sensors:
                raise ValueError("sensor_ids must be unique")
            self._sensor_ids = tuple(ids)

    # -- shape -------------------------------------------------------------

    @property
    def observations(self) -> np.ndarray:
        """Read-only observations, shape ``(rows, n_time, n_sensors)``."""
        return self._observations

    @property
    def targets(self) -> np.ndarray:
        """Read-only targets, shape ``(rows, n_targets)``."""
        return self._targets

    @property
    def times(self) -> np.ndarray | None:
        """Read-only physical time of each native sample, or ``None``."""
        return self._times

    @property
    def sensor_ids(self) -> tuple | None:
        """Sensor labels in native order, or ``None``."""
        return self._sensor_ids

    @property
    def n_rows(self) -> int:
        """Number of paired simulations."""
        return self._observations.shape[0]

    @property
    def n_time(self) -> int:
        """Native number of time samples."""
        return self._observations.shape[1]

    @property
    def n_sensors(self) -> int:
        """Native number of sensors."""
        return self._observations.shape[2]

    @property
    def n_targets(self) -> int:
        """Number of target columns."""
        return self._targets.shape[1]

    @property
    def is_regular(self) -> bool:
        """Whether native samples are equally spaced in time.

        ``True`` when ``times`` is ``None`` (only indices are known) or has a
        single sample, or when every time step equals the first one within a
        relative tolerance of ``1e-9``.
        """
        if self._times is None or self._times.shape[0] < 2:
            return True
        dt = np.diff(self._times)
        return bool(np.all(np.abs(dt - dt[0]) <= _REGULAR_RTOL * dt[0]))

    # -- selection ---------------------------------------------------------

    def sensor_positions(self, labels) -> tuple[int, ...]:
        """Native positions of the given sensor labels, in the order given.

        :raises ValueError: if the bank has no ``sensor_ids``, a label is
            unknown, or a label is repeated.
        """
        if self._sensor_ids is None:
            raise ValueError("this bank has no sensor_ids; select sensors by position")
        if isinstance(labels, (str, bytes)) or not isinstance(labels, (Sequence, np.ndarray)):
            raise TypeError("labels must be a sequence")
        labels = list(labels.tolist() if isinstance(labels, np.ndarray) else labels)
        if not labels:
            raise ValueError("labels must not be empty")
        if len(set(labels)) != len(labels):
            raise ValueError("labels must not contain duplicates")
        lookup = {label: i for i, label in enumerate(self._sensor_ids)}
        missing = [label for label in labels if label not in lookup]
        if missing:
            raise ValueError(f"unknown sensor labels: {missing}")
        return tuple(lookup[label] for label in labels)

    def select(
        self,
        sensors,
        *,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
        anchor: int | None = None,
        time_indices=None,
    ) -> DesignSelection:
        """Selection on this bank's native grid.

        Either a cadence and window (``start``, ``stop``, ``step``, ``anchor``;
        see :func:`cadence_window`) or explicit ``time_indices`` (strictly
        increasing native indices), not both.

        When the bank has irregular ``times``, a cadence ``step > 1`` would not
        be a fixed physical cadence, so it is refused; pass explicit
        ``time_indices`` instead. Windows with ``step=1`` and explicit indices
        are exact native samples and are always allowed.

        The window arguments are type-checked in both cases, so a boolean or
        float ``start``, ``stop``, ``step`` or ``anchor`` raises ``TypeError``
        even alongside ``time_indices``.
        """
        start = _check_int(start, "start", minimum=0)
        stop = None if stop is None else _check_int(stop, "stop")
        step = _check_int(step, "step", minimum=1)
        anchor = None if anchor is None else _check_int(anchor, "anchor")
        if time_indices is not None:
            if (start, stop, step, anchor) != (0, None, 1, None):
                raise ValueError("pass either time_indices or start/stop/step/anchor, not both")
            return DesignSelection(
                n_time=self.n_time,
                n_sensors=self.n_sensors,
                sensors=tuple(_index_vector(sensors, self.n_sensors, "sensors").tolist()),
                time_indices=tuple(
                    _index_vector(time_indices, self.n_time, "time_indices").tolist()
                ),
            )
        if step > 1:
            if not self.is_regular:
                raise ValueError(
                    "times are irregular, so a cadence in native samples is not a fixed "
                    "physical cadence; pass explicit time_indices instead"
                )
        return cadence_window(
            self.n_time,
            self.n_sensors,
            sensors,
            start=start,
            stop=stop,
            step=step,
            anchor=anchor,
        )

    def _check_selection(self, selection: DesignSelection) -> None:
        if not isinstance(selection, DesignSelection):
            raise TypeError(f"selection must be a DesignSelection, got {type(selection).__name__}")
        if (selection.n_time, selection.n_sensors) != (self.n_time, self.n_sensors):
            raise ValueError(
                f"selection grid ({selection.n_time}, {selection.n_sensors}) does not match "
                f"bank grid ({self.n_time}, {self.n_sensors})"
            )

    def rows(self, rows) -> np.ndarray:
        """Validated row indices: unique, non-negative, in range, caller order kept."""
        return _index_vector(rows, self.n_rows, "rows")

    def features(self, selection: DesignSelection, rows) -> tuple[np.ndarray, np.ndarray]:
        """Paired ``(X, Y)`` for the given rows under a selection.

        :param selection: a :class:`DesignSelection` on this bank's grid.
        :param rows: explicit row indices (see :meth:`rows`). There is no
            default: training and held-out rows are always named by the caller.
        :return: new arrays ``X`` of shape ``(len(rows), selection.n_features)``
            and ``Y`` of shape ``(len(rows), n_targets)``; ``X[k]`` and ``Y[k]``
            both come from bank row ``rows[k]``.
        """
        self._check_selection(selection)
        idx = self.rows(rows)
        X = selection.apply(self._observations[idx])
        Y = np.array(self._targets[idx], dtype=np.float64, copy=True)
        return X, Y
