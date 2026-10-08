# Comparing designs on a shared simulation bank

`skbel.design` compares measurement designs, meaning which sensors are read,
how often and over which period, using one shared set of simulations. Each
design selects a subset of the simulated observations, and a fresh model is
fitted for each design. The functions are generic: they work on any array of
simulated time series and do not depend on the physics that produced them.

The module provides selection and refitting only. It does not score designs,
choose a best design or simulate data. A posterior fitted on simulations
describes the simulated prior; whether it is calibrated for real records has
to be checked separately, for example with {doc}`calibration`.

## The bank

```python
from skbel.design import SimulationBank

bank = SimulationBank(observations, targets, times=times, sensor_ids=depths)
```

| Argument | Shape | Meaning |
| --- | --- | --- |
| `observations` | `(rows, n_time, n_sensors)` | simulated record of every sensor at every native time sample |
| `targets` | `(rows, n_targets)` | quantity to predict; row `i` produced `observations[i]` |
| `times` | `(n_time,)` or `None` | optional physical time of each native sample, strictly increasing |
| `sensor_ids` | `n_sensors` labels or `None` | optional unique labels, such as depths |

Both arrays must be real and finite with no empty axis, and must have the same
number of rows. They are copied to float64, so later changes to the caller's
arrays do not reach the bank, and the caller's arrays are never modified.

The stored arrays have NumPy's `writeable` flag cleared, which turns accidental
in-place assignment such as `bank.observations[0] = 0` into an error. This is
read-only by convention, not immutability: code that sets the flag back can
change the bank's storage, and the bank neither records nor detects it. When
a bank is shared between designs, the caller is responsible for not mutating
it.

## Selecting a design

A {class}`~skbel.design.bank.DesignSelection` lists native sensor positions and
native time indices. It never interpolates, repeats or reorders time samples.

```python
# sensors 0 and 3, every 4th sample, in the half-open window [96, 384)
sel = bank.select([0, 3], start=96, stop=384, step=4)

# the same sensors chosen by label
sel = bank.select(bank.sensor_positions([0.1, 0.5]), start=96, stop=384, step=4)

# explicit native indices
sel = bank.select([0, 3], time_indices=[96, 100, 120, 200])
```

Time semantics:

- Indices count native samples, from `0` to `n_time - 1`.
- The window `[start, stop)` is half-open: `start` is included and `stop` is
  not. `stop=None` means the end of the record.
- With cadence `step`, the selected indices are the `i` in the window with
  `(i - anchor) % step == 0`. The default `anchor=start` starts at `start`;
  `anchor=0` aligns every window to multiples of `step`, so windows with
  different starts share the same sampling phase.
- On a regular grid with spacing `dt`, the cadence is `step * dt` and the
  window spans `(stop - start) * dt`.
- If the bank has irregular `times`, a cadence `step > 1` is refused, because
  every `step`-th sample would not be a fixed physical cadence. Windows with
  `step=1` and explicit `time_indices` are still allowed: they select exact
  native samples. Spacing counts as regular if every step equals the first
  within a relative tolerance of `1e-9`.

Selections are rejected if they are empty, contain duplicates, use booleans or
floats (including integral floats such as `2.0`), fall out of range, have a
reversed or empty window, or have a window that contains no sample of the
requested phase. The window arguments `start`, `stop`, `step` and `anchor` are
type-checked even when explicit `time_indices` are given, so
`bank.select([0], time_indices=[0, 2], step=1.0)` raises instead of being
treated as the default. Sensors may be given in any order, and that order is
kept.

### Feature order

{meth}`DesignSelection.apply <skbel.design.bank.DesignSelection.apply>` turns an
array shaped exactly `(cases, n_time, n_sensors)` into features shaped
`(cases, len(time_indices) * len(sensors))`. The order is time-major: feature
`k` is native time `time_indices[k // len(sensors)]` at native sensor
`sensors[k % len(sensors)]`.
{attr}`~skbel.design.bank.DesignSelection.feature_index` lists these pairs. A
single record must be passed with its case axis, as `(1, n_time, n_sensors)`;
nothing is broadcast.

{meth}`SimulationBank.features(sel, rows) <skbel.design.bank.SimulationBank.features>`
returns the paired `(X, Y)` for explicit bank rows. `X[k]` and `Y[k]` both
come from bank row `rows[k]`. There is no default row set.

## Fresh refits

```python
from skbel import BEL
from skbel.design import fit_design, fit_designs

refit = fit_design(template, bank, sel, train_rows=train_rows)
x_obs = refit.features(observed)  # same mask, same order as training
samples = refit.model.predict(x_obs, n_posts=500)
```

`fit_design` clones the unfitted `template`, for example a {class}`~skbel.BEL`
with its pre-processing, regression and post-processing pipelines, using
{func}`sklearn.base.clone`. It fits that clone on `bank.features(sel,
train_rows)` only. Consequences:

- The template is never fitted or modified, and every refit gets new,
  unfitted copies of every nested object. No scaler, PCA or CCA is shared
  between designs or between refits of the same design.
- Fitted state attached to the template, such as cached pre-processed arrays,
  is not carried into the clone.
- `train_rows` is required. Held-out rows, observed records and their target
  values are used only if the caller lists them.
- The result holds the fitted model, the selection and the training rows.
  {meth}`DesignRefit.features <skbel.design.refit.DesignRefit.features>` applies the
  same selection to held-out simulations or observed records on the native
  grid.

`fit_designs(template, bank, selections, train_rows=...)` does the same for
several selections, with one fresh clone each.

Each refit costs a full model fit. Nothing is cached between refits, so
comparing many designs costs that many fits.
