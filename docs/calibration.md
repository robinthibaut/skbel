# Calibration checks for posterior draws

`skbel.metrics.calibration` checks whether posterior draws are calibrated
against known true values. It works on arrays from any sampler, not only
`BEL`: draws have shape `(cases, draws, targets)` and the true values have
shape `(cases, targets)`. Typical cases are simulations whose true parameters
are known, or held-out records with an independently known quantity.

Every check is marginal: each target is assessed on its own. Passing every
marginal check does not establish joint (multivariate) calibration. A test
result describes the supplied cases only; it does not by itself validate a
model for new conditions.

## Simulation-based calibration ranks

`sbc_ranks(samples, truth, seed=...)` counts, for each case and target, the
draws strictly below the truth. Draws exactly equal to the truth are tied; with
`ties="randomize"` the truth takes a uniformly random position among them, so
the rank lies in `{0, ..., M}` for `M` draws. `ties="error"` instead rejects
any exact tie and needs no seed.

The ranks are uniform on `{0, ..., M}` when the truth and the draws are
exchangeable, for example when the truth is drawn from the prior, data are
simulated from it, and the `M` draws are independent exact posterior draws for
those data. Autocorrelated draws such as raw MCMC output are not exchangeable
with the truth. Thinning reduces autocorrelation but does not by itself make
the draws exchangeable. The function cannot check this condition.
Weighted draws are not accepted, because weighted ranks are not SBC ranks.

`sbc_rank_histogram(ranks, n_draws, n_bins)` counts ranks in `n_bins` equal
bins, which must divide `M + 1`. Under uniform ranks and independent cases,
each count is Binomial with `cases` trials and probability `1 / n_bins`.
Overfull edge bins indicate a posterior that is too narrow or biased; an
overfull centre indicates one that is too wide.

## Probability integral transform

`empirical_pit(samples, truth, ...)` evaluates the empirical CDF of the draws
at the truth: `F(y-) + u * a`, where `F(y-)` is the fraction of draws below the
truth and `a` the fraction equal to it. The atom convention `ties` sets `u`:
`"randomize"` (uniform `u`, requires `seed`), `"midpoint"` (`u = 0.5`) or
`"error"`. Without ties, `M` unweighted draws give values on the grid `k / M`.
With the same seed and ids, the randomized PIT and `sbc_ranks` use the same
tie position.

`empirical_pit(..., weights=w)` gives the weighted empirical PIT, for example
for importance-weighted draws. It inherits any error in the weights and is not
an SBC rank.

## Central intervals and coverage

`interval_coverage(samples, truth, levels, weights=None)` returns per-case
results with shape `(cases, levels, targets)`: the closed central interval
`[lower, upper]`, its `width`, and `covered = lower <= truth <= upper`. The
endpoints are inverse-CDF empirical quantiles at `(1 - level) / 2` and
`(1 + level) / 2`, so the interval holds at least `level` of the (weighted)
empirical mass.

With `M` unweighted draws the probability that an exchangeable truth falls in
the interval is not exactly the nominal level. The exact value is returned as
`exchangeable_coverage`; compare observed coverage with it, not only with the
nominal level. It assumes continuous draws: when the distribution has atoms,
ties at the endpoints can make the closed interval cover more often.

Widths must be representable: finite endpoints whose difference overflows
(for example `-1e308` and `1e308`) raise `ValueError`. The mean width in
`summarize_coverage` is computed without overflowing the sum.

Aggregation is a separate, explicit step. `summarize_coverage(result)`
averages over cases for each level and target, never pooling targets, and
reports `nominal_se = sqrt(level * (1 - level) / cases)`. That standard error
assumes independent cases; dependent cases (for example overlapping time
windows) make it too small.

## Reproducible tie-breaking

Randomized tie-breaking uses one random stream per `(case, target)` pair,
derived from the integer `seed`, a case id and a target id by `case_rng`.
`case_ids` and `target_ids` default to positions. Pass explicit unique ids
(non-negative integers or strings) to get the same result for a case when the
batch is reordered or subset. NumPy's global random state is never used.

`case_rng(seed, case_id, stream, target_id=None)` is public, so a sampler can
derive its own per-case generators in the same way. Different `stream` labels
give unrelated generators for the same seed and case.

## Worked example

The draws below come from a simple Gaussian model where the exact posterior is
known. In the first set the draws use the correct posterior spread; in the
second they are half as wide. The example scores both with the calibration
checks, the existing `marginal_crps`, and `brier_score` for the event
`target > 0`.

```python
import numpy as np

from skbel.metrics import (
    brier_score,
    interval_coverage,
    marginal_crps,
    sbc_rank_histogram,
    sbc_ranks,
    summarize_coverage,
)

rng = np.random.default_rng(7)
cases, draws = 500, 19
centre = rng.normal(size=(cases, 1, 1))
truth = centre[:, 0, :] + rng.normal(size=(cases, 1))
calibrated = centre + rng.normal(size=(cases, draws, 1))
too_narrow = centre + 0.5 * rng.normal(size=(cases, draws, 1))

n_bins = 5
expected = cases / n_bins
binomial_sd = np.sqrt(cases * (1 / n_bins) * (1 - 1 / n_bins))

for name, samples in (("calibrated", calibrated), ("too narrow", too_narrow)):
    ranks = sbc_ranks(samples, truth, seed=0)
    counts = sbc_rank_histogram(ranks, draws, n_bins)[:, 0]
    summary = summarize_coverage(interval_coverage(samples, truth, [0.5, 0.9]))
    crps = marginal_crps(samples, truth).mean()
    p_positive = (samples[:, :, 0] > 0).mean(axis=1)
    labels = (truth[:, 0] > 0).astype(int)
    brier = brier_score(np.column_stack([1 - p_positive, p_positive]), labels).mean() / 2

    print(name)
    print("  rank counts", counts, f"expected {expected:.0f} +/- {binomial_sd:.1f}")
    print("  coverage", summary.coverage[:, 0].round(3),
          "reference", summary.exchangeable_coverage.round(3),
          "+/-", summary.nominal_se.round(3))
    print(f"  mean CRPS {crps:.3f}, binary Brier for target > 0 {brier:.3f}")
```

Reading the output: for the calibrated draws the rank counts stay within a few
binomial standard deviations of the expected count, and coverage at both levels
is close to `exchangeable_coverage`. For the too-narrow draws the edge bins are
overfull, the centre bin is underfull, and coverage falls well below the
reference. The mean CRPS is worse for the narrow draws (0.582 against
0.559), but the binary Brier score for `target > 0` is slightly better
(0.155 against 0.162): a proper score for one event can still favour an
overconfident posterior in a finite sample. These scores mix calibration with
sharpness, so the rank and coverage checks are the ones that identify the
miscalibration. The binary Brier value divides
`brier_score` by 2, as described in [Posterior metrics](paper_metrics.md).

## Sampling with BEL

`BEL.random_sample` draws each observation row from its own generator,
derived from the model's `random_state` and the row index with `case_rng`.
This applies to the `mvn`, `kde` and `tm` modes. Repeated calls with the same
seed return the same samples; a row selected with `obs_n` receives the same
samples as that row of the full batch; and two rows with identical
observations receive independent draws. As before, `tm` mode with `n_posts`
equal to the number of training rows maps the training samples instead of
drawing random reference values, so it uses no random numbers.

Setting `seed` or `random_state` only stores the value; it must be a
non-negative integer or `None`. Neither the setters nor sampling read or
reseed NumPy's global random state. If the seed is `None`, sampling takes a
fresh seed from operating-system entropy and stores it.

These are intentional behaviour changes. Earlier versions reseeded NumPy's
global random state whenever the seed was set or samples were drawn, and drew
all rows from one shared stream. The same seed therefore now gives different
numerical samples than before, code that relied on `bel.seed = s` also seeding
other NumPy-based steps (such as a randomized PCA fit) must seed those steps
explicitly, and an unseeded model no longer takes its seed from
`numpy.random.seed`. The posterior functions themselves are unchanged.
`skbel.algorithms.it_sampling` gains an optional `rng` argument; without it, it
still uses NumPy's global random state.
