# From a BEL posterior to an adaptive decision

`examples/bel_inference_to_decision.py` is one runnable application of public
SKBEL pieces: a fitted `BEL(mode="mvn")` posterior, an explicit finite
observation/loss model, `finite_acquisition_policy`, a realized branch with a
terminal action, and a separate retrospective `energy_score`.

This finite software example illustrates supplied assumptions; it does not
establish calibration or validate a real application.

## Flow

1. **Fixture (fit once).** Sixteen deterministic rows from all four sign bits
   `u, v, e, f in {-1, 1}`: `X = (u, v)`, `Y = (2u + e, 3v + 2f)`.
   `StandardScaler` pipelines on both sides, `CCA(n_components=2)`,
   `BEL(mode="mvn")`.
2. **Observation (predict once).** Two explicit observations `(0, 0)` and
   `(1/4, -1/4)` go through one `predict(..., noise=1/8, return_samples=False)`
   call. `posterior_mean` and `posterior_covariance` are read directly. No PRNG,
   `random_sample`, KDE, transport map or private/cached state is involved.
3. **Finite law.** Each canonical Gaussian becomes four equal-mass points
   `mu + L z`, with `L = cholesky(Sigma)` and `z` the four sign pairs. The points
   are back-transformed with `BEL.inverse_transform`, shape `(cases, 4, 2)`, into
   original target units. Non-finite, not positive-definite or asymmetric
   (beyond `1e-12`) covariances raise; nothing is clipped, jittered or
   symmetrized. The Cholesky factor reads the lower triangle.
4. **Caller model (assumed, never learned).**
   - Terminal actions are the fixed centres `(-1, 0)` and `(1, 0)` in original
     target units; the loss of an atom is its squared Euclidean distance to the
     chosen centre. This is an example preference, not a physical utility.
   - Candidates A, B, C return binary outcomes. A `(4, 8)` joint categorical
     likelihood over the tuples `(A, B, C)` is supplied per atom (rows
     non-negative, finite and exactly normalized; no product of marginals). The
     example uses the toy state `S = B if A == 0 else C` and gives probability
     1/4 to each tuple consistent with the atom's sign label.
   - Candidate costs (1, 1, 1) share the loss units of the squared distances.
5. **Policy.** Atoms times tuples give 32 joint rows with weight
   `(1/4) * likelihood`; any positive product mass lost to underflow raises
   before the policy runs. `finite_acquisition_policy(horizon=2)` returns every
   tied choice, every branch probability, the cost decomposition, supports and
   the terminal Bayes action set at each node. `run_policy` and `follow_path`
   walk an actual path; an impossible outcome, a reused candidate or an
   unavailable candidate raises before any posterior is invented. On an exact
   stop tie `run_policy` stops first while keeping the whole tie set.
6. **Retrospective score.** `energy_score` evaluates the same four-atom law
   against an optional caller-supplied truth, in original units. It is not the
   decision loss, never enters the policy and is not a calibration claim:
   changing the truth cannot change the policy, changing losses or costs can.

## What the example shows

With the same likelihood, centres and costs, the two observations give
different posterior atoms in original units, hence different expected
squared-distance losses. The test module asserts that the first observation
(whose root actions are essentially tied by symmetry) acquires A first, whereas
for the second observation stopping at centre `(1, 0)` is optimal. The realized
paths A -> B and A -> C end in different terminal actions that follow the atoms
left in the support. Run the example to see the numbers; this page quotes none
that were not produced by a run.

## Secondary control

A sign-tag 0/1 loss (guess `S`) that does not depend on any atom value reuses
the same expansion. Its known rational values, derived in
[finite_acquisition.md](finite_acquisition.md), are adaptive `1/25` (A, then B if
`A = 0` else C) versus `27/100` for the best fixed subset of at most two
candidates. This checks the integration only. It is not a BEL-driven gain and
not evidence of measurement superiority.

## Limits

- The four points match the canonical posterior mean and covariance only, not
  the full Gaussian, its tails or any calibration.
- `noise` is the existing multiplier of the projected predictor covariance, not a
  physical measurement standard deviation.
- Likelihood, loss, centres and costs are supplied assumptions, not posterior
  predictive evidence.
- Policy ties use exact float64 `==` as documented for the policy; near ties
  from rounding are not reported as ties (the symmetric observation's root
  actions are such a case).
- Sizes are acceptance bounds: 16 training rows, 4 atoms, 32 joint rows, 3
  candidates, 2 actions, horizon 2, one fit and one two-case predict.

## Running

```
python examples/bel_inference_to_decision.py
python -W error -m pytest skbel/testing/test_bel_inference_to_decision.py
```

Importing the example has no side effect; `main()` prints and writes no file.
The tests share one fitted fixture and compare against independent oracles: a
Gaussian Schur conditional from the fitted canonical arrays, an affine fit to the
training rows, a `math.hypot` / `fsum` score loop, an all-node `Fraction`
recursion for the policy and exhaustive rational fixed subsets.
