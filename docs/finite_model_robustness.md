# Model-robust finite acquisition and stopping

`skbel.metrics.robust.robust_acquisition_policy` chooses a measurement-and-stopping
policy for a **finite joint model that is only known up to a set of candidate
laws**. The caller supplies several models of the same world; the function returns
the complete deterministic decision tree whose worst expected total loss, over
those models, is smallest. Import it explicitly:

```python
from skbel.metrics.robust import robust_acquisition_policy
```

It is NumPy-only, simulates and fits nothing, and is independent of the BEL
estimators. It complements `finite_acquisition_policy` (one law, one Bayes
optimum). It is a software capability with hand-checkable oracles, not a
calibrated posterior and not evidence about any real monitoring network.

## Model

One common world of `N` identified atoms describes the terminal loss and every
candidate measurement outcome jointly. No independence between measurements is
assumed, and nothing is built from marginals.

| Argument | Shape | Meaning |
| --- | --- | --- |
| `losses` | `(atoms, actions)` | finite loss of each terminal action if the atom is the truth |
| `outcomes` | `(atoms, candidates)` | integer label each candidate returns for each atom |
| `model_weights` | `(models, atoms)` | each row a joint law on the same atoms; normalized by the function |
| `atom_ids`, `model_atom_ids` | strings | one id per atom; every model's ids must equal `atom_ids` in content and order |
| `costs` | `(candidates,)` or `None` | finite, non-negative acquisition cost, in the units of `losses` |
| `horizon` | integer 0, 1 or 2 | maximum number of acquisitions |
| `mixture_weights` | `(models,)` or `None` | explicit model probabilities for the mixture baseline |

The caller owns the premises that matching ids cannot certify: that atom `i` is the
same physical situation under every model, and that losses and costs share one
unit. Models that disagree about the experiment itself (different outcomes, actions
or costs) must be rewritten as one union world by the caller; nothing is aligned
silently.

A weight of exactly zero means the atom is impossible under that model. An atom
with zero weight under *every* model is dropped from the policy world, but only
after every row, loss and label has been validated, so an invalid dropped atom
still raises. A *positive* weight that normalizes to zero in float64 raises
`ValueError`; it is never dropped or repaired.

## Policy class and objective

A policy is a precommitted deterministic tree. It acquires candidates without
replacement, sees only the acquired labels, and ends every path in a terminal
action. It is complete over the *union-reachable* histories: a history that has
positive mass under any model has a branch, even if another model gives it zero
mass. For a model with zero mass on a history the contribution is exactly zero;
no conditional probability or posterior is formed for it.

For one policy and one model `m`, with that model fixed for the entire tree:

```text
risk_m = sum over atoms of  w_m[atom] * (cost of candidates acquired on the atom's path
                                          + loss of the atom's terminal action)
```

The function returns every policy minimizing `max_m risk_m`. The worst model is
chosen once, at the root, for the whole tree.

This is deliberately **not**:

- a branch-switching adversary that re-chooses the model at each node, or a
  recombination of independently robust subtrees (the worked example below shows
  these answer a different question);
- a regret objective, which subtracts model-specific optima;
- a randomized policy: randomized mixtures over trees can do strictly better and are
  not searched;
- an unrestricted global minimax claim, a calibration, a coverage statement, or a
  statement about any real system.

There is no online re-optimization. Replanning after an observation is a new
problem, not an execution of the returned commitment.

## Results

`RobustResult` holds:

- `robust`: **every** complete policy attaining the minimum worst risk, each with
  its tree (`PolicyNode`), per-model risks, worst risk and worst models. Branches of
  different tied minimizers must not be spliced; only whole trees are certified.
- `risk_table`: read-only `(policies, models)` array, one row per enumerated policy,
  in a canonical order: at each node terminal actions ascending, then each unacquired
  candidate ascending; children follow ascending label with the first label varying
  slowest. `policy(i)` returns the tree of row `i`.
- Baselines chosen **without** knowing the true model, each completed over the
  union-reachable histories: among *all* policies attaining the baseline's own
  optimum, including ties and every arbitrary completion of histories with zero mass
  under its law, the one with the best worst-model risk is selected.
  - `nominal[m]`: the Bayes optimum of model `m`, replayed under every model.
  - `mixture`: the Bayes optimum of the explicit `mixture_weights` (absent if not
    given; no uniform prior is assumed).
  - `fixed_subset`: the best worst risk over policies that acquire the same set of
    candidates on every path with terminal actions common to all models; the empty
    set (stop only) is included.
  - `stop_only`: the best common terminal action.
- `work`: exact counts (`policies`, `model_evaluations`, `nodes`, `generated`).

## Tie semantics

Arithmetic is IEEE float64. Each per-model risk is the exactly rounded sum of the
float products, and minimizers use exact `==` with no tolerance or clamping. Values
equal in exact arithmetic but different in float64 are not reported as ties; ties
are exactly detectable on dyadic inputs (weights and costs that are multiples of a
power of two, small integer losses). Comparing against `finite_acquisition_policy`
values should use a small tolerance, because that planner conditions step by step.

## Limits

The generator enforces its caps before generating any policy, tree or risk table and
rejects larger inputs with `ValueError`; it never prunes. The temporary NumPy copies
made while validating the inputs are not capped by these limits.

- At most 8 models, 32 atoms, 3 candidates, 3 actions, horizon 2, and 2 outcome
  labels per candidate over the positive-mass atoms.
- At most 1326 complete policies, 31 history nodes and 1524 generated sub-policies
  per call. For `K` candidates, `A` actions and 2 labels the policy count at horizon 2
  is at most `A + K * (A + (K - 1) * A**2) ** 2`; this is 74 for `K = A = 2`.

These are acceptance bounds, not a guarantee of runtime or memory. The test suite
exercises focused fixtures (at most 8 atoms, 3 models, 2 candidates, 2 actions); the
upper limits are enforced by the generator but are not exercised by the tests.

## Induced randomization

Outcomes are part of the joint model, so a paid candidate whose label is independent
of every model's loss-relevant state still acts as a random coin: a deterministic tree
can send half of the coin outcomes to each of two terminal actions and thereby lower
its worst-case risk, as a randomized action would. That gain is randomization, not
information, and must not be reported as the value of acquiring information. The test
suite contains such a coin fixture, where no single model values the measurement but
the worst case still falls, and a separate main exhibit where the benefit comes from
outcomes that are informative about the atoms.

## Worked example

Four atoms are the combinations of two bits `(x, y)`; candidate 0 returns `x` (cost 1)
and candidate 1 returns `y` (cost 2). The only atom where action 1 is right is
`(x, y) = (0, 1)`: action 0 loses 16 there, and action 1 loses 4 on every other atom.
Three models weight the atoms `(00, 01, 10, 11)` by sixteenths:

| Model | `00` | `01` | `10` | `11` |
| --- | --- | --- | --- | --- |
| 0 | 7 | 1 | 4 | 4 |
| 1 | 2 | 8 | 3 | 3 |
| 2 | 7 | 5 | 2 | 2 |

Exact risks (models 0, 1, 2) of representative policies. This is a selection, not a
Pareto frontier: for example, reading both candidates is strictly dominated by the
adaptive policy in the last row.

| Policy | Risks | Worst |
| --- | --- | --- |
| stop, action 0 | 1, 8, 5 | 8 |
| stop, action 1 | 15/4, 2, 11/4 | 15/4 |
| read `x`; act 1 if `x = 0` else 0 | 11/4, 3/2, 11/4 | 11/4 |
| read `y`; act 1 if `y = 1` else 0 | 3, 11/4, 5/2 | 3 |
| read `y`; if `y = 1` read `x` and act 1 if `x = 0` else 0; else act 0 | 37/16, 43/16, 39/16 | 43/16 |
| read both | 3, 3, 3 | 3 |
| **read `x`; if `x = 0` read `y` and act `y`; else act 0** | **2, 9/4, 5/2** | **5/2** |

- Each model's own Bayes optimum differs: model 0 stops with action 0, model 1 reads
  `x` only, model 2 reads `y` first and then `x` only when `y = 1`. Replayed under all
  models their worst risks are 8, 11/4 and 43/16.
- The unique robust policy (worst risk 5/2, attained by model 2) reads `x`, reads `y`
  only when `x = 0`, and acts on the outcomes. It beats every baseline: nominal 8,
  11/4 and 43/16; mixture with probabilities `(1/4, 1/2, 1/4)` 11/4; best fixed subset
  11/4; best stop-only 15/4.
- The acquisitions are informative, not coins: the value of observing under models 1
  and 2 is 1/2 and 5/16, and under model 0 it is 0, which is why the robust policy
  pays for measurements that model 0 alone would decline.
- A branch-switching criterion that picks the model again at every leaf ranks the
  two policies the other way round: its leaf-wise worst total is 3 for the fixed
  `x`-only policy and 53/16 for the adaptive one, while the root worst risks are 11/4
  and 5/2. This pairwise reversal shows that choosing the model per branch is not the
  same commitment; it does not say which policy is optimal for that criterion.

A second fixture adds a free candidate and impossible histories (one model never sees
`x = 1`, another never sees `x = 0`, one atom has zero mass under every model): there a
model's Bayes optimum has several tied completions, and the baseline keeps the one
with the best worst risk.

## Validation

`ValueError` for wrong shapes or ranks, empty axes, non-finite losses, non-finite or
negative weights and costs, a non-positive weight total, a positive weight lost to
normalization, mismatched or duplicate atom ids, inputs beyond the limits above, a
candidate with more than two labels, and any total that overflows float64. Negative
weights and costs are rejected, but finite losses of either sign are accepted.
`TypeError` for non-numeric or bool arrays, non-integer `outcomes`, non-string ids,
and a horizon that is not an integer. Inputs are copied and never modified; the risk
table is a non-writable view of immutable bytes whose write flag cannot be re-enabled,
and the other results are frozen dataclasses and tuples.
