# Finite adaptive acquisition and stopping

`skbel.metrics.finite_acquisition_policy` computes the exact optimal adaptive
measurement-and-stopping policy for a **finite, fully specified joint model**.
It is NumPy-only and independent of the BEL estimators. It chooses a second
measurement *conditional on the first outcome* and compares every choice with
stopping, which `expected_action_losses` / `bayes_action_set` (one decision, no
acquisition) do not do.

It is a software capability with hand-checkable oracles. It is not a
simulator, a calibrated posterior, or evidence about any real monitoring
network.

## Model

The caller supplies finite *atoms* that define the joint distribution of
everything the policy needs. No conditional independence between measurements
is assumed.

| Argument | Shape | Meaning |
| --- | --- | --- |
| `losses` | `(atoms, actions)` | finite loss of each terminal action if the atom is the truth |
| `outcomes` | `(atoms, candidates)` | integer label each candidate would return for each atom |
| `weights` | `(atoms,)` or `None` | finite, non-negative, positive total; normalized by the function; uniform if `None` |
| `costs` | `(candidates,)` or `None` | finite, non-negative cost per candidate; zero if `None` |
| `horizon` | integer 0, 1 or 2 | maximum number of further acquisitions |

Atoms with a supplied weight of exactly zero are removed from all conditioning.
Their losses and labels are still validated, but their labels never appear as
branches and no posterior is invented for an impossible outcome.

A *positive* supplied weight that becomes zero in float64 during normalization
(extreme dynamic range, including after overflow rescaling, e.g. weights
`[1e-200, 1e200]`) is a representability failure, not impossible support. It
raises `ValueError`; it is never dropped, clamped or approximated.

## Rule

At every node, with conditional weights `q` over the node's positive-mass atoms:

- **stop**: `stop_risk = min_a  sum_atoms q * loss[atom, a]`;
- **acquire candidate `c`** (not yet acquired on this path):
  `value = cost[c] + sum_o P(o) * value(node | o)`, where `o` ranges over the
  outcome labels of `c` with positive mass and `value(node | o)` is the same
  rule applied to the conditioned support with `horizon - 1`.

The node value is the minimum over stopping and all legal acquisitions.
Acquisition is without replacement. Costs paid earlier on the path are sunk, so
each cost enters the root objective exactly once, where it is incurred. The
horizon is a hard cap: horizon 0, or no candidate left, stops.

## Result

The function returns a frozen `AcquisitionNode`:

- `value`, `stop_risk`;
- `terminal_actions`: every action whose expected loss exactly equals `stop_risk`;
- `optimal_choices`: every choice whose value exactly equals `value`, as the
  literal `"stop"` (listed first if it ties) followed by candidate indices in
  ascending order; a single entry means a unique optimum;
- `candidate_values`: read-only map from **every** legal candidate to
  `CandidateEvidence(value, immediate_cost, expected_continuation_value, branches)`;
- `branches`: read-only map from each positive-mass integer outcome label to
  `AcquisitionBranch(probability, node)`;
- `horizon`, `acquired` (path of candidates so far), `support` (original atom
  indices with positive conditional mass).

Everything needed to execute the policy and to audit the objective is exposed:
`value == immediate_cost + expected_continuation_value` exactly, and
`expected_continuation_value == sum(probability * node.value)` over branches.
Inputs are copied and never modified.

```python
import numpy as np
from skbel.metrics import finite_acquisition_policy

# Two equiprobable states, one perfect measurement costing 1/4.
policy = finite_acquisition_policy(
    losses=[[0, 1], [1, 0]], outcomes=[[0], [1]], costs=[0.25], horizon=1
)
policy.stop_risk  # 0.5
policy.value  # 0.25
policy.optimal_choices  # (0,)
```

## Worked eight-atom example

Atoms are the eight combinations of three fair bits `(A, B, C)`. The hidden
state is `S = B` if `A = 0` and `S = C` if `A = 1`; the action is a guess of
`S` with 0/1 loss; each measurement costs 1/50. Exact values:

| Strategy | Total risk + cost |
| --- | --- |
| stop | 1/2 |
| measure `A` only | 13/25 (`A` says nothing about `S`) |
| `B` only, `C` only | 27/100 |
| any fixed pair | 29/100 |
| **adaptive**: `A`, then `B` if `A = 0` else `C` | **1/25** |

The best non-adaptive subset of at most two measurements achieves 27/100, so
the adaptive policy is strictly better. Its first choice `A` is unique and the
second choice depends on the observed `A`. The tests check these values against
hand calculations and against independent exhaustive rational (`Fraction`)
recursions for both the adaptive policy and fixed subsets.

## Tie semantics

All arithmetic is float64 and minimizers are found with exact `==`. There is no
`isclose`, tolerance or clamping, so a nonoptimal choice is never added to
`optimal_choices` or `terminal_actions`.

- Two choices tie only if their float64 values are exactly equal under `==`
  (signed zero equals signed zero; this is not a bit-pattern comparison). Values
  equal in exact arithmetic but different by rounding are **not** reported as
  ties.
  Ties are exactly detectable on dyadic-rational inputs (for example weights
  and costs that are multiples of a power of two with small integer losses).
- A measurement that cannot split the current support (a single outcome) passes
  the conditional weights to its child unchanged with probability exactly
  `1.0`, so a free uninformative measurement ties exactly with stopping for any
  float input.

## Validation

`ValueError` for wrong shapes or ranks, empty atom or action axes, non-finite
losses/weights/costs, negative weights or costs, non-positive weight total,
a positive weight lost to float64 normalization, out-of-range horizon, and values that overflow float64. `TypeError` for
non-numeric dtypes, bool arrays, non-integer `outcomes` dtype, and a horizon
that is bool or not an integer. `candidates == 0` (shape `(atoms, 0)`) is valid
and stops.

## Limits

- Work is exponential in the horizon: about `1 + K*B + K*(K-1)*B**2` nodes for
  `K` candidates with at most `B` outcome labels, each costing
  `O(atoms * actions)`. Test fixtures (at most 128 atoms, 6 candidates,
  4 actions, horizon 2) are acceptance bounds, not a guarantee of runtime or
  memory for arbitrary input.
- The result is only as meaningful as the supplied joint model. This module
  supplies no application-specific loss, prior, simulator or calibration; the
  loss table is the caller's responsibility.
