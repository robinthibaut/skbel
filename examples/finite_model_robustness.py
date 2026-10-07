#  Copyright (c) 2026. Robin Thibaut, Ghent University

"""Model-robust acquisition and stopping on a finite illustrative table.

Four atoms are the combinations of two bits ``(x, y)``. Candidate 0 returns ``x``
and candidate 1 returns ``y``. Terminal action 1 is right only for the atom
``(x, y) = (0, 1)``. Three caller models disagree about how likely each atom is.

The table is illustrative: the weights, losses and costs are made up and carry no
calibration or physical meaning.
"""

from skbel.metrics.robust import robust_acquisition_policy

ATOM_IDS = ["x0y0", "x0y1", "x1y0", "x1y1"]
LOSSES = [[0, 4], [16, 0], [0, 4], [0, 4]]  # (atom, action)
OUTCOMES = [[0, 0], [0, 1], [1, 0], [1, 1]]  # (atom, candidate): x, y
COSTS = [1, 2]
MODEL_WEIGHTS = [[7, 1, 4, 4], [2, 8, 3, 3], [7, 5, 2, 2]]  # sixteenths per atom
MIXTURE = [0.25, 0.5, 0.25]


def describe(node, indent=""):
    """Print a policy tree."""
    if node.kind == "stop":
        print(f"{indent}stop, act {node.action}")
        return
    print(f"{indent}measure candidate {node.candidate}")
    for label, child in node.branches.items():
        print(f"{indent}  outcome {label}:")
        describe(child, indent + "    ")


def main():
    result = robust_acquisition_policy(
        LOSSES,
        OUTCOMES,
        MODEL_WEIGHTS,
        ATOM_IDS,
        [ATOM_IDS] * len(MODEL_WEIGHTS),
        costs=COSTS,
        horizon=2,
        mixture_weights=MIXTURE,
    )
    print(f"{result.policy_count} complete policies, {len(result.robust)} robust minimizer(s)")
    for policy in result.robust:
        print(f"worst-model risk {policy.worst_risk}; risk per model {policy.model_risks}")
        describe(policy.root)
    print("\nWorst-model risk of each comparison policy:")
    for baseline in (*result.nominal, result.mixture, result.fixed_subset, result.stop_only):
        print(f"  {baseline.name:>14}: {baseline.worst_risk}")


if __name__ == "__main__":
    main()
