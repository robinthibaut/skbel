#  Copyright (c) 2026. Robin Thibaut, Ghent University

"""Runnable example: finite prospective risk/utility ranking of candidates.

This composes three existing primitives -- :func:`skbel.metrics.expected_action_losses`,
:func:`skbel.metrics.bayes_action_set`, and
:func:`skbel.metrics.rank_prospective_measurements` -- into a tiny, fully
finite, hand-checkable decision problem. Everything here (the states, the
prior, the loss table, and each candidate's outcomes/likelihoods) is made up
for illustration: the loss table's numbers and units are entirely the
caller's choice, this example performs no calibrated inference and no
posterior update against real data, it has no hydrological or other
field/domain performance meaning, and it demonstrates no methodological
novelty. None of the composed functions run a simulator or an experimental
design search themselves; this script is the one enumerating candidates,
outcomes, and posteriors explicitly, by hand, over a fixed finite set.

Decision problem
-----------------
Three finite latent states with a nonuniform prior, two actions, and a
caller-defined finite loss table ``LOSS_TABLE[state, action]`` (arbitrary
units). Three prospective candidates are compared:

- ``no_measurement``: an uninformative candidate with a single, fully
  uninformative outcome (its posterior always equals the prior).
- ``informative_probe`` / ``informative_probe_duplicate``: two candidates
  with identical, informative, state-conditional outcome likelihoods,
  included as literal duplicates of each other so the example also exercises
  exact-tie handling in the rankings.

For each candidate, this script exactly enumerates every possible future
outcome, computes each outcome's predictive probability and posterior state
weights via Bayes' rule, and computes the expected future Bayes risk as

    E_y[ min_a E[L(a, state) | y] ]

i.e. it minimizes over actions *after* conditioning on each outcome, then
averages over outcomes weighted by their predictive probability -- never the
other way around, and never by looking at only one favorable outcome.
Utility is defined as the current (prior) Bayes risk minus this expected
future Bayes risk (the expected risk reduction from measuring).
"""

from __future__ import annotations

import numpy as np

from skbel.metrics import bayes_action_set, expected_action_losses, rank_prospective_measurements

STATES = ("low", "mid", "high")
PRIOR = np.array([0.5, 0.3, 0.2])

ACTIONS = ("treat", "monitor")
# LOSS_TABLE[state, action], in caller-defined arbitrary loss units.
LOSS_TABLE = np.array(
    [
        [10.0, 0.0],
        [4.0, 6.0],
        [1.0, 20.0],
    ]
)

# Each candidate's likelihood matrix has shape (states, outcomes); row s is
# the finite conditional outcome distribution p(outcome | state=s), and must
# sum to 1 (checked explicitly below, not assumed).
CANDIDATE_OUTCOMES = {
    "no_measurement": ("none",),
    "informative_probe": ("y0", "y1"),
    "informative_probe_duplicate": ("y0", "y1"),
}
CANDIDATE_LIKELIHOODS = {
    "no_measurement": np.array([[1.0], [1.0], [1.0]]),
    "informative_probe": np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]]),
    "informative_probe_duplicate": np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]]),
}

_NORM_ATOL = 1e-12


def check_normalized(mass: np.ndarray, label: str) -> None:
    """Explicit normalization check: raise unless ``mass`` sums to 1."""
    total = float(np.sum(mass))
    if not np.isclose(total, 1.0, atol=_NORM_ATOL):
        raise ValueError(f"{label} does not sum to 1 (got {total})")


def check_probabilities(values: np.ndarray, label: str) -> None:
    """Explicit validity check: raise unless every entry is finite and nonnegative."""
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} contains nonfinite values")
    if np.any(values < 0):
        raise ValueError(f"{label} contains negative values")


def predictive_outcome_probabilities(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """Exact enumeration: p(y) = sum_s prior[s] * p(y | state=s), for every y."""
    predictive = prior @ likelihood
    check_normalized(predictive, "predictive outcome probabilities")
    return predictive


def posterior_given_outcome(
    prior: np.ndarray, likelihood: np.ndarray, outcome_index: int, predictive: np.ndarray
) -> np.ndarray:
    """Bayes' rule: p(state=s | y) = prior[s] * p(y | state=s) / p(y).

    Raises ``ValueError`` before any division if ``p(y)`` is exactly zero:
    no posterior exists for an impossible outcome.
    """
    check_probabilities(predictive, "predictive outcome probabilities")
    if predictive[outcome_index] == 0.0:
        raise ValueError(
            f"outcome index {outcome_index} has zero predictive probability; "
            "no posterior is defined for an impossible outcome"
        )
    posterior = prior * likelihood[:, outcome_index] / predictive[outcome_index]
    check_normalized(posterior, f"posterior for outcome index {outcome_index}")
    return posterior


def current_bayes_risk(loss_table: np.ndarray, prior: np.ndarray) -> tuple[float, np.ndarray]:
    """Current (prior) expected loss per action and the resulting Bayes risk."""
    check_normalized(prior, "prior")
    action_losses = expected_action_losses(loss_table, weights=prior)
    min_risk, _ = bayes_action_set(loss_table, weights=prior)
    return min_risk, action_losses


def expected_future_bayes_risk(
    loss_table: np.ndarray, prior: np.ndarray, likelihood: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """E_y[min_a E[L(a, state) | y]]: minimize per outcome, then average over outcomes.

    This order is deliberate and load-bearing: minimizing after averaging
    over outcomes (i.e. committing to one action before observing anything)
    or conditioning on a single favorable outcome instead of enumerating all
    of them would both silently change the answer -- see the independent
    test module for hand-computed counterexamples.

    Outcomes with exactly zero predictive probability (e.g. an all-zero
    likelihood column, or one reachable only from states with zero prior mass)
    can never be observed: no posterior exists for them. Their entry in the
    returned ``per_outcome_min_risk`` is ``np.nan`` (undefined, not a risk),
    and they are excluded from the weighted sum by a boolean mask rather than
    by multiplying ``0 * nan``.
    """
    check_probabilities(prior, "prior")
    check_probabilities(likelihood, "likelihood")
    if prior.ndim != 1 or likelihood.ndim != 2 or likelihood.shape[0] != prior.shape[0]:
        raise ValueError(f"incompatible shapes: prior {prior.shape}, likelihood {likelihood.shape}")
    check_normalized(prior, "prior")
    for s in range(likelihood.shape[0]):
        check_normalized(likelihood[s], f"likelihood row for state {s}")

    predictive = predictive_outcome_probabilities(prior, likelihood)
    n_outcomes = likelihood.shape[1]
    per_outcome_min_risk = np.full(n_outcomes, np.nan)
    possible = predictive > 0.0
    for y in np.flatnonzero(possible):
        posterior = posterior_given_outcome(prior, likelihood, int(y), predictive)
        min_risk, _ = bayes_action_set(loss_table, weights=posterior)
        per_outcome_min_risk[y] = min_risk

    expected_future_risk = float(np.sum(predictive[possible] * per_outcome_min_risk[possible]))
    return expected_future_risk, predictive, per_outcome_min_risk


def compute_rankings(
    loss_table: np.ndarray = LOSS_TABLE,
    prior: np.ndarray = PRIOR,
    likelihoods: dict[str, np.ndarray] = CANDIDATE_LIKELIHOODS,
):
    """Compute current risk, per-candidate expected future risk/utility, and rankings."""
    current_risk, _ = current_bayes_risk(loss_table, prior)

    candidate_names = tuple(likelihoods.keys())
    expected_future_risks = []
    utilities = []
    details = {}
    for name in candidate_names:
        efr, predictive, per_outcome_min_risk = expected_future_bayes_risk(
            loss_table, prior, likelihoods[name]
        )
        utility = current_risk - efr
        expected_future_risks.append(efr)
        utilities.append(utility)
        details[name] = {
            "expected_future_risk": efr,
            "utility": utility,
            "predictive_outcome_probabilities": predictive,
            "per_outcome_min_risk": per_outcome_min_risk,
        }

    risk_ranking = rank_prospective_measurements(
        candidate_names, np.array(expected_future_risks), criterion="risk"
    )
    utility_ranking = rank_prospective_measurements(
        candidate_names, np.array(utilities), criterion="utility"
    )
    return current_risk, details, risk_ranking, utility_ranking


def main() -> tuple[float, dict, object, object]:
    current_risk, details, risk_ranking, utility_ranking = compute_rankings()

    print(f"Current Bayes risk (prior only, no measurement): {current_risk}")
    for name, info in details.items():
        print(
            f"  {name}: expected future risk = {info['expected_future_risk']}, "
            f"utility = {info['utility']}"
        )
    print(f"Risk ranking (best/lowest first): {risk_ranking.order}")
    print(f"Utility ranking (best/highest first): {utility_ranking.order}")

    return current_risk, details, risk_ranking, utility_ranking


if __name__ == "__main__":
    main()
