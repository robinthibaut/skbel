#  Copyright (c) 2026. Robin Thibaut, Ghent University

"""Neural posterior ensemble on a synthetic simulation bank, from designs to decisions.

One runnable chain through public SKBEL API:

1. a seeded synthetic :class:`~skbel.design.SimulationBank` (two targets drive a
   periodic and a linear response at four sensors) split into fitting,
   development and test rows;
2. a small unfitted :class:`~skbel.neural.NeuralPosterior` template, refitted
   once per design with :func:`~skbel.design.fit_designs` (each design gets a
   fresh clone);
3. joint draws of the development rows and a
   :class:`~skbel.neural.QuantileRecalibrator` fitted on them against their
   truths, with groups routed by a caller rule that reads the observations only;
4. recalibrated draws of the test rows scored with
   :func:`~skbel.evaluation.score_samples`, an event probability from
   :func:`~skbel.evaluation.event_probabilities` and a two-action decision from
   :func:`~skbel.evaluation.decision_risks`.

The network sizes and epoch counts are deliberately tiny so the example runs
in seconds; they are not recommended settings. The recalibration is marginal
(per target and group): it does not recalibrate the dependence between
targets.

Requires the optional extra: ``pip install 'skbel[neural]'``. Run with
``python examples/neural_posterior.py``. Importing this module has no side
effect; ``main`` prints and writes no file.
"""

from __future__ import annotations

import numpy as np

from skbel.design import SimulationBank, fit_designs
from skbel.evaluation import decision_risks, event_probabilities, score_samples
from skbel.metrics import summarize_coverage
from skbel.neural import NeuralPosterior, QuantileRecalibrator

N_ROWS, N_TIME, N_SENSORS = 600, 24, 4
FIT = np.arange(0, 360)
DEVELOPMENT = np.arange(360, 480)
TEST = np.arange(480, 600)
N_DRAWS = 63
LEVELS = [0.5, 0.9]


def synthetic_bank(seed: int = 0) -> SimulationBank:
    """Seeded bank: target 0 scales a damped wave, target 1 a linear depth profile."""
    rng = np.random.default_rng(seed)
    targets = rng.normal(size=(N_ROWS, 2))
    t = np.arange(N_TIME)
    depth = np.arange(N_SENSORS)
    wave = np.sin(2 * np.pi * t / 12)[None, :, None] * np.exp(-depth / 2)[None, None, :]
    obs = (
        targets[:, 0, None, None] * wave
        + targets[:, 1, None, None] * (depth / 3)[None, None, :]
        + 0.3 * rng.normal(size=(N_ROWS, N_TIME, N_SENSORS))
    )
    return SimulationBank(obs, targets)


def template() -> NeuralPosterior:
    return NeuralPosterior(
        n_members=3,
        covariance="full",
        n_components=3,
        hidden=(16, 16),
        n_pca=6,
        batch_size=32,
        max_epochs=150,
        n_validation=60,
        patience=10,
        random_state=1,
    )


def route(observations) -> np.ndarray:
    """Caller-defined group per (case, target) from the observations alone (no truth)."""
    signal = observations[:, :, -1].mean(axis=1)
    return np.repeat((signal > 0).astype(np.int64)[:, None], 2, axis=1)


def main(verbose: bool = True) -> dict:
    bank = synthetic_bank()
    designs = [
        bank.select([0, 3], start=0, stop=N_TIME, step=2),
        bank.select([1, 2], start=0, stop=N_TIME, step=6),
    ]
    refits = fit_designs(template(), bank, designs, train_rows=FIT)
    obs, truth = bank.observations, bank.targets
    results = {}
    for name, refit in zip(("dense", "sparse"), refits, strict=True):
        dev = refit.model.sample(
            refit.features(obs[DEVELOPMENT]), N_DRAWS, case_ids=DEVELOPMENT, seed=10
        )
        recal = QuantileRecalibrator().fit(
            dev, truth[DEVELOPMENT], route(obs[DEVELOPMENT]), seed=11, min_cases=30
        )
        raw = refit.model.sample(refit.features(obs[TEST]), N_DRAWS, case_ids=TEST, seed=12)
        draws = recal.transform(raw, route(obs[TEST]))
        scores = score_samples(draws, truth[TEST], levels=LEVELS)
        coverage = summarize_coverage(scores.coverage)
        p_event = event_probabilities(draws[:, :, :1] > 1.0)[:, 0]
        # act (cost 1) or wait (loss 5 if the event happens)
        losses = np.stack([np.ones(draws.shape[:2]), 5.0 * (draws[:, :, 0] > 1.0)], axis=2)
        risks = decision_risks(losses)
        results[name] = {
            "crps": scores.crps.mean(axis=0),
            "coverage": coverage.coverage,
            "mean_event_probability": float(p_event.mean()),
            "act_fraction": float(np.mean([0 in a for a in risks.bayes_actions])),
        }
        if verbose:
            print(
                f"{name:>6}: mean CRPS {np.round(results[name]['crps'], 3)}, "
                f"coverage {np.round(results[name]['coverage'], 2).tolist()}, "
                f"mean P(event) {results[name]['mean_event_probability']:.3f}, "
                f"act in {results[name]['act_fraction']:.0%} of cases"
            )
    return results


if __name__ == "__main__":
    main()
