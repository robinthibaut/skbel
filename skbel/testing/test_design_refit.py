#  Copyright (c) 2026. Robin Thibaut, Ghent University

"""Tests for fresh per-design BEL refits on a shared simulation bank."""

import numpy as np
import pytest
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from skbel import BEL
from skbel.design import DesignRefit, SimulationBank, fit_design, fit_designs

N_ROWS, N_TIME, N_SENSORS = 90, 24, 4
TRAIN = list(range(70))
HELDOUT = list(range(70, 90))


def _synthetic_bank(seed=0):
    """Small seeded bank: two targets drive a periodic and a linear sensor response."""
    rng = np.random.default_rng(seed)
    targets = rng.normal(size=(N_ROWS, 2))
    t = np.arange(N_TIME)
    depth = np.arange(N_SENSORS)
    wave = np.sin(2 * np.pi * t / 12)[None, :, None] * np.exp(-depth / 2)[None, None, :]
    obs = (
        targets[:, 0, None, None] * wave
        + targets[:, 1, None, None] * (depth / 3)[None, None, :]
        + 0.05 * rng.normal(size=(N_ROWS, N_TIME, N_SENSORS))
    )
    return obs, targets


def _template():
    # The CCA template keeps n_components=1; BEL.fit sets it on whatever it fits,
    # so an unchanged template shows that only the clone was fitted.
    return BEL(
        mode="mvn",
        X_pre_processing=Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=3))]),
        Y_pre_processing=Pipeline([("scaler", StandardScaler())]),
        regression_model=CCA(n_components=1),
        random_state=7,
    )


@pytest.fixture
def bank_and_designs():
    obs, targets = _synthetic_bank()
    bank = SimulationBank(obs, targets)
    designs = [
        bank.select([0, 3], start=0, stop=24, step=2),  # two sensors, every 2nd sample
        bank.select([2, 1, 0], start=6, stop=18, step=3, anchor=0),  # three sensors, short window
    ]
    return obs, targets, bank, designs


def _assert_unfitted(template):
    for step in (
        template.X_pre_processing.named_steps["scaler"],
        template.X_pre_processing.named_steps["pca"],
        template.Y_pre_processing.named_steps["scaler"],
    ):
        with pytest.raises(NotFittedError):
            check_is_fitted(step)
    assert not hasattr(template.regression_model, "x_rotations_")
    assert template.regression_model.n_components == 1
    assert template.x_pre_processed is None


def test_fresh_clone_per_design_and_template_untouched(bank_and_designs):
    obs, targets, bank, designs = bank_and_designs
    obs_before, targets_before = obs.copy(), targets.copy()
    template = _template()
    refits = fit_designs(template, bank, designs, train_rows=TRAIN)

    assert len(refits) == 2
    assert all(isinstance(r, DesignRefit) for r in refits)
    a, b = refits
    for r in refits:
        assert r.model is not template
        assert r.model.X_pre_processing is not template.X_pre_processing
        assert r.model.regression_model is not template.regression_model
        assert r.train_rows == tuple(TRAIN)
    pairs = [
        (a.model.X_pre_processing, b.model.X_pre_processing),
        (a.model.X_pre_processing.named_steps["pca"], b.model.X_pre_processing.named_steps["pca"]),
        (a.model.Y_pre_processing, b.model.Y_pre_processing),
        (a.model.regression_model, b.model.regression_model),
        (a.model.X_post_processing, b.model.X_post_processing),
    ]
    for left, right in pairs:
        assert left is not right

    _assert_unfitted(template)
    np.testing.assert_array_equal(obs, obs_before)
    np.testing.assert_array_equal(targets, targets_before)


def test_each_refit_learns_only_its_design_and_training_rows(bank_and_designs):
    _, _, bank, designs = bank_and_designs
    template = _template()
    for sel in designs:
        refit = fit_design(template, bank, sel, train_rows=TRAIN)
        X_train, Y_train = bank.features(sel, TRAIN)
        scaler = refit.model.X_pre_processing.named_steps["scaler"]
        assert scaler.n_features_in_ == sel.n_features
        np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0))
        np.testing.assert_allclose(
            refit.model.Y_pre_processing.named_steps["scaler"].mean_, Y_train.mean(axis=0)
        )
        # Matches a manual fit of an independently built pipeline on the same rows.
        manual = _template()
        manual.fit(X_train, Y_train)
        np.testing.assert_allclose(refit.model.X_f, manual.X_f)
        np.testing.assert_allclose(refit.model.Y_f, manual.Y_f)
    assert designs[0].n_features != designs[1].n_features


def test_refits_do_not_share_state(bank_and_designs):
    _, _, bank, designs = bank_and_designs
    template = _template()
    first = fit_design(template, bank, designs[0], train_rows=TRAIN)
    X_f_before = first.model.X_f.copy()
    pca_before = first.model.X_pre_processing.named_steps["pca"].components_.copy()

    # Fitting another design, and the same design on other rows, leaves it unchanged.
    fit_design(template, bank, designs[1], train_rows=TRAIN)
    other_rows = fit_design(template, bank, designs[0], train_rows=TRAIN[:50])
    np.testing.assert_array_equal(first.model.X_f, X_f_before)
    np.testing.assert_array_equal(
        first.model.X_pre_processing.named_steps["pca"].components_, pca_before
    )
    assert other_rows.model.X_f.shape[0] == 50

    # A repeat refit on identical inputs reproduces the first one.
    again = fit_design(template, bank, designs[0], train_rows=TRAIN)
    np.testing.assert_allclose(again.model.X_f, first.model.X_f)
    _assert_unfitted(template)


def test_heldout_transformation_and_prediction(bank_and_designs):
    obs, targets, bank, designs = bank_and_designs
    template = _template()
    for sel in designs:
        refit = fit_design(template, bank, sel, train_rows=TRAIN)
        X_held = refit.features(obs[HELDOUT])
        X_bank, Y_bank = bank.features(sel, HELDOUT)
        np.testing.assert_array_equal(X_held, X_bank)
        np.testing.assert_array_equal(Y_bank, targets[HELDOUT])

        samples = refit.model.predict(X_held, n_posts=200)
        assert samples.shape == (len(HELDOUT), 200, bank.n_targets)
        assert np.all(np.isfinite(samples))
        # Posterior means follow the held-out targets they were never fitted on.
        err = samples.mean(axis=1) - Y_bank
        spread = Y_bank.std(axis=0)
        assert np.all(np.sqrt(np.mean(err**2, axis=0)) < 0.5 * spread)

        # Same seed and features give the same draws.
        np.testing.assert_array_equal(refit.model.predict(X_held, n_posts=200), samples)

        with pytest.raises(ValueError, match="shape"):
            refit.features(obs[HELDOUT][:, :, :3])


def test_fit_design_requires_explicit_valid_rows_and_bank(bank_and_designs):
    obs, targets, bank, designs = bank_and_designs
    template = _template()
    with pytest.raises(TypeError):
        fit_design(template, bank, designs[0])  # no default training rows
    with pytest.raises(ValueError):
        fit_design(template, bank, designs[0], train_rows=[0, 0, 1])
    with pytest.raises(ValueError):
        fit_design(template, bank, designs[0], train_rows=[N_ROWS])
    with pytest.raises(TypeError):
        fit_design(template, bank, designs[0], train_rows=np.ones(N_ROWS, dtype=bool))
    with pytest.raises(TypeError, match="SimulationBank"):
        fit_design(template, obs, designs[0], train_rows=TRAIN)
    with pytest.raises(TypeError):
        fit_designs(template, bank, designs[0], train_rows=TRAIN)
    with pytest.raises(ValueError, match="empty"):
        fit_designs(template, bank, [], train_rows=TRAIN)
    _assert_unfitted(template)
