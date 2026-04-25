"""Regenerate reference arrays for skbel/testing/test_basic.py.

Run from repo root:

    uv run python scripts/regenerate_test_references.py

Reference arrays are deterministic outputs of the BEL pipeline at fixed seed
123456. They drift whenever scikit-learn changes its CCA sign convention or
its underlying numerics. The regression test compares against them; when the
test fails for non-skbel reasons, run this script to refresh the baseline
and commit the updated `.npy` files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

from skbel import BEL
from skbel.tmaps import TransportMap

TESTING_DIR = Path(__file__).resolve().parents[1] / "skbel" / "testing"


def _bel() -> BEL:
    return BEL(
        X_pre_processing=Pipeline(
            [("scaler", StandardScaler(with_mean=False)), ("pca", PCA(n_components=50))]
        ),
        Y_pre_processing=Pipeline(
            [("scaler", StandardScaler(with_mean=False)), ("pca", PCA(n_components=30))]
        ),
        X_post_processing=Pipeline(
            [("normalizer", PowerTransformer(method="yeo-johnson", standardize=True))]
        ),
        Y_post_processing=Pipeline(
            [("normalizer", PowerTransformer(method="yeo-johnson", standardize=True))]
        ),
        regression_model=CCA(n_components=30),
    )


def regenerate_mvn() -> None:
    bel = _bel()
    bel.seed = 123456
    bel.mode = "mvn"
    bel.n_posts = 10

    X_train = np.load(TESTING_DIR / "X_train.npy")
    y_train = np.load(TESTING_DIR / "y_train.npy")
    X_test = np.load(TESTING_DIR / "X_test.npy")

    bel.fit(X=X_train, Y=y_train)
    bel.predict(X_obs=X_test)

    np.save(TESTING_DIR / "ref_mean.npy", bel.posterior_mean[0])
    np.save(TESTING_DIR / "ref_covariance.npy", bel.posterior_covariance[0])
    print(f"  ref_mean.npy:        shape={bel.posterior_mean[0].shape}")
    print(f"  ref_covariance.npy:  shape={bel.posterior_covariance[0].shape}")


def regenerate_kde() -> None:
    model = _bel()
    model.seed = 123456
    model.mode = "kde"
    model.n_posts = 10

    X_train = np.load(TESTING_DIR / "X_train.npy")
    y_train = np.load(TESTING_DIR / "y_train.npy")
    X_test = np.load(TESTING_DIR / "X_test.npy")

    model.fit(X=X_train, Y=y_train)
    y_samples = model.predict(X_test)

    np.save(TESTING_DIR / "y_samples_kde.npy", y_samples)
    print(f"  y_samples_kde.npy:   shape={y_samples.shape}")


def regenerate_tm() -> None:
    N = 1000
    X = scipy.stats.norm.rvs(size=(N, 2), random_state=123456)
    X[:, 1] += X[:, 0] ** 2

    nonmonotone = [
        [
            [],
            [0],
            [0, 0, "HF"],
            [0, 0, 0, "HF"],
            [0, 0, 0, 0, "HF"],
            [0, 0, 0, 0, 0, "HF"],
            [0, 0, 0, 0, 0, 0, "HF"],
            [0, 0, 0, 0, 0, 0, 0, "HF"],
        ]
    ]
    monotone = [[[1]] + ["iRBF 1"] * 8]

    tm = TransportMap(
        monotone=monotone,
        nonmonotone=nonmonotone,
        X=X,
        polynomial_type="probabilist's hermite",
        monotonicity="separable monotonicity",
        standardize_samples=True,
        workers=1,
    )
    tm.optimize()

    norm_samples = scipy.stats.norm.rvs(size=(500, 1), random_state=42)
    X_precalc = np.ones((500, 1)) * 2.32
    X_star = tm.inverse_map(X_precalc=X_precalc, Y=norm_samples)

    np.save(TESTING_DIR / "x_star_tm.npy", X_star)
    print(f"  x_star_tm.npy:       shape={X_star.shape}")


if __name__ == "__main__":
    print("Regenerating MVN references...")
    regenerate_mvn()
    print("Regenerating KDE references...")
    regenerate_kde()
    print("Regenerating TM reference...")
    regenerate_tm()
    print("Done.")
