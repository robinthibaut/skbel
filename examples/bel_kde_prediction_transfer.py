"""Transfer KDE-mode BEL predictions as a data-only capsule.

The producer fits a small KDE-mode ``BEL`` on a deterministic toy data set and
exports a :class:`~skbel.learning.portable_kde.KDEPredictionCapsule` with fixed
bandwidths.  The consumer only receives bytes and a digest: it restores the
capsule and draws original-unit conditional samples for new predictor rows from
caller-owned uniforms, without the training data, fitted estimators or fitting.

Run with ``python examples/bel_kde_prediction_transfer.py``.  The consumer step
can run in another process or machine; only ``data`` and ``digest`` are needed.
"""

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skbel import BEL
from skbel.learning.portable_kde import KDEPredictionCapsule, export_kde


def toy_data():
    """A 4 x 4 grid: the first target is linear, the second curved in the predictors."""
    levels = np.array([-3.0, -1.0, 1.0, 3.0])
    first, second = np.meshgrid(levels, levels, indexing="ij")
    X = np.column_stack([first.ravel(), second.ravel()])
    Y = np.column_stack([X[:, 0], X[:, 1] ** 2 + X[:, 1] / 4])
    return X, Y


def produce():
    """Fit, export and serialize; returns the bytes and their digest."""
    X, Y = toy_data()
    bel = BEL(
        mode="kde",
        X_pre_processing=Pipeline([("scale", StandardScaler())]),
        Y_pre_processing=Pipeline([("scale", StandardScaler())]),
        regression_model=CCA(n_components=2, scale=False, max_iter=1000, tol=1e-10),
    )
    bel.fit(X, Y)
    capsule = export_kde(bel, bandwidths=[0.5, 0.5])
    return capsule.to_bytes(), capsule.sha256()


def consume(data, digest, X_new, n_samples=200, seed=0):
    """Restore from bytes and sample; uses a local Generator, not the global RNG."""
    capsule = KDEPredictionCapsule.from_bytes(data, expected_sha256=digest)
    rng = np.random.default_rng(seed)
    uniforms = rng.random((X_new.shape[0], n_samples, capsule.canonical_dim))
    return capsule, capsule.sample(X_new, uniforms)


def main():
    data, digest = produce()
    X_new = np.array([[-0.5, -0.5], [0.5, 0.5]])
    capsule, draws = consume(data, digest, X_new)
    print(f"{capsule!r}: {len(data)} bytes, sha256 {digest[:12]}...")
    for row, case in zip(X_new, draws, strict=True):
        quartiles = np.quantile(case, [0.25, 0.5, 0.75], axis=0)
        print(f"x = {row.tolist()}: target quartiles per column")
        print(np.array2string(quartiles.T, precision=3))


if __name__ == "__main__":
    main()
