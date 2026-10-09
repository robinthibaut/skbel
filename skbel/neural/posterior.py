#  Copyright (c) 2026. Robin Thibaut, Ghent University

"""Conditional mixture-density posterior, equal-weight ensemble and quantile recalibration.

Arrays in, joint posterior draws out. Nothing here knows about any physical
model, prior, site or score: predictors ``X`` are ``(n, features)``, targets
``Y`` are ``(n, D)`` real vectors in whatever (latent) coordinates the caller
chose, and draws come back in the same coordinates. NumPy and scikit-learn's
estimator base class are the only third-party imports at module level;
PyTorch is imported on demand, so the recalibrator and the ensemble sampler of
precomputed mixtures run without it.

:class:`NeuralPosterior`
    The scikit-learn estimator: ``fit(X, Y)`` trains ``n_members``
    independently seeded :class:`MixturePosterior` members and ``sample``
    returns joint draws ``(cases, draws, targets)`` of their equal-weight
    mixture. It is cloneable, so it can be the template of
    :func:`skbel.design.fit_design`.
:class:`MixturePosterior`
    Fit-only ``StandardScaler`` then randomized ``PCA(n_pca)`` predictors, an
    MLP of SiLU hidden layers (float64, CPU) to a ``n_components`` Gaussian
    mixture over the joint target vector, either ``"diagonal"`` (log-scales
    ``lo + (hi - lo) * sigmoid(raw)``) or ``"full"`` (Cholesky ``L L^T``: that
    diagonal plus unrestricted strictly-lower entries, joint log density by one
    triangular solve). Adam on the mean joint negative log density; the last
    ``n_validation`` rows (in the given order) are internal validation with
    early stopping (patience, minimum improvement); the transforms and network
    are then refitted on every row for the chosen epoch count with the same
    PCA/init/shuffle seeds. No jitter, clipping, value repair, masking or
    redraw; a non-finite loss, parameter or mixture output raises.
:class:`EqualWeightEnsemble`
    Each joint draw first picks one member uniformly (probability ``1/M``),
    then one component of that member by its weights, then one Gaussian
    vector of that component, shared by all targets. Member and component
    labels are returned with every draw.
:class:`QuantileRecalibrator`
    One strictly increasing piecewise-linear map per (target, group) fitted on
    randomized PIT values of held-out draws against their truths, applied as
    a deterministic within-case marginal rank transform of the same draws.

Every sampler uses a per-case generator ``SeedSequence(seed, spawn_key=(case_id,))``,
so draws do not depend on the order, set or chunking of the evaluated cases;
NumPy's legacy and PyTorch's global random states are checked unchanged.
Persistence is ``.npz`` plus JSON strings, read with ``allow_pickle=False``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator

__all__ = [
    "CHUNK",
    "COVARIANCES",
    "ENSEMBLE_FORMAT",
    "FIT_SEEDS",
    "MODEL_FORMAT",
    "RECALIBRATOR_FORMAT",
    "EqualWeightEnsemble",
    "MixturePosterior",
    "MixtureSettings",
    "NeuralPosterior",
    "QuantileRecalibrator",
    "empirical_quantile",
    "member_seeds",
    "mid_cdf_knots",
    "midrank_probabilities",
    "pit_from_counts",
    "pit_grid",
    "pit_grid_indices",
    "pit_uniforms",
    "randomized_pit",
    "sample_ensemble_latent",
    "sample_mixture_latent",
]

COVARIANCES = ("diagonal", "full")
FIT_SEEDS = ("pca", "init", "shuffle")
MODEL_FORMAT = "neural_posterior.MixturePosterior/1"
ENSEMBLE_FORMAT = "neural_posterior.EqualWeightEnsemble/1"
RECALIBRATOR_FORMAT = "neural_posterior.QuantileRecalibrator/1"
CHUNK = 1024  # cases per network pass when predicting; bounds memory only
PREPROCESS_CHUNK = 16384  # rows per predictor transform; bounds memory only
_PIT_DENOMINATOR_LIMIT = 2**51  # exact PIT denominators stay below this
_INSTALL_HINT = "install the optional extra with: pip install 'skbel[neural]'"


@dataclass(frozen=True)
class MixtureSettings:
    """Mixture-density network, preprocessing and training rule (defaults: the fixed MDN)."""

    covariance: str = "full"
    n_components: int = 20
    hidden: tuple[int, ...] = (128, 128, 128)
    n_pca: int = 64
    learning_rate: float = 1e-3
    batch_size: int = 512
    max_epochs: int = 200
    n_validation: int = 1024
    patience: int = 20
    min_improvement: float = 1e-4
    log_scale_bounds: tuple[float, float] = (-7.0, 3.0)

    def __post_init__(self) -> None:
        if self.covariance not in COVARIANCES:
            raise ValueError(f"covariance must be one of {COVARIANCES}")
        for name in (
            "n_components",
            "n_pca",
            "batch_size",
            "max_epochs",
            "n_validation",
            "patience",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int")
        hidden = tuple(self.hidden)
        if not hidden or any(
            isinstance(h, bool) or not isinstance(h, int) or h < 1 for h in hidden
        ):
            raise ValueError("hidden must be a non-empty tuple of positive ints")
        object.__setattr__(self, "hidden", hidden)
        lo, hi = (float(v) for v in self.log_scale_bounds)
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError("log_scale_bounds must be finite with lower < upper")
        object.__setattr__(self, "log_scale_bounds", (lo, hi))
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(self.min_improvement) or self.min_improvement < 0:
            raise ValueError("min_improvement must be non-negative and finite")

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> MixtureSettings:
        names = {f.name for f in fields(cls)}
        if set(d) != names:
            raise ValueError(f"settings keys {sorted(d)} differ from {sorted(names)}")
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


# ---------------------------------------------------------------------------
# Lazy optional imports and global-state guards
# ---------------------------------------------------------------------------


def _torch():
    """Import PyTorch on demand; a missing install names the optional extra."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            f"training and evaluating neural posteriors needs PyTorch; {_INSTALL_HINT}"
        ) from exc
    return torch


def _global_rng_state() -> tuple:
    """Hashable snapshot of NumPy's legacy global random state (read, never set)."""
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return name, keys.tobytes(), int(pos), int(has_gauss), float(cached)


class _GlobalStateGuard:
    """Raises on exit if NumPy's legacy or (when loaded) PyTorch's global RNG state changed."""

    def __init__(self, what: str, torch=None) -> None:
        self.what = what
        self.torch = torch

    def __enter__(self):
        self.np_state = _global_rng_state()
        self.torch_state = None if self.torch is None else self.torch.random.get_rng_state()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            return
        if _global_rng_state() != self.np_state or (
            self.torch is not None
            and not self.torch.equal(self.torch.random.get_rng_state(), self.torch_state)
        ):
            raise RuntimeError(f"{self.what} changed a global random state")


def _case_rng(seed: int, case_id: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(int(case_id),)))


def _check_seed(seed, name: str = "seed") -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"{name} must be a non-negative int")
    return int(seed)


def _check_positive(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive int")
    return int(value)


def _case_ids(case_ids, n: int) -> np.ndarray:
    ids = np.asarray(case_ids)
    if ids.shape != (n,) or ids.dtype.kind not in "iu":
        raise ValueError(f"case_ids must be {n} integers")
    ids = ids.astype(np.int64)
    if np.any(ids < 0) or np.unique(ids).size != n:
        raise ValueError("case_ids must be distinct non-negative integers")
    return ids


# ---------------------------------------------------------------------------
# Predictor preprocessing: fit-only StandardScaler + randomized PCA
# ---------------------------------------------------------------------------


def fit_preprocessing(x, n_pca: int, seed: int) -> dict[str, np.ndarray]:
    """``StandardScaler`` then randomized ``PCA(n_pca)`` on ``x``; returns the fitted arrays."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x)
    pca = PCA(n_components=n_pca, svd_solver="randomized", random_state=seed)
    pca.fit(scaler.transform(x))
    return {
        "scaler_mean": scaler.mean_.astype(float),
        "scaler_scale": scaler.scale_.astype(float),
        "pca_mean": pca.mean_.astype(float),
        "pca_components": pca.components_.astype(float),
        "pca_explained_variance": pca.explained_variance_.astype(float),
    }


def apply_preprocessing(x, pre: dict[str, np.ndarray], chunk: int = PREPROCESS_CHUNK) -> np.ndarray:
    """Stored scaler then PCA projection (``whiten=False``), ``chunk`` rows at a time.

    Row blocks only bound memory for disk-backed inputs; a single block is the
    plain one-shot transform.
    """
    if x.ndim != 2 or x.shape[1] != pre["scaler_mean"].shape[0]:
        raise ValueError("predictors do not match the fitted preprocessing")
    out = np.empty((x.shape[0], pre["pca_components"].shape[0]))
    for a in range(0, x.shape[0], chunk):
        z = (np.asarray(x[a : a + chunk], dtype=float) - pre["scaler_mean"]) / pre["scaler_scale"]
        out[a : a + chunk] = (z - pre["pca_mean"]) @ pre["pca_components"].T
    return out


# ---------------------------------------------------------------------------
# Network, mixture parameters and joint log density
# ---------------------------------------------------------------------------


def n_outputs(n_targets: int, settings: MixtureSettings) -> int:
    """Per component: a logit, D means and D log-scales (diagonal) or D(D+1)/2 Cholesky entries."""
    k, d = settings.n_components, n_targets
    if settings.covariance == "diagonal":
        return k * (1 + 2 * d)
    return k * (1 + d * (d + 3) // 2)


def build_network(n_inputs: int, n_targets: int, settings: MixtureSettings, init_seed: int):
    """MLP of SiLU hidden layers to the mixture parameters; local seeded float64 init."""
    torch = _torch()
    nn = torch.nn
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        layers, width = [], n_inputs
        for h in settings.hidden:
            layers += [nn.Linear(width, h, dtype=torch.float64), nn.SiLU()]
            width = h
        layers.append(nn.Linear(width, n_outputs(n_targets, settings), dtype=torch.float64))
        return nn.Sequential(*layers)


def cholesky_factors(raw_diag, lower, settings: MixtureSettings):
    """``L`` (n, K, D, D): diagonal ``exp(lo + (hi - lo) sigmoid(raw))``, strict-lower entries free.

    Strictly lower entries are filled in row-major ``tril_indices(D, D, -1)`` order.
    """
    torch = _torch()
    lo, hi = settings.log_scale_bounds
    chol = torch.diag_embed(torch.exp(lo + (hi - lo) * torch.sigmoid(raw_diag)))
    d = raw_diag.shape[-1]
    if d > 1:
        rows, cols = torch.tril_indices(d, d, -1)
        strict = torch.zeros_like(chol)
        strict[..., rows, cols] = lower
        chol = chol + strict
    return chol


def mixture_parameters(net, x, n_targets: int, settings: MixtureSettings):
    """``(log_weights (n, K), means (n, K, D), shape)``; shape is log-scales or Cholesky ``L``."""
    torch = _torch()
    k, d = settings.n_components, n_targets
    out = net(x)
    logits = out[:, :k]
    means = out[:, k : k + k * d].reshape(-1, k, d)
    if settings.covariance == "diagonal":
        raw = out[:, k + k * d :].reshape(-1, k, d)
        lo, hi = settings.log_scale_bounds
        shape = lo + (hi - lo) * torch.sigmoid(raw)
    else:
        raw_diag = out[:, k + k * d : k + 2 * k * d].reshape(-1, k, d)
        lower = out[:, k + 2 * k * d :].reshape(-1, k, d * (d - 1) // 2)
        shape = cholesky_factors(raw_diag, lower, settings)
    return torch.log_softmax(logits, dim=1), means, shape


def mixture_log_prob(log_weights, means, shape, z, covariance: str):
    """Joint log density of ``z`` (n, D) under the diagonal or full-covariance mixture."""
    torch = _torch()
    if covariance == "diagonal":
        std = (z[:, None, :] - means) * torch.exp(-shape)
        comp = (-0.5 * std.square() - shape - 0.5 * math.log(2.0 * math.pi)).sum(dim=2)
        return torch.logsumexp(log_weights + comp, dim=1)
    if covariance != "full":
        raise ValueError(f"covariance must be one of {COVARIANCES}")
    d = z.shape[1]
    resid = (z[:, None, :] - means).unsqueeze(-1)
    y = torch.linalg.solve_triangular(shape, resid, upper=False).squeeze(-1)
    log_det = torch.log(torch.diagonal(shape, dim1=-2, dim2=-1)).sum(dim=-1)
    comp = -0.5 * y.square().sum(dim=-1) - log_det - 0.5 * d * math.log(2.0 * math.pi)
    return torch.logsumexp(log_weights + comp, dim=1)


class EarlyStop:
    """Patience rule on the validation loss: best epoch, improvement > min_improvement."""

    def __init__(self, patience: int, min_improvement: float) -> None:
        self.patience = patience
        self.min_improvement = min_improvement
        self.best = math.inf
        self.best_epoch = 0
        self.stale = 0

    def update(self, epoch: int, loss: float) -> bool:
        """Record epoch ``epoch`` (1-based); return True when training must stop."""
        if not math.isfinite(loss):
            raise RuntimeError(f"validation loss is not finite at epoch {epoch}")
        if loss < self.best - self.min_improvement:
            self.best, self.best_epoch, self.stale = loss, epoch, 0
        else:
            self.stale += 1
        return self.stale >= self.patience


def _finite_parameters(net) -> None:
    torch = _torch()
    for name, p in net.named_parameters():
        if not bool(torch.isfinite(p).all()):
            raise RuntimeError(f"network parameter {name} is not finite")


def train_network(
    x, z, settings: MixtureSettings, seeds: dict, n_epochs: int, x_val=None, z_val=None
):
    """Adam on the mean joint NLL for up to ``n_epochs``; optional per-epoch validation and stop."""
    torch = _torch()
    n_targets = z.shape[1]
    net = build_network(x.shape[1], n_targets, settings, seeds["init"])
    opt = torch.optim.Adam(net.parameters(), lr=settings.learning_rate)
    xt = torch.as_tensor(x, dtype=torch.float64)
    zt = torch.as_tensor(z, dtype=torch.float64)
    if x_val is not None:
        xv = torch.as_tensor(x_val, dtype=torch.float64)
        zv = torch.as_tensor(z_val, dtype=torch.float64)
    rng = np.random.default_rng(seeds["shuffle"])
    stop = EarlyStop(settings.patience, settings.min_improvement)
    train_loss, val_loss = [], []
    for epoch in range(1, n_epochs + 1):
        net.train()
        order = torch.as_tensor(rng.permutation(x.shape[0]))
        total = 0.0
        for start in range(0, x.shape[0], settings.batch_size):
            idx = order[start : start + settings.batch_size]
            loss = -mixture_log_prob(
                *mixture_parameters(net, xt[idx], n_targets, settings), zt[idx], settings.covariance
            ).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * idx.shape[0]
        train_loss.append(total / x.shape[0])
        if not math.isfinite(train_loss[-1]):
            raise RuntimeError(f"training loss is not finite at epoch {epoch}")
        _finite_parameters(net)
        if x_val is not None:
            net.eval()
            with torch.no_grad():
                v = -mixture_log_prob(
                    *mixture_parameters(net, xv, n_targets, settings), zv, settings.covariance
                ).mean()
            val_loss.append(float(v))
            if stop.update(epoch, val_loss[-1]):
                break
    return net, {"train_loss": train_loss, "validation_loss": val_loss, "stop": stop}


# ---------------------------------------------------------------------------
# Joint sampling of precomputed mixtures (NumPy only)
# ---------------------------------------------------------------------------


def _component_vectors(means, shape, labels, eps, covariance: str) -> np.ndarray:
    if covariance == "diagonal":
        return means[labels] + shape[labels] * eps
    return means[labels] + np.einsum("nij,nj->ni", shape[labels], eps)


def _mixture_shape(mixture: dict) -> tuple[str, np.ndarray]:
    if ("scales" in mixture) == ("chol" in mixture):
        raise ValueError("a mixture holds exactly one of 'scales' (diagonal) or 'chol' (full)")
    return ("diagonal", mixture["scales"]) if "scales" in mixture else ("full", mixture["chol"])


def sample_mixture_latent(mixture: dict, case_ids, n_draws: int, seed: int):
    """Joint draws ``(n, n_draws, D)`` and component labels of ONE mixture per case.

    Per case ``rng.choice(K, n_draws, p=w / w.sum())`` then one standard normal
    vector per draw (``mu + s * eps`` or ``mu + L eps``).

    :param mixture: ``weights (n, K)``, ``means (n, K, D)`` and either
        ``scales (n, K, D)`` or ``chol (n, K, D, D)``, as returned by
        :meth:`MixturePosterior.predict_mixture`.
    :param case_ids: ``(n,)`` distinct non-negative integer case IDs.
    :param n_draws: draws per case.
    :param seed: non-negative int; case ``c`` uses ``SeedSequence(seed, spawn_key=(c,))``.
    """
    w, mu = mixture["weights"], mixture["means"]
    covariance, shape = _mixture_shape(mixture)
    n, k, d = mu.shape
    ids = _case_ids(case_ids, n)
    z = np.empty((n, n_draws, d))
    labels = np.empty((n, n_draws), dtype=np.int64)
    for i, cid in enumerate(ids):
        rng = _case_rng(seed, cid)
        labels[i] = rng.choice(k, size=n_draws, p=w[i] / w[i].sum())
        eps = rng.standard_normal((n_draws, d))
        z[i] = _component_vectors(mu[i], shape[i], labels[i], eps, covariance)
    return z, labels


def sample_ensemble_latent(mixtures: list[dict], case_ids, n_draws: int, seed: int) -> dict:
    """Equal-weight joint draws of M member mixtures of the same cases.

    Per case, with its own generator: ``member = rng.integers(0, M, n_draws)``
    (probability exactly ``1/M`` each), ``u = rng.random(n_draws)``, component
    = the first index whose cumulative member weight exceeds ``u * total``
    (zero-weight components are never chosen), then ``eps =
    rng.standard_normal((n_draws, D))`` and the component vector. Returns
    ``latent``, ``member`` and ``component``.
    """
    if not mixtures:
        raise ValueError("an ensemble needs at least one member")
    seed = _check_seed(seed)
    if isinstance(n_draws, bool) or not isinstance(n_draws, int) or n_draws < 1:
        raise ValueError("n_draws must be a positive int")
    kinds = {_mixture_shape(m)[0] for m in mixtures}
    if len(kinds) != 1:
        raise ValueError("ensemble members must share one covariance type")
    covariance = kinds.pop()
    w = np.stack([m["weights"] for m in mixtures])  # (M, n, K)
    mu = np.stack([m["means"] for m in mixtures])  # (M, n, K, D)
    shape = np.stack([_mixture_shape(m)[1] for m in mixtures])
    n_members, n, k, d = mu.shape
    if w.shape != (n_members, n, k):
        raise ValueError("member weights and means disagree in shape")
    for name, value in (("weights", w), ("means", mu), ("shape", shape)):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"mixture {name} are not finite")
    if np.any(w < 0) or np.any(w.sum(axis=2) <= 0):
        raise ValueError("mixture weights must be non-negative with a positive total")
    ids = _case_ids(case_ids, n)
    cdf = np.cumsum(w, axis=2)  # (M, n, K)
    z = np.empty((n, n_draws, d))
    member = np.empty((n, n_draws), dtype=np.int64)
    component = np.empty((n, n_draws), dtype=np.int64)
    for i, cid in enumerate(ids):
        rng = _case_rng(seed, cid)
        m = rng.integers(0, n_members, size=n_draws)
        u = rng.random(n_draws)
        rows = cdf[m, i]  # (n_draws, K)
        c = np.sum(rows <= (u * rows[:, -1])[:, None], axis=1)
        eps = rng.standard_normal((n_draws, d))
        member[i], component[i] = m, c
        if covariance == "diagonal":
            z[i] = mu[m, i, c] + shape[m, i, c] * eps
        else:
            z[i] = mu[m, i, c] + np.einsum("nij,nj->ni", shape[m, i, c], eps)
    return {"latent": z, "member": member, "component": component}


# ---------------------------------------------------------------------------
# One mixture posterior
# ---------------------------------------------------------------------------


class MixturePosterior:
    """Conditional Gaussian-mixture posterior of a target vector given predictors."""

    def __init__(self, settings: MixtureSettings | None = None) -> None:
        self.settings = MixtureSettings() if settings is None else settings
        self.net = None
        self.preprocessing: dict[str, np.ndarray] | None = None
        self.n_inputs: int | None = None
        self.n_targets: int | None = None
        self.history: dict | None = None

    # -- fitting ------------------------------------------------------------

    def fit(self, X, Z, ids, seeds: dict) -> MixturePosterior:
        """Epoch choice on the LAST ``n_validation`` rows, then the refit on every row.

        :param X: ``(n, features)`` predictors, read only (a memory map is fine).
        :param Z: ``(n, D)`` finite targets.
        :param ids: ``(n,)`` strictly increasing row IDs (recorded; order defines validation).
        :param seeds: ``pca`` (fits 32 bits), ``init`` and ``shuffle`` non-negative ints.
        """
        torch = _torch()
        s = self.settings
        if X.ndim != 2:
            raise ValueError("X must be (rows, features)")
        Z = np.asarray(Z, dtype=float)
        ids = np.asarray(ids)
        n = X.shape[0]
        if Z.ndim != 2 or Z.shape[0] != n or ids.shape != (n,) or ids.dtype.kind not in "iu":
            raise ValueError("Z must be (rows, targets) and ids one integer per row")
        if not np.all(np.diff(ids) > 0):
            raise ValueError("ids must be strictly increasing")
        if not np.all(np.isfinite(Z)):
            raise ValueError("targets must be finite; nothing is nudged or clipped")
        if set(seeds) != set(FIT_SEEDS):
            raise ValueError(f"seeds must be exactly {FIT_SEEDS}")
        for name in FIT_SEEDS:
            _check_seed(seeds[name], name)
        if seeds["pca"] >= 2**32:
            raise ValueError("the PCA seed must fit in 32 bits")
        if s.n_validation >= n:
            raise ValueError("n_validation must leave training rows")
        train, val = slice(0, n - s.n_validation), slice(n - s.n_validation, n)
        with _GlobalStateGuard("training", torch):
            t0 = time.perf_counter()
            pre_v = fit_preprocessing(X[train], s.n_pca, seeds["pca"])
            _, hist = train_network(
                apply_preprocessing(X[train], pre_v),
                Z[train],
                s,
                seeds,
                s.max_epochs,
                apply_preprocessing(X[val], pre_v),
                Z[val],
            )
            selection_seconds = time.perf_counter() - t0
            chosen = hist["stop"].best_epoch
            t0 = time.perf_counter()
            pre = fit_preprocessing(X, s.n_pca, seeds["pca"])
            net, refit = train_network(apply_preprocessing(X, pre), Z, s, seeds, chosen)
            refit_seconds = time.perf_counter() - t0
        self.net, self.preprocessing = net, pre
        self.n_inputs, self.n_targets = int(X.shape[1]), int(Z.shape[1])
        self.history = {
            "n_rows": int(n),
            "row_id_range": [int(ids[0]), int(ids[-1])],
            "validation_rows": {"n": s.n_validation, "id_range": [int(ids[val][0]), int(ids[-1])]},
            "seeds": {k: int(seeds[k]) for k in FIT_SEEDS},
            "selection_train_loss": hist["train_loss"],
            "selection_validation_loss": hist["validation_loss"],
            "epochs_run": len(hist["validation_loss"]),
            "chosen_epochs": int(chosen),
            "best_validation_loss": hist["stop"].best,
            "early_stopped": len(hist["validation_loss"]) < s.max_epochs,
            "refit_train_loss": refit["train_loss"],
            "final_refit_train_loss": refit["train_loss"][-1],
            "n_parameters": int(sum(p.numel() for p in net.parameters())),
            "selection_seconds": selection_seconds,
            "refit_seconds": refit_seconds,
            "torch_threads": torch.get_num_threads(),
            "loss": "mean joint negative log density of the targets (nats)",
        }
        return self

    def _check_fitted(self) -> None:
        if self.net is None:
            raise RuntimeError("the posterior is not fitted or loaded")

    # -- evaluation ---------------------------------------------------------

    def predict_mixture(self, X, chunk: int = CHUNK) -> dict[str, np.ndarray]:
        """Per-case ``weights (n, K)``, ``means (n, K, D)`` and ``scales`` or ``chol``; float64."""
        self._check_fitted()
        torch = _torch()
        if chunk < 1:
            raise ValueError("chunk must be positive")
        s = self.settings
        key = "scales" if s.covariance == "diagonal" else "chol"
        self.net.eval()
        xt = torch.as_tensor(apply_preprocessing(X, self.preprocessing), dtype=torch.float64)
        parts = {"weights": [], "means": [], key: []}
        with torch.no_grad():
            for a in range(0, xt.shape[0], chunk):
                log_w, means, shape = mixture_parameters(
                    self.net, xt[a : a + chunk], self.n_targets, s
                )
                parts["weights"].append(torch.exp(log_w).numpy())
                parts["means"].append(means.numpy())
                parts[key].append((torch.exp(shape) if key == "scales" else shape).numpy())
        out = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
        for name, value in out.items():
            if not np.all(np.isfinite(value)):
                raise RuntimeError(f"mixture {name} are not finite")
        return out

    def log_prob(self, X, Z) -> np.ndarray:
        """Joint log density ``(n,)`` of targets ``Z`` given predictors ``X``."""
        self._check_fitted()
        torch = _torch()
        self.net.eval()
        xt = torch.as_tensor(apply_preprocessing(X, self.preprocessing), dtype=torch.float64)
        zt = torch.as_tensor(np.asarray(Z, dtype=float), dtype=torch.float64)
        with torch.no_grad():
            lp = mixture_log_prob(
                *mixture_parameters(self.net, xt, self.n_targets, self.settings),
                zt,
                self.settings.covariance,
            )
        return lp.numpy()

    def sample(self, X, case_ids, n_draws: int, seed: int):
        """Joint draws ``(n, n_draws, D)`` and component labels ``(n, n_draws)``."""
        torch = _torch()
        seed = _check_seed(seed)
        with _GlobalStateGuard("sampling", torch):
            mixture = self.predict_mixture(X)
            return sample_mixture_latent(mixture, case_ids, n_draws, seed)

    # -- persistence --------------------------------------------------------

    def save(self, path) -> None:
        """Write a NEW ``.npz``: weights, preprocessing, settings and history (no pickles)."""
        self._check_fitted()
        arrays = {f"weight/{k}": v.detach().numpy() for k, v in self.net.state_dict().items()}
        arrays |= {f"pre/{k}": v for k, v in self.preprocessing.items()}
        arrays["format"] = np.array(MODEL_FORMAT)
        arrays["settings_json"] = np.array(json.dumps(self.settings.to_dict()))
        arrays["history_json"] = np.array(json.dumps(self.history))
        arrays["n_inputs"] = np.int64(self.n_inputs)
        arrays["n_targets"] = np.int64(self.n_targets)
        with open(Path(path), "xb") as f:
            np.savez(f, **arrays)

    @classmethod
    def load(cls, path) -> MixturePosterior:
        """Rebuild a persisted posterior without any training."""
        torch = _torch()
        with np.load(Path(path), allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        if str(arrays.get("format")) != MODEL_FORMAT:
            raise ValueError(f"{path} is not a {MODEL_FORMAT} file")
        self = cls(MixtureSettings.from_dict(json.loads(str(arrays["settings_json"]))))
        self.history = json.loads(str(arrays["history_json"]))
        self.n_inputs = int(arrays["n_inputs"])
        self.n_targets = int(arrays["n_targets"])
        self.preprocessing = {k[4:]: v for k, v in arrays.items() if k.startswith("pre/")}
        if self.preprocessing["pca_components"].shape != (self.settings.n_pca, self.n_inputs):
            raise ValueError("stored PCA does not match the stored predictor count")
        net = build_network(self.settings.n_pca, self.n_targets, self.settings, 0)
        state = {k[7:]: torch.as_tensor(v) for k, v in arrays.items() if k.startswith("weight/")}
        net.load_state_dict(state, strict=True)
        self.net = net
        return self


# ---------------------------------------------------------------------------
# Equal-weight ensemble
# ---------------------------------------------------------------------------


class EqualWeightEnsemble:
    """Equal-weight mixture of fitted ``MixturePosterior`` members (member order is retained)."""

    def __init__(self, members) -> None:
        members = list(members)
        if not members:
            raise ValueError("an ensemble needs at least one member")
        for m in members:
            m._check_fitted()
        first = members[0]
        for m in members[1:]:
            if (m.settings, m.n_inputs, m.n_targets) != (
                first.settings,
                first.n_inputs,
                first.n_targets,
            ):
                raise ValueError("ensemble members must share settings, predictors and targets")
        self.members = members

    @property
    def weights(self) -> np.ndarray:
        return np.full(len(self.members), 1.0 / len(self.members))

    def predict_mixtures(self, X, chunk: int = CHUNK) -> list[dict]:
        return [m.predict_mixture(X, chunk) for m in self.members]

    def sample(self, X, case_ids, n_draws: int, seed: int, chunk: int = CHUNK, *, mixtures=False):
        """Joint ``latent`` draws ``(n, n_draws, D)`` with ``member`` and ``component`` labels.

        Cases go ``chunk`` at a time through every member; each case keeps its
        own generator, so the draws do not depend on the chunking. With
        ``mixtures=True`` the stacked member mixtures ``(M, n, ...)`` are returned too.
        """
        torch = _torch()
        seed = _check_seed(seed)
        n = X.shape[0]
        ids = _case_ids(case_ids, n)
        d = self.members[0].n_targets
        out = {
            "latent": np.empty((n, n_draws, d)),
            "member": np.empty((n, n_draws), dtype=np.int64),
            "component": np.empty((n, n_draws), dtype=np.int64),
        }
        kept: list[list[dict]] = []
        with _GlobalStateGuard("ensemble sampling", torch):
            for a in range(0, n, chunk):
                part = self.predict_mixtures(X[a : a + chunk], chunk)
                draws = sample_ensemble_latent(part, ids[a : a + chunk], n_draws, seed)
                for k, v in draws.items():
                    out[k][a : a + chunk] = v
                if mixtures:
                    kept.append(part)
        if mixtures:
            keys = kept[0][0].keys()
            out["mixtures"] = {
                k: np.stack(
                    [np.concatenate([p[j][k] for p in kept]) for j in range(len(self.members))]
                )
                for k in keys
            }
        return out

    def save(self, directory) -> list[str]:
        """Write ``member_<i>.npz`` and ``ensemble.json`` into a NEW directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        names = [f"member_{i}.npz" for i in range(len(self.members))]
        for name, m in zip(names, self.members, strict=True):
            m.save(directory / name)
        with open(directory / "ensemble.json", "x") as f:
            json.dump({"format": ENSEMBLE_FORMAT, "members": names, "weights": "equal"}, f)
        return names

    @classmethod
    def load(cls, paths) -> EqualWeightEnsemble:
        """Members from ``paths`` in that order, or from an ``ensemble.json`` directory.

        A single ``str`` or ``Path`` that is not a directory names one member file.
        A directory record must name equal weights and list member files that
        are plain ``.npz`` names inside that directory.
        """
        if isinstance(paths, (str, Path)) and Path(paths).is_dir():
            with open(Path(paths) / "ensemble.json") as f:
                meta = json.load(f)
            if meta.get("format") != ENSEMBLE_FORMAT:
                raise ValueError(f"{paths} is not a {ENSEMBLE_FORMAT} directory")
            names = meta.get("members")
            if meta.get("weights") != "equal" or not isinstance(names, list):
                raise ValueError(f"{paths}: ensemble.json must list members with equal weights")
            for name in names:
                if (
                    not isinstance(name, str)
                    or Path(name).name != name
                    or name in (".", "..")
                    or not name.endswith(".npz")
                ):
                    raise ValueError(f"{paths}: member {name!r} is not a plain .npz file name")
            paths = [Path(paths) / name for name in names]
        elif isinstance(paths, (str, Path)):
            paths = [Path(paths)]
        paths = [Path(p) for p in paths]
        for p in paths:
            if not p.is_file():
                raise FileNotFoundError(f"member file not found: {p}")
        return cls([MixturePosterior.load(p) for p in paths])


# ---------------------------------------------------------------------------
# Scikit-learn estimator
# ---------------------------------------------------------------------------


def _seed_word(entropy: int, key: tuple, bits32: bool) -> int:
    seq = np.random.SeedSequence(entropy, spawn_key=key)
    if bits32:
        return int(seq.generate_state(1, np.uint32)[0])
    return int(seq.generate_state(1, np.uint64)[0] >> np.uint64(1))


def member_seeds(random_state: int, n_members: int) -> list[dict[str, int]]:
    """Independent fit seeds of ``n_members`` ensemble members from one integer.

    Member ``m`` stream ``i`` (``pca``, ``init``, ``shuffle`` in that order) is
    the first word of ``SeedSequence(random_state, spawn_key=(m, i))``: 32 bits
    for ``pca`` (its fitting limit), 63 bits otherwise. Each member's seeds
    depend only on ``(random_state, m)``, so adding members never changes
    earlier ones.
    """
    random_state = _check_seed(random_state, "random_state")
    n_members = _check_positive(n_members, "n_members")
    return [
        {s: _seed_word(random_state, (m, i), s == "pca") for i, s in enumerate(FIT_SEEDS)}
        for m in range(n_members)
    ]


class NeuralPosterior(BaseEstimator):
    """Equal-weight ensemble of independently seeded mixture-density posteriors.

    A scikit-learn estimator with array ``fit(X, Y)`` and joint posterior
    draws from :meth:`sample`. Every member is a :class:`MixturePosterior`
    with the same settings and its own ``pca``/``init``/``shuffle`` seeds;
    the members are fitted one after another on the same rows. Draws pick a
    member with probability ``1/n_members``, then a component, then one
    Gaussian vector shared by all targets (:func:`sample_ensemble_latent`).

    The mixture parameters are those of :class:`MixtureSettings` (same
    defaults). Parameters are stored unchanged and validated by :meth:`fit`,
    so :func:`sklearn.base.clone` gives an unfitted copy with the same
    settings and seeds, as :func:`skbel.design.fit_design` requires.

    :param n_members: number of ensemble members.
    :param random_state: non-negative int from which :func:`member_seeds`
        derives the member seeds; ignored when ``member_seeds`` is given.
        ``RandomState`` objects and ``None`` are refused: the fit is fully
        determined by the parameters.
    :param member_seeds: optional explicit seeds, one dict with exactly the
        keys ``pca``, ``init`` and ``shuffle`` per member, to reproduce
        particular members.

    Fitted attributes: ``ensemble_`` (the :class:`EqualWeightEnsemble`),
    ``members_``, ``member_seeds_``, ``n_features_in_`` and ``n_targets_``.
    Training needs the optional PyTorch extra (``pip install 'skbel[neural]'``).
    """

    def __init__(
        self,
        n_members: int = 5,
        *,
        covariance: str = "full",
        n_components: int = 20,
        hidden: tuple[int, ...] = (128, 128, 128),
        n_pca: int = 64,
        learning_rate: float = 1e-3,
        batch_size: int = 512,
        max_epochs: int = 200,
        n_validation: int = 1024,
        patience: int = 20,
        min_improvement: float = 1e-4,
        log_scale_bounds: tuple[float, float] = (-7.0, 3.0),
        random_state: int | None = 0,
        member_seeds: list[dict] | None = None,
    ) -> None:
        self.n_members = n_members
        self.covariance = covariance
        self.n_components = n_components
        self.hidden = hidden
        self.n_pca = n_pca
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.n_validation = n_validation
        self.patience = patience
        self.min_improvement = min_improvement
        self.log_scale_bounds = log_scale_bounds
        self.random_state = random_state
        self.member_seeds = member_seeds

    # -- parameters ---------------------------------------------------------

    def settings(self) -> MixtureSettings:
        """The validated :class:`MixtureSettings` of every member."""
        return MixtureSettings(
            covariance=self.covariance,
            n_components=self.n_components,
            hidden=tuple(self.hidden),
            n_pca=self.n_pca,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            n_validation=self.n_validation,
            patience=self.patience,
            min_improvement=self.min_improvement,
            log_scale_bounds=tuple(self.log_scale_bounds),
        )

    def _seeds(self) -> list[dict[str, int]]:
        n_members = _check_positive(self.n_members, "n_members")
        if self.member_seeds is None:
            return member_seeds(self.random_state, n_members)
        seeds = list(self.member_seeds)
        if len(seeds) != n_members:
            raise ValueError(f"member_seeds must hold {n_members} seed dicts")
        out = []
        for s in seeds:
            if not isinstance(s, dict) or set(s) != set(FIT_SEEDS):
                raise ValueError(f"each member seed dict must have exactly the keys {FIT_SEEDS}")
            out.append({k: _check_seed(s[k], k) for k in FIT_SEEDS})
            if out[-1]["pca"] >= 2**32:
                raise ValueError("the PCA seed must fit in 32 bits")
        return out

    # -- fitting ------------------------------------------------------------

    def fit(self, X, Y, row_ids=None) -> NeuralPosterior:
        """Fit every member on the same rows; the LAST ``n_validation`` rows choose the epochs.

        Each member selects its epoch count on the last ``n_validation`` rows
        in the given order and is then refitted on every row (see
        :meth:`MixturePosterior.fit`), so the row order matters.

        :param X: ``(rows, features)`` real predictors. A NumPy array (or memory
            map) is used as is; anything else is converted to float64.
        :param Y: ``(rows, targets)`` finite targets in the caller's coordinates.
        :param row_ids: optional strictly increasing integer row IDs recorded in
            each member's history; default ``0 .. rows - 1``.
        :return: ``self``.
        """
        settings = self.settings()
        seeds = self._seeds()
        _torch()
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=float)
        if X.ndim != 2 or 0 in X.shape:
            raise ValueError(f"X must be a non-empty (rows, features) array, got {X.shape}")
        if X.dtype.kind not in "iuf":
            raise ValueError(f"X must be real numeric, got dtype {X.dtype}")
        Y = np.asarray(Y, dtype=float)
        if Y.ndim != 2 or Y.shape[0] != X.shape[0] or Y.shape[1] == 0:
            raise ValueError(f"Y must be ({X.shape[0]}, targets), got {Y.shape}")
        ids = np.arange(X.shape[0], dtype=np.int64) if row_ids is None else np.asarray(row_ids)
        members = [MixturePosterior(settings).fit(X, Y, ids, s) for s in seeds]
        self.ensemble_ = EqualWeightEnsemble(members)
        self.member_seeds_ = seeds
        self.n_features_in_ = int(X.shape[1])
        self.n_targets_ = int(Y.shape[1])
        return self

    @property
    def members_(self) -> list[MixturePosterior]:
        return self.ensemble_.members

    def _check_X(self, X) -> np.ndarray:
        if not hasattr(self, "ensemble_"):
            raise RuntimeError("the estimator is not fitted or loaded")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must be (cases, {self.n_features_in_}), got {X.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must be finite")
        return X

    # -- evaluation ---------------------------------------------------------

    def sample(
        self, X, n_draws: int, *, case_ids, seed: int, chunk: int = CHUNK, return_labels=False
    ):
        """Joint posterior draws ``(cases, n_draws, targets)`` in the coordinates of ``Y``.

        Case ``c`` draws from ``SeedSequence(seed, spawn_key=(c,))`` only, so its
        draws do not depend on the other cases, their order or ``chunk``
        (up to the floating-point rounding of batched network passes).
        The result can be passed directly to :func:`skbel.evaluation.score_samples`,
        :func:`skbel.evaluation.event_probabilities` or the metrics in
        :mod:`skbel.metrics`.

        :param X: ``(cases, features)`` finite predictors.
        :param n_draws: positive int.
        :param case_ids: ``(cases,)`` distinct non-negative integer case IDs (required).
        :param seed: non-negative int (required).
        :param chunk: cases per network pass; bounds memory only.
        :param return_labels: also return the ``member`` and ``component`` labels.
        :return: the draws, or with ``return_labels=True`` a dict with ``latent``
            (the draws), ``member`` and ``component`` ``(cases, n_draws)``.
        """
        X = self._check_X(X)
        n_draws = _check_positive(n_draws, "n_draws")
        chunk = _check_positive(chunk, "chunk")
        out = self.ensemble_.sample(X, case_ids, n_draws, seed, chunk)
        return out if return_labels else out["latent"]

    def predict_mixtures(self, X, chunk: int = CHUNK) -> list[dict]:
        """Per-member mixture parameters of each case (see :meth:`MixturePosterior.predict_mixture`)."""
        X = self._check_X(X)
        return self.ensemble_.predict_mixtures(X, _check_positive(chunk, "chunk"))

    # -- persistence --------------------------------------------------------

    def save(self, directory) -> list[str]:
        """Write the members and ``ensemble.json`` into a NEW directory (see :meth:`EqualWeightEnsemble.save`)."""
        if not hasattr(self, "ensemble_"):
            raise RuntimeError("the estimator is not fitted or loaded")
        return self.ensemble_.save(directory)

    @classmethod
    def from_ensemble(cls, ensemble: EqualWeightEnsemble) -> NeuralPosterior:
        """Wrap fitted members; parameters are read from their settings and recorded seeds."""
        if not isinstance(ensemble, EqualWeightEnsemble):
            raise TypeError("ensemble must be an EqualWeightEnsemble")
        first = ensemble.members[0]
        s = first.settings
        seeds = []
        for m in ensemble.members:
            recorded = (m.history or {}).get("seeds")
            if not isinstance(recorded, dict) or set(recorded) != set(FIT_SEEDS):
                raise ValueError("every member history must record its fit seeds")
            seeds.append({k: int(recorded[k]) for k in FIT_SEEDS})
        self = cls(
            len(ensemble.members),
            covariance=s.covariance,
            n_components=s.n_components,
            hidden=s.hidden,
            n_pca=s.n_pca,
            learning_rate=s.learning_rate,
            batch_size=s.batch_size,
            max_epochs=s.max_epochs,
            n_validation=s.n_validation,
            patience=s.patience,
            min_improvement=s.min_improvement,
            log_scale_bounds=s.log_scale_bounds,
            random_state=None,
            member_seeds=seeds,
        )
        self.ensemble_ = ensemble
        self.member_seeds_ = seeds
        self.n_features_in_ = int(first.n_inputs)
        self.n_targets_ = int(first.n_targets)
        return self

    @classmethod
    def load(cls, paths) -> NeuralPosterior:
        """Load an ensemble directory or an ordered list of member ``.npz`` files."""
        return cls.from_ensemble(EqualWeightEnsemble.load(paths))


# ---------------------------------------------------------------------------
# Marginal quantile recalibration
# ---------------------------------------------------------------------------


def _check_draws(draws) -> np.ndarray:
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 3 or draws.shape[1] < 1:
        raise ValueError("draws must be (cases, draws, targets)")
    if not np.all(np.isfinite(draws)):
        raise ValueError("draws must be finite; nothing is masked or redrawn")
    return draws


def _check_groups(groups, shape) -> np.ndarray:
    groups = np.asarray(groups)
    if groups.shape != shape or groups.dtype.kind not in "iu":
        raise ValueError(f"groups must be integer labels of shape {shape}")
    return groups.astype(np.int64)


def pit_grid(n_draws: int) -> int:
    """Size ``K`` of the open tie-breaking grid for ``M = n_draws`` draws.

    ``K = (2**51 - 1) // (2 (M + 1))`` so that the exact PIT denominator
    ``D = 2 K (M + 1)`` stays below ``2**51``.
    """
    if isinstance(n_draws, bool) or not isinstance(n_draws, (int, np.integer)) or n_draws < 1:
        raise ValueError("n_draws must be a positive int")
    k = (_PIT_DENOMINATOR_LIMIT - 1) // (2 * (int(n_draws) + 1))
    if k < 1:
        raise ValueError("too many draws for the exact PIT grid")
    return int(k)


def pit_grid_indices(case_ids, n_targets: int, seed: int, n_draws: int) -> np.ndarray:
    """Grid index ``j`` per (case, target).

    ``j`` is the first ``integers(0, K)`` of
    ``default_rng(SeedSequence(seed, spawn_key=(case_id, target)))``.
    """
    seed = _check_seed(seed)
    k = pit_grid(n_draws)
    ids = np.asarray(case_ids, dtype=np.int64)
    out = np.empty((ids.size, n_targets), dtype=np.int64)
    for i, cid in enumerate(ids):
        for t in range(n_targets):
            rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(int(cid), t)))
            out[i, t] = rng.integers(0, k)
    return out


def pit_uniforms(case_ids, n_targets: int, seed: int, n_draws: int) -> np.ndarray:
    """The tie-breaking ``U = (2 j + 1) / (2 K)`` per (case, target), strictly inside (0, 1)."""
    k = pit_grid(n_draws)
    return (2 * pit_grid_indices(case_ids, n_targets, seed, n_draws) + 1) / (2.0 * k)


def pit_from_counts(below, ties, j, n_draws: int) -> np.ndarray:
    """Exact ``u = (below + U (ties + 1)) / (M + 1)`` with ``U = (2 j + 1) / (2 K)``.

    Combined integer numerator ``N = 2 K below + (2 j + 1)(ties + 1)`` over
    ``D = 2 K (M + 1) < 2**51``; ``0 < N < D`` exactly, so the single correctly
    rounded division lies strictly inside (0, 1): no clipping, nudging, redraw
    or dropped rank.
    """
    k = pit_grid(n_draws)
    below = np.asarray(below, dtype=np.int64)
    ties = np.asarray(ties, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    m = int(n_draws)
    if (
        np.any(below < 0)
        or np.any(ties < 0)
        or np.any(below + ties > m)
        or np.any(j < 0)
        or np.any(j >= k)
    ):
        raise ValueError("counts must satisfy 0 <= below, ties; below + ties <= M; 0 <= j < K")
    numerator = 2 * k * below + (2 * j + 1) * (ties + 1)
    denominator = 2 * k * (m + 1)
    if not (np.all(numerator > 0) and np.all(numerator < denominator)):
        raise RuntimeError("PIT numerator outside the open interval")
    u = numerator.astype(float) / float(denominator)
    if not np.all((u > 0.0) & (u < 1.0)):
        raise RuntimeError("PIT rounded onto an endpoint")
    return u


def randomized_pit(draws, truth, seed: int, case_ids=None) -> np.ndarray:
    """``u = (#draws < truth + U (#draws == truth + 1)) / (M + 1)`` per case and target.

    ``U`` lies on the open grid ``(2 j + 1) / (2 K)`` (``pit_grid``), keyed by
    (case ID, target); ``u`` is computed exactly by ``pit_from_counts`` and lies
    strictly inside (0, 1). Nothing is clipped or discarded.
    """
    draws = _check_draws(draws)
    truth = np.asarray(truth, dtype=float)
    n, m, d = draws.shape
    if truth.shape != (n, d) or not np.all(np.isfinite(truth)):
        raise ValueError("truth must be finite (cases, targets)")
    ids = _case_ids(np.arange(n) if case_ids is None else case_ids, n)
    below = np.sum(draws < truth[:, None, :], axis=1)
    ties = np.sum(draws == truth[:, None, :], axis=1)
    return pit_from_counts(below, ties, pit_grid_indices(ids, d, seed, m), m)


def mid_cdf_knots(values) -> tuple[np.ndarray, np.ndarray]:
    """Strictly increasing knots of the empirical mid-CDF of values in (0, 1).

    Each distinct value gets ``(preceding_count + 0.5 * tie_count) / n``; the
    exact endpoints ``(0, 0)`` and ``(1, 1)`` are added.
    """
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0 or np.any(~np.isfinite(values)) or np.any((values <= 0) | (values >= 1)):
        raise ValueError("PIT values must be finite and lie strictly inside (0, 1)")
    uniq, counts = np.unique(values, return_counts=True)
    preceding = np.cumsum(counts) - counts
    y = (preceding + 0.5 * counts) / values.size
    x = np.concatenate([[0.0], uniq, [1.0]])
    y = np.concatenate([[0.0], y, [1.0]])
    if not (np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)):
        raise RuntimeError("calibration knots are not strictly increasing")
    return x, y


def midrank_probabilities(draws) -> np.ndarray:
    """Within-case empirical midrank probability ``(#less + 0.5 #equal) / M`` of every draw."""
    draws = _check_draws(draws)
    n, m, d = draws.shape
    out = np.empty_like(draws)
    for i in range(n):
        for t in range(d):
            col = draws[i, :, t]
            s = np.sort(col)
            less = np.searchsorted(s, col, side="left")
            equal = np.searchsorted(s, col, side="right") - less
            out[i, :, t] = (less + 0.5 * equal) / m
    return out


def empirical_quantile(sorted_draws, v) -> np.ndarray:
    """Generalized inverse of the empirical CDF: ``x_(max(ceil(v M), 1))`` per row, v in [0, 1]."""
    sorted_draws = np.asarray(sorted_draws, dtype=float)
    v = np.asarray(v, dtype=float)
    m = sorted_draws.shape[-1]
    if np.any(~np.isfinite(v)) or np.any((v < 0) | (v > 1)):
        raise ValueError("quantile levels must lie in [0, 1]")
    idx = np.maximum(np.ceil(v * m).astype(np.int64), 1) - 1
    return np.take_along_axis(sorted_draws, idx, axis=-1)


class QuantileRecalibrator:
    """Monotone per-(target, group) marginal recalibration of sample-based posteriors.

    ``fit`` forms randomized PIT values of held-out draws against their truths
    and, for every target and every group label with at least ``min_cases``
    cases, the strictly increasing piecewise-linear mid-CDF map ``H`` of those
    values (endpoints exact). Smaller groups are recorded INCOMPLETE: no
    pooled fallback, smoothing or extra sampling. ``transform`` takes each
    draw's within-case midrank probability ``p``, ``v = H^-1(p)`` of its routed
    group, and returns the raw empirical quantile ``Q_raw(v)`` of the same
    case/target, keeping every draw index (the joint draw order is shared
    across targets; ties stay ties). An identity map returns the draws
    exactly. This is a sample-based marginal transform, not a joint posterior.
    """

    def __init__(self) -> None:
        self.maps: dict[tuple[int, int], dict] | None = None
        self.n_targets: int | None = None
        self.labels: list[int] | None = None
        self.min_cases: int | None = None
        self.seed: int | None = None
        self.pit_: np.ndarray | None = None

    def fit(
        self, draws, truth, groups, seed: int, min_cases: int = 100, *, labels=None, case_ids=None
    ) -> QuantileRecalibrator:
        """Fit one map per (target, group) on held-out draws and their truths.

        :param draws: ``(cases, draws, targets)`` finite held-out draws.
        :param truth: ``(cases, targets)`` finite realized values.
        :param groups: ``(cases, targets)`` integer group labels of the fitting cases.
        :param seed: non-negative int of the PIT tie-breaking grid.
        :param min_cases: smallest group size that gets a map.
        :param labels: optional complete label set (every present group must be
            in it); labels without enough cases are recorded INCOMPLETE.
        :param case_ids: optional ``(cases,)`` distinct case IDs keying the PIT
            tie-breaking; default ``0 .. cases - 1``.
        """
        draws = _check_draws(draws)
        n, _, d = draws.shape
        groups = _check_groups(groups, (n, d))
        if isinstance(min_cases, bool) or not isinstance(min_cases, int) or min_cases < 1:
            raise ValueError("min_cases must be a positive int")
        present = sorted(int(g) for g in np.unique(groups))
        labels = present if labels is None else sorted(int(g) for g in labels)
        if len(set(labels)) != len(labels) or not set(present) <= set(labels):
            raise ValueError("labels must be distinct and include every group present")
        pit = randomized_pit(draws, truth, seed, case_ids)
        maps = {}
        for t in range(d):
            for g in labels:
                values = pit[groups[:, t] == g, t]
                entry = {"n": int(values.size), "complete": values.size >= min_cases}
                if entry["complete"]:
                    entry["x"], entry["y"] = mid_cdf_knots(values)
                maps[(t, g)] = entry
        self.maps, self.n_targets, self.labels = maps, d, labels
        self.min_cases, self.seed, self.pit_ = min_cases, _check_seed(seed), pit
        return self

    @property
    def complete(self) -> bool:
        return self.maps is not None and all(e["complete"] for e in self.maps.values())

    def incomplete(self) -> list[dict]:
        return [
            {"target": t, "group": g, "n": e["n"]}
            for (t, g), e in sorted(self.maps.items())
            if not e["complete"]
        ]

    def transform(self, draws, groups) -> np.ndarray:
        """Recalibrated draws, same shape and draw indices; ``groups`` are the ROUTED labels."""
        if self.maps is None:
            raise RuntimeError("the recalibrator is not fitted or loaded")
        draws = _check_draws(draws)
        n, m, d = draws.shape
        if d != self.n_targets:
            raise ValueError(f"draws have {d} targets, the maps {self.n_targets}")
        groups = _check_groups(groups, (n, d))
        unknown = sorted(set(np.unique(groups).tolist()) - set(self.labels))
        if unknown:
            raise ValueError(f"routed groups {unknown} have no map")
        routed_incomplete = sorted(
            {(t, int(g)) for t in range(d) for g in np.unique(groups[:, t])}
            & {k for k, e in self.maps.items() if not e["complete"]}
        )
        if routed_incomplete:
            raise ValueError(f"cases are routed to INCOMPLETE maps {routed_incomplete}")
        p = midrank_probabilities(draws)
        out = np.empty_like(draws)
        for t in range(d):
            s = np.sort(draws[:, :, t], axis=1)
            for g in np.unique(groups[:, t]):
                rows = groups[:, t] == g
                entry = self.maps[(t, int(g))]
                v = np.interp(p[rows, :, t], entry["y"], entry["x"])
                out[rows, :, t] = empirical_quantile(s[rows], v)
        return out

    def map_values(self, target: int, group: int, u) -> np.ndarray:
        """``H(u)`` of one complete map (for reporting)."""
        entry = self.maps[(target, group)]
        if not entry["complete"]:
            raise ValueError("the map is INCOMPLETE")
        return np.interp(np.asarray(u, dtype=float), entry["x"], entry["y"])

    def save(self, path) -> None:
        """Write a NEW ``.npz`` with every knot array, the fit PIT values and a JSON record."""
        if self.maps is None:
            raise RuntimeError("the recalibrator is not fitted")
        arrays = {"format": np.array(RECALIBRATOR_FORMAT)}
        meta = {
            "n_targets": self.n_targets,
            "labels": self.labels,
            "min_cases": self.min_cases,
            "seed": self.seed,
            "maps": [],
        }
        for (t, g), e in sorted(self.maps.items()):
            meta["maps"].append({"target": t, "group": g, "n": e["n"], "complete": e["complete"]})
            if e["complete"]:
                arrays[f"map/{t}/{g}/x"] = e["x"]
                arrays[f"map/{t}/{g}/y"] = e["y"]
        if self.pit_ is not None:
            arrays["fit_pit"] = self.pit_
        arrays["meta_json"] = np.array(json.dumps(meta))
        with open(Path(path), "xb") as f:
            np.savez(f, **arrays)

    @classmethod
    def load(cls, path) -> QuantileRecalibrator:
        """Read a persisted recalibrator; every complete map must be strictly increasing with exact ends."""
        with np.load(Path(path), allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        if str(arrays.get("format")) != RECALIBRATOR_FORMAT:
            raise ValueError(f"{path} is not a {RECALIBRATOR_FORMAT} file")
        meta = json.loads(str(arrays["meta_json"]))
        self = cls()
        self.n_targets, self.labels = int(meta["n_targets"]), [int(g) for g in meta["labels"]]
        self.min_cases, self.seed = int(meta["min_cases"]), int(meta["seed"])
        self.pit_ = arrays.get("fit_pit")
        self.maps = {}
        for e in meta["maps"]:
            key = (int(e["target"]), int(e["group"]))
            entry = {"n": int(e["n"]), "complete": bool(e["complete"])}
            if entry["complete"]:
                x, y = arrays[f"map/{key[0]}/{key[1]}/x"], arrays[f"map/{key[0]}/{key[1]}/y"]
                if not (
                    x[0] == 0
                    and y[0] == 0
                    and x[-1] == 1
                    and y[-1] == 1
                    and np.all(np.diff(x) > 0)
                    and np.all(np.diff(y) > 0)
                ):
                    raise ValueError(
                        f"{path}: map {key} is not strictly increasing with exact ends"
                    )
                entry["x"], entry["y"] = x, y
            self.maps[key] = entry
        return self
