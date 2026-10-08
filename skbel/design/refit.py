#  Copyright (c) 2026. Robin Thibaut, Ghent University

"""Fresh model refits for each design on a shared simulation bank.

Comparing designs on one :class:`~skbel.design.bank.SimulationBank` requires one
model per design, each learned from that design's features alone. Reusing a
fitted pipeline across designs would leak scaling, dimension reduction or
regression state from one design into another. :func:`fit_design` therefore
clones an *unfitted* template with :func:`sklearn.base.clone` for every refit,
which gives new, unfitted copies of every nested pre-processing,
regression and post-processing object, and fits that clone on the named
training rows only.

The template is never fitted or modified. Fitted state attached to the
template (including cached pre-processed arrays on a :class:`~skbel.BEL`) is
not a constructor parameter and is not carried into the clone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.base import clone

from .bank import DesignSelection, SimulationBank

__all__ = ["DesignRefit", "fit_design", "fit_designs"]


@dataclass(frozen=True)
class DesignRefit:
    """A model fitted for one design, with the selection needed to use it.

    :param model: the fitted clone of the template.
    :param selection: the :class:`~skbel.design.bank.DesignSelection` that produced
        the training features.
    :param train_rows: bank rows used for fitting, in fitting order.
    """

    model: object
    selection: DesignSelection
    train_rows: tuple[int, ...]

    def features(self, observations) -> np.ndarray:
        """Mask observations exactly as the training features were masked.

        :param observations: ``(cases, n_time, n_sensors)`` on the bank's
            native grid (held-out simulations or observed records).
        :return: ``(cases, n_features)`` in the training feature order.
        """
        return self.selection.apply(observations)


def fit_design(
    template,
    bank: SimulationBank,
    selection: DesignSelection,
    *,
    train_rows,
) -> DesignRefit:
    """Fit a fresh clone of ``template`` on one design's training features.

    :param template: unfitted scikit-learn compatible estimator with
        ``fit(X, Y)``, typically a :class:`~skbel.BEL` with its pipelines.
        It is cloned, never fitted or modified.
    :param bank: the shared :class:`~skbel.design.bank.SimulationBank`.
    :param selection: sensors and time indices of the design, on the bank's grid.
    :param train_rows: bank rows to fit on. Required: no rows are used by
        default, so held-out rows, observed records and their labels never
        enter the fit unless the caller lists them here.
    :return: a :class:`DesignRefit` with the fitted clone and its selection.
    """
    if not isinstance(bank, SimulationBank):
        raise TypeError(f"bank must be a SimulationBank, got {type(bank).__name__}")
    rows = bank.rows(train_rows)
    X, Y = bank.features(selection, rows)
    model = clone(template)
    if model is template:
        raise RuntimeError("clone returned the template itself")
    model.fit(X, Y)
    return DesignRefit(model=model, selection=selection, train_rows=tuple(rows.tolist()))


def fit_designs(
    template,
    bank: SimulationBank,
    selections: Sequence[DesignSelection],
    *,
    train_rows,
) -> tuple[DesignRefit, ...]:
    """Fit one fresh clone of ``template`` per selection on the same training rows.

    Equivalent to calling :func:`fit_design` once per selection: each design
    gets its own unfitted clone, and no fitted object is shared between
    designs.

    :param selections: non-empty sequence of selections on the bank's grid.
    :return: one :class:`DesignRefit` per selection, in the same order.
    """
    if isinstance(selections, DesignSelection) or not isinstance(selections, Sequence):
        raise TypeError("selections must be a sequence of DesignSelection")
    if len(selections) == 0:
        raise ValueError("selections must not be empty")
    return tuple(fit_design(template, bank, s, train_rows=train_rows) for s in selections)
