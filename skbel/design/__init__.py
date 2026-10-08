#  Copyright (c) 2026. Robin Thibaut, Ghent University

"""Design comparison on a shared simulation bank.

:mod:`skbel.design.bank` holds paired simulated observations and targets and
selects exact sensor, cadence and window subsets of them (NumPy only).
:mod:`skbel.design.refit` fits a fresh clone of an unfitted model, such as a
:class:`~skbel.BEL`, for each selection.
"""

from .bank import DesignSelection, SimulationBank, cadence_window
from .refit import DesignRefit, fit_design, fit_designs

__all__ = [
    "DesignRefit",
    "DesignSelection",
    "SimulationBank",
    "cadence_window",
    "fit_design",
    "fit_designs",
]
