#  Copyright (c) 2022. Robin Thibaut, Ghent University

from importlib.metadata import PackageNotFoundError, version

from . import design, evaluation, learning, metrics
from .learning import *

try:
    __version__ = version("skbel")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "BEL",
    "algorithms",
    "design",
    "evaluation",
    "goggles",
    "learning",
    "metrics",
    "preprocessing",
    "spatial",
    "tmaps",
    "utils",
]
