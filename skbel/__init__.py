#  Copyright (c) 2022. Robin Thibaut, Ghent University

from .learning import *

__version__ = "2.1.x"

source = __name__.split(".")[-1]

__all__ = [
    "utils",
    "goggles",
    "algorithms",
    "BEL",
    "preprocessing",
    "spatial",
    "tmaps",
]
