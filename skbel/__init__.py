#  Copyright (c) 2021. Robin Thibaut, Ghent University
# import os
# from datetime import date
# from os.path import join as jp
#
# from loguru import logger

__version__ = "1.0.2"

source = __name__.split(".")[-1]
# Set up logger
# logger.add(
#     jp(os.getcwd(), "logs", f"{source}_{date.today()}.log"),
#     backtrace=True,
#     diagnose=True,
#     enqueue=True,
# )

__all__ = [
    "utils",
    "goggles",
    "algorithms",
    "learning",
    "preprocessing",
    "spatial",
]
