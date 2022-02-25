# -*- coding: utf-8 -*-
from .rng import RNG, beta, gamma, normal, permutation, poisson, seed, shuffle, uniform

__all__ = [
    "RNG",
    "beta",
    "gamma",
    "normal",
    "permutation",
    "poisson",
    "seed",
    "uniform",
    "shuffle",
]
# pylint: disable=undefined-variable
del rng  # type: ignore[name-defined]
