# -*- coding: utf-8 -*-
from .rng import (
    RNG,
    beta,
    exponential,
    gamma,
    normal,
    permutation,
    poisson,
    seed,
    shuffle,
    uniform,
)

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
    "exponential",
]
# pylint: disable=undefined-variable
del rng  # type: ignore[name-defined]
