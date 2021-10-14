# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
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
