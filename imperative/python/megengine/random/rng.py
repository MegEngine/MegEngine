# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import time
from typing import Iterable, Optional

from numpy.random import MT19937

from .. import Tensor
from ..core._imperative_rt.core2 import apply
from ..core._imperative_rt.ops import delete_rng_handle as _delete_rng_handle
from ..core._imperative_rt.ops import get_global_rng_seed as _get_global_rng_seed
from ..core._imperative_rt.ops import new_rng_handle as _new_rng_handle
from ..core._imperative_rt.ops import set_global_rng_seed as _set_global_rng_seed
from ..core.ops.builtin import GaussianRNG, UniformRNG
from ..core.tensor import utils
from ..device import get_default_device

_rng = None


def _normal(
    mean: float,
    std: float,
    size: Optional[Iterable[int]],
    seed: int,
    device: str,
    handle: int,
) -> Tensor:
    if size is None:
        size = (1,)
    op = GaussianRNG(seed=seed, mean=mean, std=std, handle=handle)
    _ref = Tensor([], dtype="int32", device=device)
    shape = utils.astensor1d(size, _ref, dtype="int32", device=device)
    (output,) = apply(op, shape)
    return output


def _uniform(
    low: float,
    high: float,
    size: Optional[Iterable[int]],
    seed: int,
    device: str,
    handle: int,
) -> Tensor:
    assert low < high, "Uniform is not defined when low >= high"
    if size is None:
        size = (1,)
    op = UniformRNG(seed=seed, handle=handle)
    _ref = Tensor([], dtype="int32", device=device)
    shape = utils.astensor1d(size, _ref, dtype="int32", device=device)
    (output,) = apply(op, shape)
    return low + (high - low) * output


class RNG:
    def __init__(self, seed=0, device=None):
        self.seed = seed
        self.device = device if device else get_default_device()
        self.handle = _new_rng_handle(self.device, self.seed)

    def uniform(
        self, low: float = 0, high: float = 1, size: Optional[Iterable[int]] = None
    ):
        return _uniform(
            low=low,
            high=high,
            size=size,
            seed=self.seed,
            device=self.device,
            handle=self.handle,
        )

    def normal(
        self, mean: float = 0, std: float = 1, size: Optional[Iterable[int]] = None
    ):
        return _normal(
            mean=mean,
            std=std,
            size=size,
            seed=self.seed,
            device=self.device,
            handle=self.handle,
        )

    def __del__(self):
        _delete_rng_handle(self.handle)


def _random_seed_generator():
    assert _rng
    while True:
        yield _rng.random_raw()


def seed(seed: int):
    global _rng  # pylint: disable=global-statement
    _rng = MT19937(seed=seed)
    _set_global_rng_seed(seed)


seed(int(time.time()))
