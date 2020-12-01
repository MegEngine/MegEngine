# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Iterable, Optional

from .. import Tensor
from ..core._imperative_rt import invoke_op
from ..core._imperative_rt.core2 import apply
from ..core.ops.builtin import GaussianRNG, UniformRNG
from ..core.tensor import utils
from .rng import _random_seed_generator

__all__ = ["normal", "uniform"]


def normal(
    mean: float = 0, std: float = 1, size: Optional[Iterable[int]] = None
) -> Tensor:
    r"""
    Random variable with Gaussian distribution :math:`N(\mu, \sigma)`.

    :param size: output tensor size.
    :param mean: the mean or expectation of the distribution.
    :param std: the standard deviation of the distribution (variance = :math:`\sigma ^ 2`).
    :return: the output tensor.

    Examples:

    .. testcode::

        import megengine as mge
        import megengine.random as rand

        x = rand.normal(mean=0, std=1, size=(2, 2))
        print(x.numpy())
    
    Outputs:
    
    .. testoutput::
        :options: +SKIP

        [[-0.20235455 -0.6959438 ]
         [-1.4939808  -1.5824696 ]]

    """
    if size is None:
        size = (1,)
    seed = _random_seed_generator().__next__()
    op = GaussianRNG(seed=seed, mean=mean, std=std)
    _ref = Tensor([], dtype="int32")
    size = utils.astensor1d(size, _ref, dtype="int32")
    (output,) = apply(op, size)
    return output


def uniform(
    low: float = 0, high: float = 1, size: Optional[Iterable[int]] = None
) -> Tensor:
    r"""
    Random variable with uniform distribution $U(0, 1)$.

    :param size: output tensor size.
    :param low: lower range.
    :param high: upper range.
    :return: the output tensor.

    Examples:

    .. testcode::

        import megengine as mge
        import megengine.random as rand

        x = rand.uniform(size=(2, 2))
        print(x.numpy())
    
    Outputs:
    
    .. testoutput::
        :options: +SKIP

        [[0.76901674 0.70496535]
         [0.09365904 0.62957656]]

    """
    assert low < high, "Uniform is not defined when low >= high"

    if size is None:
        size = (1,)
    seed = _random_seed_generator().__next__()
    op = UniformRNG(seed=seed)
    _ref = Tensor([], dtype="int32")
    size = utils.astensor1d(size, _ref, dtype="int32")
    (output,) = apply(op, size)

    return low + (high - low) * output
