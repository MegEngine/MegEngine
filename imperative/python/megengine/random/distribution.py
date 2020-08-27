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
from ..core.ops.builtin import GaussianRNG, UniformRNG
from ..core.tensor import utils
from ..core.tensor.core import apply
from .rng import _random_seed_generator

__all__ = ["gaussian", "uniform"]


def gaussian(shape: Iterable[int], mean: float = 0, std: float = 1,) -> Tensor:
    r"""Random variable with Gaussian distribution $N(\mu, \sigma)$

    :param shape: Output tensor shape
    :param mean: The mean or expectation of the distribution
    :param std: The standard deviation of the distribution (variance = $\sigma ^ 2$)
    :return: The output tensor

    Examples:

    .. testcode::

        import megengine as mge
        import megengine.random as rand

        x = rand.gaussian((2, 2), mean=0, std=1)
        print(x.numpy())

    .. testoutput::
        :options: +SKIP

        [[-0.20235455 -0.6959438 ]
         [-1.4939808  -1.5824696 ]]

    """
    seed = _random_seed_generator().__next__()
    op = GaussianRNG(seed=seed, mean=mean, std=std)
    shape = Tensor(shape, dtype="int32")
    (output,) = apply(op, shape)
    return output


def uniform(shape: Iterable[int], low: float = 0, high: float = 1,) -> Tensor:
    r"""Random variable with uniform distribution $U(0, 1)$

    :param shape: Output tensor shape
    :param low: Lower range
    :param high: Upper range
    :return: The output tensor

    Examples:

    .. testcode::

        import megengine as mge
        import megengine.random as rand

        x = rand.uniform((2, 2))
        print(x.numpy())

    .. testoutput::
        :options: +SKIP

        [[0.76901674 0.70496535]
         [0.09365904 0.62957656]]

    """
    assert low < high, "Uniform is not defined when low >= high"

    seed = _random_seed_generator().__next__()
    op = UniformRNG(seed=seed)
    shape = Tensor(shape, dtype="int32")
    (output,) = apply(op, shape)

    return low + (high - low) * output
