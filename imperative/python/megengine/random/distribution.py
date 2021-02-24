# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Iterable, Optional

from .. import Tensor
from ..core._imperative_rt.ops import get_global_rng_seed as _get_global_rng_seed
from .rng import _normal, _uniform

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
    return _normal(
        mean=mean,
        std=std,
        size=size,
        seed=_get_global_rng_seed(),
        device=None,
        handle=0,
    )


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
    return _uniform(
        low=low,
        high=high,
        size=size,
        seed=_get_global_rng_seed(),
        device=None,
        handle=0,
    )
