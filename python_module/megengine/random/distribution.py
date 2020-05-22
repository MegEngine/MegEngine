# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Iterable, Optional

import megengine._internal as mgb
from megengine._internal import CompGraph, CompNode

from ..core.graph import _use_default_if_none
from ..core.tensor import Tensor, wrap_io_tensor
from .rng import _random_seed_generator

__all__ = ["gaussian", "uniform"]


@wrap_io_tensor
def gaussian(
    shape: Iterable[int],
    mean: float = 0,
    std: float = 1,
    comp_node: Optional[CompNode] = None,
    comp_graph: Optional[CompGraph] = None,
) -> Tensor:
    r"""Random variable with Gaussian distribution $N(\mu, \sigma)$

    :param shape: Output tensor shape
    :param mean: The mean or expectation of the distribution
    :param std: The standard deviation of the distribution (variance = $\sigma ^ 2$)
    :param comp_node: The comp node output on, default to None
    :param comp_graph: The graph in which output is, default to None
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
    comp_node, comp_graph = _use_default_if_none(comp_node, comp_graph)
    seed = _random_seed_generator().__next__()
    return mgb.opr.gaussian_rng(
        shape, seed=seed, mean=mean, std=std, comp_node=comp_node, comp_graph=comp_graph
    )


@wrap_io_tensor
def uniform(
    shape: Iterable[int],
    low: float = 0,
    high: float = 1,
    comp_node: Optional[CompNode] = None,
    comp_graph: Optional[CompGraph] = None,
) -> Tensor:
    r"""Random variable with uniform distribution $U(0, 1)$

    :param shape: Output tensor shape
    :param low: Lower range
    :param high: Upper range
    :param comp_node: The comp node output on, default to None
    :param comp_graph: The graph in which output is, default to None
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

    comp_node, comp_graph = _use_default_if_none(comp_node, comp_graph)
    seed = _random_seed_generator().__next__()
    return low + (high - low) * mgb.opr.uniform_rng(
        shape, seed=seed, comp_node=comp_node, comp_graph=comp_graph
    )
