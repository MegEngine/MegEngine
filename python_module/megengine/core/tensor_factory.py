# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Iterable, Optional, Union

import megengine._internal as mgb

from .graph import _use_default_if_none
from .tensor import Tensor

__all__ = ["zeros", "ones"]


def scalar(
    value,
    dtype: type = None,
    device: Optional[mgb.CompNode] = None,
    comp_graph: Optional[mgb.CompGraph] = None,
) -> Tensor:
    device, comp_graph = _use_default_if_none(device, comp_graph)
    return Tensor(mgb.make_immutable(device, comp_graph, value, dtype=dtype, name=None))


def zeros(
    shape: Union[int, Iterable[int], Tensor],
    dtype: type = None,
    device: Optional[mgb.CompNode] = None,
    comp_graph: Optional[mgb.CompGraph] = None,
) -> Tensor:
    """
    Create a tensor filled with 0.

    :param shape: tensor shape
    :param dtype: data type, Default: "int32"
    :param device: Compute node of the matrix, Default: None
    :param comp_graph: Compute graph of the matrix, Default: None
    :return: tensor of zeros

    Examples:

    .. testcode::

        import megengine as mge

        t = mge.zeros((2, 2), dtype="int32")
        print(t.numpy())

    Outputs:

    .. testoutput::

        [[0 0]
         [0 0]]

    """
    device, comp_graph = _use_default_if_none(device, comp_graph)
    if isinstance(shape, (int, Tensor)):
        shape = (shape,)
    tensor = scalar(0, dtype=dtype, device=device, comp_graph=comp_graph)
    tensor = tensor.broadcast(*shape)
    return tensor


def ones(
    shape: Union[int, Iterable[int], Tensor],
    dtype: type = None,
    device: Optional[mgb.CompNode] = None,
    comp_graph: Optional[mgb.CompGraph] = None,
) -> Tensor:
    """
    Create a tensor filled with 1.

    :param shape: tensor shape
    :param dtype: data type, Default: "int32"
    :param device: Compute node of the matrix, Default: None
    :param comp_graph: Compute graph of the matrix, Default: None
    :return: tensor of ones

    Examples:

    .. testcode::

        import megengine as mge

        t = mge.ones((2, 2), dtype="float32")
        print(t.numpy())

    Outputs:

    .. testoutput::

        [[1. 1.]
         [1. 1.]]

    """
    device, comp_graph = _use_default_if_none(device, comp_graph)
    if isinstance(shape, (int, Tensor)):
        shape = (shape,)
    tensor = scalar(1, dtype=dtype, device=device, comp_graph=comp_graph)
    tensor = tensor.broadcast(*shape)
    return tensor
