# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import functools
from typing import Iterable, List, Optional, Union

import numpy as np

import megengine._internal as mgb
from megengine._internal import CompGraph, CompNode

from ..core import zeros
from ..core.graph import _use_default_if_none
from ..core.tensor import Tensor, wrap_io_tensor
from .utils import _decide_comp_node_and_comp_graph


@wrap_io_tensor
def broadcast_to(inp: Tensor, shape: Union[int, Iterable[int]]) -> Tensor:
    """
    Broadcast a tensor to ``shape``

    :param inp: The input tensor
    :param shape: The target shape
    :return: The output tensor

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        data = tensor(np.arange(0, 6, dtype=np.float32).reshape(2, 3))
        out = F.broadcast_to(data, (4, 2, 3))
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[[0. 1. 2.]
          [3. 4. 5.]]

         [[0. 1. 2.]
          [3. 4. 5.]]

         [[0. 1. 2.]
          [3. 4. 5.]]

         [[0. 1. 2.]
          [3. 4. 5.]]]

    """

    if isinstance(shape, int):
        shape = (shape,)
    return mgb.opr.broadcast(inp, shape)


def _get_idx(index, axis):
    index_dims = len(index.imm_shape)
    idx = []
    comp_node, comp_graph = _decide_comp_node_and_comp_graph(index)
    for i in range(index_dims):
        if i != axis:
            shape = [1] * index_dims
            shape[i] = index.axis_shape(i)
            arange = mgb.opr.linspace(
                0,
                index.axis_shape(i) - 1,
                index.axis_shape(i),
                comp_node=comp_node,
                comp_graph=comp_graph,
            )
            arange = (
                arange.reshape(*shape)
                .broadcast(index.shape)
                .reshape(-1)
                .astype(np.int32)
            )
            idx.append(arange)
        else:
            idx.append(index.reshape(-1))
    return tuple(idx)


@wrap_io_tensor
def gather(inp: Tensor, axis: int, index: Tensor) -> Tensor:
    r"""
    Gather data from :attr:`inp` on :attr:`axis` using :attr:`index`.

    For a 3-D tensor, the output is specified by::

        out[i][j][k] = inp[index[i][j][k]][j][k] # if axis == 0
        out[i][j][k] = inp[i][index[i][j][k]][k] # if axis == 1
        out[i][j][k] = inp[i][j][index[i][j][k]] # if axis == 2

    if :attr:`inp` is an n-dimensional tensor with size
    :math:`(x_0,x_1,...,x_{i-1},x_i,x_{i+1},...,x_{n-1})` and axis=i,
    then :attr:`index` must be an n-dimensional tensor with size
    :math:`(x_0,x_1,...,x_{i-1},y,x_{i+1},...,x_{n-1})` where :math:`y\ge 1` and
    output will have the same size as :attr:`index`.


    :param inp: the source tensor
    :param axis: the axis along which to index
    :param index: the indices of elements to gather

    Examples:

    .. testcode::

        import megengine.functional as F
        from megengine.core import tensor

        inp = tensor([
            [1,2], [3,4], [5,6],
        ])
        index = tensor([[0,2], [1,0]])
        oup = F.gather(inp, 0, index)
        print(oup.numpy())

    Outputs:

    .. testoutput::

        [[1 6]
         [3 2]]

    """

    input_shape = inp.imm_shape
    index_shape = index.imm_shape
    input_dims = len(input_shape)
    index_dims = len(index_shape)
    if input_dims != index_dims:
        raise ValueError(
            "The index tensor must have same dimensions as input tensor, "
            "But the input dims:{}, the index dims:{}".format(input_dims, index_dims)
        )

    if axis < 0 or axis >= input_dims:
        raise ValueError(
            "Index axis {} is output of bounds, should in range [0 {})".format(
                axis, input_dims
            )
        )

    for i in range(input_dims):
        if i != axis and input_shape[i] != index_shape[i]:
            raise ValueError(
                "The input {} and index {} must have the same size apart from axis {}".format(
                    input_shape, index_shape, axis
                )
            )

    idx = _get_idx(index, axis)
    return mgb.opr.advanced_indexing(inp)[idx].reshape(
        index.shape
    )  # pylint: disable=no-member


@wrap_io_tensor
def concat(
    inps: Iterable[Tensor],
    axis: int = 0,
    device: Optional[CompNode] = None,
    comp_graph: Optional[CompGraph] = None,
) -> Tensor:
    r"""
    Concat some tensors

    :param inps: Input tensors to concat
    :param axis: the dimension over which the tensors are concatenated,
        default to 0
    :param device: The comp node output on, default to None
    :param comp_graph: The graph in which output is, default to None
    :return: The output tensor

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        data1 = tensor(np.arange(0, 6, dtype=np.float32).reshape((2, 3)))
        data2 = tensor(np.arange(6, 12, dtype=np.float32).reshape((2, 3)))
        out = F.concat([data1, data2])
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[ 0.  1.  2.]
         [ 3.  4.  5.]
         [ 6.  7.  8.]
         [ 9. 10. 11.]]

    """

    # Output buffer not supported
    return mgb.opr.concat(
        *list(inps), axis=axis, comp_node=device, comp_graph=comp_graph
    )


@wrap_io_tensor
def scatter(inp: Tensor, axis: int, index: Tensor, source: Tensor) -> Tensor:
    r"""
    Writes all values from the tensor :attr:`source` into :attr:`inp` at the indices specified in the :attr:`index` tensor.

    For each value in :attr:`source`, its output index is specified by its index
    in :attr:`source` for ``axis != dimension`` and by the corresponding value in
    :attr:`index` for ``axis = dimension``.

    For a 3-D tensor, :attr:`inp` is updated as::

        inp[index[i][j][k]][j][k] = source[i][j][k]  # if axis == 0
        inp[i][index[i][j][k]][k] = source[i][j][k]  # if axis == 1
        inp[i][j][index[i][j][k]] = source[i][j][k]  # if axis == 2

    :attr:`inp`, :attr:`index` and :attr:`source` should have same number of dimensions.

    It is also required that ``source.shape(d) <= inp.shape(d)`` and ``index.shape(d) == source.shape(d)``
    for all dimensions ``d``.

    Moreover, the values of :attr:`index` must be between ``0`` and ``inp.shape(axis) - 1`` inclusive.


    :param inp: the inp tensor which to be scattered
    :param axis: the axis along which to index
    :param index: the indices of elements to scatter
    :param source: the source element(s) to scatter

    Examples:

    .. testcode::

        import numpy as np
        import megengine.functional as F
        from megengine.core import tensor
        inp = tensor(np.zeros(shape=(3,5),dtype=np.float32))
        source = tensor([[0.9935,0.9465,0.2256,0.8926,0.4396],[0.7723,0.0718,0.5939,0.357,0.4576]])
        index = tensor([[0,2,0,2,1],[2,0,0,1,2]])
        oup = F.scatter(inp, 0, index,source)
        print(oup.numpy())

    Outputs:

    .. testoutput::

        [[0.9935 0.0718 0.5939 0.     0.    ]
         [0.     0.     0.     0.357  0.4396]
         [0.7723 0.9465 0.     0.8926 0.4576]]

    """

    input_shape = inp.imm_shape
    index_shape = index.imm_shape
    source_shape = source.imm_shape
    input_dims = len(input_shape)
    index_dims = len(index_shape)
    source_dims = len(source_shape)

    if input_dims != index_dims or input_dims != source_dims:
        raise ValueError("The input, source and index tensor must have same dimensions")

    if axis < 0 or axis >= input_dims:
        raise ValueError(
            "Index axis {} is output of bounds, should in range [0 {})".format(
                axis, input_dims
            )
        )

    for i in range(source_dims):
        if source_shape[i] > input_shape[i]:
            raise ValueError(
                "The each shape size for source {} must be less than or equal to input {} ".format(
                    source_shape, input_shape
                )
            )

    for i in range(index_dims):
        if index_shape[i] != source_shape[i]:
            raise ValueError(
                "The each shape size for index {} must be equal to source {} ".format(
                    index_shape, source_shape
                )
            )

    for i in range(index_dims):
        if i != axis and index_shape[i] > input_shape[i]:
            raise ValueError(
                "The index {} must be less than or equal to input {} size apart from axis {}".format(
                    index_shape, input_shape, axis
                )
            )

    idx = _get_idx(index, axis)
    return mgb.opr.set_advanced_indexing(inp, source.flatten())[idx]


@wrap_io_tensor
def where(mask: Tensor, x: Tensor, y: Tensor) -> Tensor:
    r"""
    Select elements either from Tensor x or Tensor y, according to mask.
    
    .. math::
        
        \textrm{out}_i = x_i \textrm{ if } \textrm{mask}_i \textrm{ is True else } y_i

    :param mask: a mask used for choosing x or y
    :param x: the first choice
    :param y: the second choice

    Examples:

    .. testcode::

        from megengine import tensor
        import megengine.functional as F
        mask = tensor(np.array([[1, 0], [0, 1]], dtype=np.int32))
        x = tensor(np.array([[1, np.inf], [np.nan, 4]],
            dtype=np.float32))
        y = tensor(np.array([[5, 6], [7, 8]], dtype=np.float32))
        out = F.where(mask, x, y)
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[1. 6.]
         [7. 4.]]
    """
    v0, index0 = mgb.opr.cond_take(
        x, mask, mode=mgb.opr_param_defs.CondTake.Mode.EQ, val=1
    )
    v1, index1 = mgb.opr.cond_take(
        y, mask, mode=mgb.opr_param_defs.CondTake.Mode.EQ, val=0
    )
    out = x.flatten()
    out = mgb.opr.set_advanced_indexing(out, v0)[index0]
    out = mgb.opr.set_advanced_indexing(out, v1)[index1]
    out = out.reshape(x.shape)
    return out


def shapeof(x: Tensor, axis=None):
    r"""
    The shape of input tensor.
    """
    return x.shapeof(axis=axis)


@wrap_io_tensor
def dimshuffle(inp: Tensor, pattern: Iterable[int]) -> Tensor:
    r
    """
    Swap shapes and strides according to given pattern

    :param inp: Input tensor
    :param pattern: a list of integers including 0, 1, ... , ``ndim``-1, and any number of ``'x'`` char in dimensions where this tensor should be broadcasted. For examples:

        * (``'x'``) -> make a 0d (scalar) into a 1d vector
        * (0, 1) -> identity for 2d vectors
        * (1, 0) -> inverts the first and second dimensions
        * (``'x'``, 0) -> make a row out of a 1d vector (N to 1xN)
        * (0, ``'x'``) -> make a column out of a 1d vector (N to Nx1)
        * (2, 0, 1) -> AxBxC to CxAxB
        * (0, ``'x'``, 1) -> AxB to Ax1xB
        * (1, ``'x'``, 0) -> AxB to Bx1xA
        * (1,) -> This remove dimensions 0. It must be a broadcastable dimension (1xA to A)

    :return: The output tensor

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F
        x = tensor(np.array([[1, 1], [0, 0]], dtype=np.int32))
        out = F.dimshuffle(x, (1, 0))
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[1 0]
         [1 0]]

    """
    return mgb.opr.dimshuffle(inp, pattern)


@wrap_io_tensor
def reshape(inp: Tensor, target_shape: Iterable[int]) -> Tensor:
    r
    """
    Reshape a tensor to given target shape; total number of logical elements must
    remain unchanged

    :param inp: Input tensor
    :param target_shape: target shape, the components would be concatenated to form the
        target shape, and it can contain an element of -1 representing unspec_axis.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F
        x = tensor(np.arange(12, dtype=np.int32))
        out = F.reshape(x, (3, 2, 2))
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[[ 0  1]
          [ 2  3]]

         [[ 4  5]
          [ 6  7]]

         [[ 8  9]
          [10 11]]]

    """
    return mgb.opr.reshape(inp, target_shape)


@functools.wraps(dimshuffle)
def transpose(*args, **kwargs):
    r
    """See :func:`dimshuffle`
    """
    return dimshuffle(*args, **kwargs)


@wrap_io_tensor
def add_axis(inp: Tensor, axis: Union[int, Iterable[int]]) -> Tensor:
    r"""
    Add dimension(s) before given axis/axes

    :param inp: Input tensor
    :param axis: Place(s) of new axes
    :return: The output tensor

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F
        x = tensor([1, 2])
        out = F.add_axis(x, (0, 2))
        print(out.shape)

    Outputs:

    .. testoutput::

        (1, 2, 1)

    """
    return mgb.opr.add_axis(inp, axis)


@wrap_io_tensor
def remove_axis(inp: Tensor, axis: Union[int, Iterable[int]]) -> Tensor:
    r"""
    Remove dimension(s) of shape 1

    :param inp: Input tensor
    :param axis: Place(s) of axes to be removed
    :return: The output tensor

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F
        x = tensor(np.array([1, 2], dtype=np.int32).reshape(1, 1, 2, 1))
        out = F.remove_axis(x, (0, 0, 1))
        print(out.shape)

    Outputs:

    .. testoutput::

        (2,)

    """
    return mgb.opr.remove_axis(inp, axis)


def linspace(
    start: Union[int, float, Tensor],
    stop: Union[int, float, Tensor],
    num: int = 100,
    dtype=np.float32,
    device: Optional[CompNode] = None,
    comp_graph: Optional[CompGraph] = None,
) -> Tensor:
    r"""
    Return equally spaced numbers over a specified interval

    :param start: Starting value of the squence, shoule be scalar
    :param stop: The last value of the squence, shoule be scalar
    :param num: number of values to generate
    :param dtype: result data type
    :return: The generated tensor

    Examples:

    .. testcode::

        import numpy as np
        import megengine.functional as F

        a = F.linspace(3,10,5)
        print(a.numpy())

    .. testoutput::

        [ 3.    4.75  6.5   8.25 10.  ]

    """
    if dtype is not np.float32:
        raise ValueError("linspace is only implemented for float32")

    device, comp_graph = _use_default_if_none(device, comp_graph)
    ret = Tensor(
        mgb.opr.linspace(start, stop, num, comp_node=device, comp_graph=comp_graph)
    )
    return ret.astype(dtype)


def zeros_like(inp: Tensor) -> Tensor:
    r"""
    Returns a zero tensor with the same shape as input tensor

    :param inp: input tensor

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        inp = tensor(np.arange(1, 7, dtype=np.int32).reshape(2,3))
        out = F.zeros_like(inp)
        print(out.numpy())

    .. testoutput::

        [[0 0 0]
         [0 0 0]]

    """
    return zeros(inp.shapeof()).astype(inp.dtype)
