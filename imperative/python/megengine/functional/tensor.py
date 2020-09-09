# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import functools
import math
from itertools import accumulate
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..core._imperative_rt import CompNode
from ..core._wrap import device as as_device
from ..core.ops import builtin
from ..core.ops._internal import param_defs as P
from ..core.ops.special import Const
from ..core.tensor.core import TensorBase, TensorWrapperBase, apply
from ..core.tensor.utils import (
    astensor1d,
    convert_inputs,
    convert_single_value,
    dtype_promotion,
    get_device,
)
from ..device import get_default_device
from ..tensor import Tensor
from .elemwise import ceil

__all__ = [
    "add_axis",
    "arange",
    "broadcast",
    "concat",
    "cond_take",
    "dimshuffle",
    "expand_dims",
    "eye",
    "full",
    "full_like",
    "gather",
    "linspace",
    "ones",
    "ones_like",
    "param_pack_concat",
    "param_pack_split",
    "reshape",
    "remove_axis",
    "split",
    "squeeze",
    "stack",
    "scatter",
    "transpose",
    "where",
    "zeros",
    "zeros_like",
]


def eye(n: int, *, dtype=None, device: Optional[CompNode] = None) -> Tensor:
    """
    Returns a 2D tensor with ones on the diagonal and zeros elsewhere.

    :param n: The number of rows
    :param m: The number of columns. Default: None
    :param dtype: The data type. Default: None
    :param device: Compute node of the matrix. Default: None
    :param comp_graph: Compute graph of the matrix. Default: None
    :return: The eye matrix

    Examples:

    .. testcode::

        import numpy as np
        import megengine.functional as F

        data_shape = (4, 6)
        n, m = data_shape
        out = F.eye(n, m, dtype=np.float32)
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[1. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0.]]

    """
    op = builtin.Eye(k=0, dtype=dtype, comp_node=device)
    (result,) = apply(op, Tensor(n, dtype="int32", device=device))
    return result


def full(shape, value, dtype="float32", device=None):
    if isinstance(shape, int):
        shape = (shape,)
    if device is None:
        device = get_default_device()
    (x,) = Const(value, dtype=dtype, device=device)(
        Tensor(value, dtype=dtype, device=device)
    )
    return broadcast(x, shape)


def ones(shape, dtype="float32", device=None):
    return full(shape, 1.0, dtype=dtype, device=device)


def zeros(shape, dtype="float32", device=None):
    return full(shape, 0.0, dtype=dtype, device=device)


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
    return zeros(inp.shape, dtype=inp.dtype, device=inp.device)


def ones_like(inp: Tensor) -> Tensor:
    r"""
    Returns a identity tensor with the same shape as input tensor
    """
    return ones(inp.shape, dtype=inp.dtype, device=inp.device)


def full_like(inp: Tensor, value: Union[int, float]) -> Tensor:
    r"""
    Returns a tensor filled with value val with the same shape as input tensor
    """
    return full(inp.shape, value, dtype=inp.dtype, device=inp.device)


def broadcast(inp: Tensor, shape: Union[int, Iterable[int]]) -> Tensor:
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
        out = F.broadcast(data, (4, 2, 3))
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
    shape = astensor1d(shape, inp, dtype="int32", device=inp.device)
    (result,) = apply(builtin.Broadcast(), inp, shape)
    return result


def concat(inps: Iterable[Tensor], axis: int = 0, device=None) -> Tensor:
    r"""
    Concat some tensors

    :param inps: Input tensors to concat
    :param axis: the dimension over which the tensors are concatenated. Default: 0
    :param device: The comp node output on. Default: None
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
    if len(inps) == 1:
        return inps[0]

    dtype = dtype_promotion(inps)
    if device is None:
        device = get_device(inps)
    device = as_device(device)

    def convert(x):
        return convert_single_value(x, inps, dtype=dtype)

    inps = tuple(map(convert, inps))
    (result,) = apply(builtin.Concat(axis=axis, comp_node=device.to_c()), *inps)
    return result


def stack(inps, axis=0, device=None):
    """Concats a sequence of tensors along a new axis.
    The input tensors must have the same shape.

    :param inps: The input tensors.
    :param axis: Which axis will be concatenated.
    :param device: The comp node output on. Default: None
    :return: The output concatenated tensor.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x1 = tensor(np.arange(0, 6, dtype=np.float32).reshape((2, 3)))
        x2 = tensor(np.arange(6, 12, dtype=np.float32).reshape((2, 3)))
        out = F.stack([x1, x2], axis=0)
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[[ 0.  1.  2.]
          [ 3.  4.  5.]]

         [[ 6.  7.  8.]
          [ 9. 10. 11.]]]

    """
    if len(inps) > 0 and not isinstance(inps[0].shape, inps[0].__class__):
        shapes = {arr.shape for arr in inps}
        if len(shapes) != 1:
            raise ValueError("All input tensors must have the same shape")

    inps = [add_axis(inp, axis=axis) for inp in inps]
    return concat(inps, axis=axis, device=device)


def split(inp, nsplits_or_sections, axis=0):
    """Splits the input tensor into several smaller tensors.
    When nsplits_or_sections is int, the last tensor may be smaller than others.

    :param inp: The input tensor.
    :param nsplits_or_sections: Number of sub tensors or section information list.
    :param axis: Which axis will be splited.
    :return: The output tensor list.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = tensor(np.random.random((2,3,4,5)), dtype=np.float32)
        out = F.split(x, 2, axis=3)
        print(out[0].shape, out[1].shape)

    Outputs:

    .. testoutput::

        (2, 3, 4, 3) (2, 3, 4, 2)

    """
    sub_tensors = []
    sections = []

    def swapaxis(inp, src, dst):
        if src == dst:
            return inp
        shape = [i for i in range(inp.ndim)]
        shape[src] = dst
        shape[dst] = src
        return inp.transpose(shape)

    inp = swapaxis(inp, 0, axis)

    if isinstance(nsplits_or_sections, int):
        incr_step = ceil(inp.shape[0] / nsplits_or_sections)
        nsplits = nsplits_or_sections
        while nsplits > 0:
            nsplits -= 1
            sections.append(incr_step.astype("int32"))
            incr_step += nsplits_or_sections
    else:
        sections = nsplits_or_sections

    st = 0
    for se in sections:
        sub_tensors.append(swapaxis(inp[st:se], axis, 0))
        st = se

    if st < inp.shape[0]:
        sub_tensors.append(swapaxis(inp[st:], axis, 0))

    return sub_tensors


def _get_idx(index, axis):
    index_dims = len(index.shape)
    idx = []
    for i in range(index_dims):
        if i != axis:
            shape = [1] * index_dims
            shape[i] = index.shape[i]
            arange = linspace(
                0, index.shape[i] - 1, index.shape[i], device=index.device,
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
        from megengine import tensor

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
    input_shape = inp.shape
    index_shape = index.shape
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
    return inp[idx].reshape(index.shape)  # pylint: disable=no-member


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

    .. note::
        Please notice that, due to performance issues, the result is uncertain on the GPU device
        if scatter difference positions from source to the same destination position
        regard to index tensor.

        Show the case using the following examples, the oup[0][2] is maybe
        from source[0][2] which value is 0.2256 or source[1][2] which value is 0.5339
        if set the index[1][2] from 1 to 0.

    :param inp: the inp tensor which to be scattered
    :param axis: the axis along which to index
    :param index: the indices of elements to scatter
    :param source: the source element(s) to scatter

    Examples:

    .. testcode::

        import numpy as np
        import megengine.functional as F
        from megengine import tensor

        inp = tensor(np.zeros(shape=(3,5),dtype=np.float32))
        source = tensor([[0.9935,0.9465,0.2256,0.8926,0.4396],[0.7723,0.0718,0.5939,0.357,0.4576]])
        index = tensor([[0,2,0,2,1],[2,0,1,1,2]])
        oup = F.scatter(inp, 0, index,source)
        print(oup.numpy())

    Outputs:

    .. testoutput::

        [[0.9935 0.0718 0.2256 0.     0.    ]
         [0.     0.     0.5939 0.357  0.4396]
         [0.7723 0.9465 0.     0.8926 0.4576]]

    """
    input_shape = inp.shape
    index_shape = index.shape
    source_shape = source.shape
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
    inp[idx] = source.flatten()
    return inp


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
        mask = tensor(np.array([[True, False], [False, True]], dtype=np.bool))
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

    x, y = convert_inputs(x, y)
    if not isinstance(x, (TensorWrapperBase, TensorBase)):
        raise TypeError("input x must be a tensor")
    if not isinstance(y, (TensorWrapperBase, TensorBase)):
        raise TypeError("input y must be a tensor")
    if not isinstance(mask, (TensorWrapperBase, TensorBase)):
        raise TypeError("mask must be a tensor")
    if mask.dtype != np.bool_:
        raise ValueError("mask must be bool")
    if x.device != mask.device:
        raise ValueError("ambiguous device: {} vs {}".format(x.device, mask.device))

    v0, index0 = cond_take(mask, x)
    v1, index1 = cond_take(~mask, y)

    if v0.shape == (0,):
        out = v1
    elif v1.shape == (0,):
        out = v0
    else:
        out = concat([v0, v1])

    out[index0] = v0
    out[index1] = v1
    out = out.reshape(x.shape)
    return out


def cond_take(mask: Tensor, x: Tensor) -> Tensor:
    r"""
    Take elements from data if specific condition is satisfied on mask.
    This operator has two outputs: the first is the elements taken,
    and the second is the indices corresponding to those elements;
    they are both 1-dimensional. High-dimension input would first be flattened.

    :param mask: condition param; must be the same shape with data
    :param x: input tensor from which to take elements

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F
        mask = tensor(np.array([[True, False], [False, True]], dtype=np.bool_))
        x = tensor(np.array([[1, np.inf], [np.nan, 4]],
            dtype=np.float32))
        v, index = F.cond_take(mask, x)
        print(v.numpy(), index.numpy())

    Outputs:

    .. testoutput::

        Tensor([1. 4.]) Tensor([0 3], dtype=int32)

    """
    if not isinstance(x, (TensorWrapperBase, TensorBase)):
        raise TypeError("input must be a tensor")
    if not isinstance(mask, (TensorWrapperBase, TensorBase)):
        raise TypeError("mask must be a tensor")
    if mask.dtype != np.bool_:
        raise ValueError("mask must be bool")
    if x.device != mask.device:
        raise ValueError("ambiguous device: {} vs {}".format(x.device, mask.device))

    op = builtin.CondTake()
    v, index = apply(op, x, mask)
    return v, index


def dimshuffle(inp: Tensor, pattern: Iterable[int]) -> Tensor:
    r"""
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
    op = builtin.Dimshuffle(pattern)
    (inp,) = convert_inputs(inp)
    (result,) = apply(op, inp)
    return result


transpose = dimshuffle


def reshape(inp: Tensor, target_shape: Iterable[int]) -> Tensor:
    r"""
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
    if isinstance(target_shape, (TensorBase, TensorWrapperBase)):
        target_shape = target_shape.numpy()
    target_shape = tuple(map(int, target_shape))
    unspec_axis = None
    for i, s in enumerate(target_shape):
        if s < 0:
            if s != -1:
                raise ValueError("expect shape[{}] >= -1, got {}".format(i, s))
            if unspec_axis is not None:
                raise ValueError("multiple -1 in shape: {} & {}".format(unspec_axis, i))
            unspec_axis = i

    # TODO: device should be None (cpu)
    (target_shape,) = Const(target_shape, dtype="int32", device=inp.device)(inp)
    if unspec_axis is None:
        op = builtin.Reshape()
    else:
        op = builtin.Reshape(unspec_axis=unspec_axis)
    (x,) = apply(op, inp, target_shape)
    return x


AxisAddRemove = builtin.AxisAddRemove
AxisDesc = AxisAddRemove.AxisDesc


def add_axis(inp: Tensor, axis: Union[int, Sequence[int]]) -> Tensor:
    r"""
    Add dimension before given axis.

    :param inp: Input tensor
    :param axis: Place of new axes
    :return: The output tensor

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F
        x = tensor([1, 2])
        out = F.add_axis(x, 0)
        print(out.shape)

    Outputs:

    .. testoutput::

        (1, 2)

    """
    Param = AxisAddRemove.Param

    def get_axes():
        try:
            return [int(axis)]
        except (TypeError, ValueError):
            pass
        return list(map(int, axis))

    axis = get_axes()
    ndim = inp.ndim + len(axis)
    axis = sorted(i + ndim if i < 0 else i for i in axis)

    param = Param(*map(AxisDesc.make_add, axis))
    op = AxisAddRemove(param=param)
    (result,) = apply(op, inp)
    return result


expand_dims = add_axis


def remove_axis(
    inp: Tensor, axis: Optional[Union[int, Sequence[int]]] = None
) -> Tensor:
    r"""
    Remove dimension of shape 1.

    :param inp: Input tensor
    :param axis: Place of axis to be removed, if None, all axis=1 will be removed. Default: None
    :return: The output tensor

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F
        x = tensor(np.array([1, 2], dtype=np.int32).reshape(1, 1, 2, 1))
        out = F.remove_axis(x, 3)
        print(out.shape)

    Outputs:

    .. testoutput::

        (1, 1, 2)

    """
    Param = AxisAddRemove.Param

    def get_axes():
        if axis is None:
            return [i for i, s in enumerate(inp.shape) if s == 1]
        try:
            return [int(axis)]
        except (TypeError, ValueError):
            pass
        return list(map(int, axis))

    axis = get_axes()
    axis = sorted(i + inp.ndim if i < 0 else i for i in axis)
    axis = [a - i for i, a in enumerate(axis)]

    param = Param(*map(AxisDesc.make_remove, axis))
    op = AxisAddRemove(param=param)
    (result,) = apply(op, inp)
    return result


squeeze = remove_axis


def linspace(
    start: Union[int, float, Tensor],
    stop: Union[int, float, Tensor],
    num: Union[int, Tensor],
    dtype="float32",
    device: Optional[CompNode] = None,
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
    start = Tensor(start, device=device)
    stop = Tensor(stop, device=device)
    num = Tensor(num, device=device)

    device = device if device is None else device.to_c()
    op = builtin.Linspace(comp_node=device)
    (result,) = apply(op, start, stop, num)
    if np.dtype(dtype) == np.int32:
        return result.astype(dtype)
    return result


def arange(
    start: Union[int, float, Tensor] = 0,
    end: Optional[Union[int, float, Tensor]] = None,
    step: Union[int, float, Tensor] = 1,
    dtype="float32",
    device: Optional[CompNode] = None,
) -> Tensor:
    r"""
    Returns a Tensor with values from `start` to `end` with adjacent interval `step`

    :param start: starting value of the squence, shoule be scalar
    :param end: ending value of the squence, shoule be scalar
    :param step: the gap between each pair of adjacent values. Default 1
    :param dtype: result data type
    :return: The generated tensor

    Examples:

    .. testcode::

        import numpy as np
        import megengine.functional as F

        a = F.arange(5)
        print(a.numpy())

    .. testoutput::

        [1. 2. 3. 4.]

    """
    if end is None:
        start, end = 0, start

    if isinstance(start, Tensor):
        start = start.astype("float32")
    if isinstance(end, Tensor):
        end = end.astype("float32")
    if isinstance(step, Tensor):
        step = step.astype("float32")
    num = ceil(Tensor((end - start) / step, device=device))
    stop = start + step * (num - 1)
    result = linspace(start, stop, num, device=device)
    if np.dtype(dtype) == np.int32:
        return result.astype(dtype)
    return result


def param_pack_split(inp: Tensor, offsets: List, shapes: List) -> Tensor:
    r"""
    Returns split Tensor to Tensor list as offsets and shapes described,
            only used for parampack.

    :param inp: Input tensor
    :param offsets: offsets of outputs, length of 2 * n,
            while n is tensor nums you want to split,
            format [begin0, end0, begin1, end1].
    :param shapes: tensor shapes of outputs
    :return: split tensors

    Examples:

    .. testcode::

        import numpy as np
        import megengine.functional as F
        from megengine import tensor

        a = tensor(np.ones((10,), np.int32))
        b, c = F.param_pack_split(a, [0, 1, 1, 10], [(1,), (3, 3)])
        print(b.numpy())
        print(c.numpy())

    .. testoutput::

        [1]
        [[1 1 1]
         [1 1 1]
         [1 1 1]]

    """
    op = builtin.ParamPackSplit()
    op.offsets = offsets
    op.shapes = shapes
    return apply(op, inp)


def param_pack_concat(inps: List, offsets: Tensor, offsets_val: List) -> Tensor:
    r"""
    Returns concat Tensor, only used for parampack.

    :param inps: Input tensors
    :param offsets: device value of offsets
    :param offsets_val: offsets of inputs, length of 2 * n,
            format [begin0, end0, begin1, end1].
    :return: split tensors

    Examples:

    .. testcode::

        import numpy as np
        import megengine.functional as F
        from megengine import tensor

        a = tensor(np.ones((1,), np.int32))
        b = tensor(np.ones((3, 3), np.int32))
        offsets_val = [0, 1, 1, 10]
        offsets = tensor(offsets, np.int32)
        c = F.param_pack_concat([a, b], offsets, offsets_val)
        print(c.numpy())

    .. testoutput::

        [1 1 1 1 1 1 1 1 1 1]

    """
    op = builtin.ParamPackConcat()
    op.offsets = offsets_val
    return apply(op, *inps, offsets)[0]
