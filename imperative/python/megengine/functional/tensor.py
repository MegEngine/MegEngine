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
from ..core.tensor.tensor_wrapper import _remove_axis
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
    "arange",
    "broadcast",
    "concat",
    "cond_take",
    "dimshuffle",
    "expand_dims",
    "eye",
    "flatten",
    "full",
    "full_like",
    "gather",
    "identity",
    "linspace",
    "ones",
    "ones_like",
    "param_pack_concat",
    "param_pack_split",
    "reshape",
    "split",
    "squeeze",
    "stack",
    "scatter",
    "transpose",
    "where",
    "zeros",
    "zeros_like",
]


def eye(shape, *, dtype="float32", device: Optional[CompNode] = None) -> Tensor:
    """Returns a 2D tensor with ones on the diagonal and zeros elsewhere.

    :param shape: expected shape of output tensor.
    :param dtype: data type. Default: None
    :param device: compute node of the matrix. Default: None
    :return: eye matrix.

    Examples:

    .. testcode::

        import numpy as np
        import megengine.functional as F

        data_shape = (4, 6)
        out = F.eye(data_shape, dtype=np.float32)
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[1. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0.]]

    """
    op = builtin.Eye(k=0, dtype=dtype, comp_node=device)
    (result,) = apply(op, Tensor(shape, dtype="int32", device=device))
    return result


def full(shape, value, dtype="float32", device=None):
    """Returns a tensor with given shape and value.
    """
    if isinstance(shape, int):
        shape = (shape,)
    if device is None:
        device = get_default_device()
    (x,) = Const(value, dtype=dtype, device=device)(
        Tensor(value, dtype=dtype, device=device)
    )
    return broadcast(x, shape)


def ones(shape, dtype="float32", device=None):
    """Returns a ones tensor with given shape.

    :param inp: input tensor.
    :return: output zero tensor.

    Examples:

    .. testcode::

        import megengine.functional as F

        out = F.ones((2, 1))
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[1.]
         [1.]]

    """
    return full(shape, 1.0, dtype=dtype, device=device)


def zeros(shape, dtype="float32", device=None):
    """Returns a zero tensor with given shape.
    """
    return full(shape, 0.0, dtype=dtype, device=device)


def zeros_like(inp: Tensor) -> Tensor:
    """Returns a zero tensor with the same shape as input tensor.

    :param inp: input tensor.
    :return: output zero tensor.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        inp = tensor(np.arange(1, 7, dtype=np.int32).reshape(2,3))
        out = F.zeros_like(inp)
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[0 0 0]
         [0 0 0]]

    """
    return zeros(inp.shape, dtype=inp.dtype, device=inp.device)


def ones_like(inp: Tensor) -> Tensor:
    """Returns a ones tensor with the same shape as input tensor.
    """
    return ones(inp.shape, dtype=inp.dtype, device=inp.device)


def full_like(inp: Tensor, value: Union[int, float]) -> Tensor:
    """Returns a tensor filled with given value with the same shape as input tensor.
    """
    return full(inp.shape, value, dtype=inp.dtype, device=inp.device)


def identity(inp: Tensor) -> Tensor:
    """Applies an identity transformation to input tensor.

    :param inp: input tensor.
    :return: output tensor.
    """
    op = builtin.Identity()
    (data,) = convert_inputs(inp)
    (output,) = apply(op, data)
    return output


def broadcast(inp: Tensor, shape: Union[int, Iterable[int]]) -> Tensor:
    """
    Broadcasts a tensor to given shape.

    :param inp: input tensor.
    :param shape: target shape.
    :return: output tensor.

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
    return inp.broadcast(shape)


def concat(inps: Iterable[Tensor], axis: int = 0, device=None) -> Tensor:
    r"""
    Concat some tensors

    :param inps: input tensors to concat.
    :param axis: over which dimension the tensors are concatenated. Default: 0
    :param device: which device output will be. Default: None
    :return: output tensor.

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

    :param inps: input tensors.
    :param axis: which axis will be concatenated.
    :param device: the device output will be. Default: None
    :return: output concatenated tensor.

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

    inps = [expand_dims(inp, axis=axis) for inp in inps]
    return concat(inps, axis=axis, device=device)


def split(inp, nsplits_or_sections, axis=0):
    """Splits the input tensor into several smaller tensors.
    When nsplits_or_sections is int, the last tensor may be smaller than others.

    :param inp: input tensor.
    :param nsplits_or_sections: number of sub tensors or sections information list.
    :param axis: which axis will be splited.
    :return: output tensor list.

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
    r"""Gathers data from input tensor on axis using index.

    For a 3-D tensor, the output is specified by::

        out[i][j][k] = inp[index[i][j][k]][j][k] # if axis == 0
        out[i][j][k] = inp[i][index[i][j][k]][k] # if axis == 1
        out[i][j][k] = inp[i][j][index[i][j][k]] # if axis == 2

    if input tensor is a n-dimensional tensor with size
    :math:`(x_0,x_1,...,x_{i-1},x_i,x_{i+1},...,x_{n-1})` and axis=i,
    then index must be a n-dimensional tensor with size
    :math:`(x_0,x_1,...,x_{i-1},y,x_{i+1},...,x_{n-1})` where :math:`y\ge 1` and
    output will have the same size as index.

    :param inp: input tensor.
    :param axis: along which axis to index.
    :param index: indices of elements to gather.
    :return: output tensor.

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
    r"""Writes all values from the tensor source into input tensor
    at the indices specified in the index tensor.

    For each value in source, its output index is specified by its index
    in source for ``axis != dimension`` and by the corresponding value in
    index for ``axis = dimension``.

    For a 3-D tensor, input tensor is updated as::

        inp[index[i][j][k]][j][k] = source[i][j][k]  # if axis == 0
        inp[i][index[i][j][k]][k] = source[i][j][k]  # if axis == 1
        inp[i][j][index[i][j][k]] = source[i][j][k]  # if axis == 2

    ``inp``, ``index`` and ``source`` should have same number of dimensions.

    It is also required that ``source.shape(d) <= inp.shape(d)`` and ``index.shape(d) == source.shape(d)``
    for all dimensions ``d``.

    Moreover, the values of index must be between ``0`` and ``inp.shape(axis) - 1`` inclusive.

    .. note::
        Please notice that, due to performance issues, the result is uncertain on the GPU device
        if scattering different positions from source to the same destination position
        regard to index tensor.

        Check the following examples, the oup[0][2] is maybe
        from source[0][2] which value is 0.2256 or source[1][2] which value is 0.5339
        if set the index[1][2] from 1 to 0.

    :param inp: inp tensor which to be scattered.
    :param axis: axis along which to index.
    :param index: indices of elements to scatter.
    :param source: source element(s) to scatter.
    :return: output tensor.

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
    r"""Selects elements either from Tensor x or Tensor y, according to mask.

    .. math::

        \textrm{out}_i = x_i \textrm{ if } \textrm{mask}_i \textrm{ is True else } y_i

    :param mask: a mask used for choosing ``x`` or ``y``.
    :param x: first choice.
    :param y: second choice.
    :return: output tensor.

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
    Takes elements from data if specific condition is satisfied on mask.
    This operator has two outputs: the first is the elements taken,
    and the second is the indices corresponding to those elements;
    they are both 1-dimensional. High-dimension input would first be flattened.

    :param mask: condition param; must be the same shape with data.
    :param x: input tensor from which to take elements.

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

        [1. 4.] [0 3]

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


def transpose(inp: Tensor, pattern: Iterable[int]) -> Tensor:
    r"""
    Swaps shapes and strides according to given pattern.

    :param inp: input tensor.
    :param pattern: a list of integers including 0, 1, ... , ``ndim``-1,
    and any number of ``'x'`` char in dimensions where this tensor should be broadcasted. For examples:

        * (``'x'``) -> make a 0d (scalar) into a 1d vector
        * (0, 1) -> identity for 2d vectors
        * (1, 0) -> inverts the first and second dimensions
        * (``'x'``, 0) -> make a row out of a 1d vector (N to 1xN)
        * (0, ``'x'``) -> make a column out of a 1d vector (N to Nx1)
        * (2, 0, 1) -> AxBxC to CxAxB
        * (0, ``'x'``, 1) -> AxB to Ax1xB
        * (1, ``'x'``, 0) -> AxB to Bx1xA
        * (1,) -> this removes dimensions 0. It must be a broadcastable dimension (1xA to A)

    :return: output tensor.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F
        x = tensor(np.array([[1, 1], [0, 0]], dtype=np.int32))
        out = F.transpose(x, (1, 0))
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[1 0]
         [1 0]]

    """
    return inp.transpose(pattern)


def dimshuffle(inp: Tensor, pattern: Iterable[int]) -> Tensor:
    r"""Same as :func:`~.transpose`.
    """
    return transpose(inp, pattern)


def reshape(inp: Tensor, target_shape: Iterable[int]) -> Tensor:
    r"""
    Reshapes a tensor to given target shape; total number of logical elements must
    remain unchanged

    :param inp: input tensor.
    :param target_shape: target shape, it can contain an element of -1 representing ``unspec_axis``.

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
    return inp.reshape(target_shape)


AxisAddRemove = builtin.AxisAddRemove
AxisDesc = AxisAddRemove.AxisDesc


def flatten(inp: Tensor, start_axis: int = 0, end_axis: int = -1) -> Tensor:
    r"""Reshapes the tensor by flattening the sub-tensor from dimension ``start_axis`` to dimension ``end_axis``.

    :param inp: input tensor.
    :param start_axis: start dimension that the sub-tensor to be flattened. Default: 0
    :param end_axis: end dimension that the sub-tensor to be flattened. Default: -1
    :return: output tensor.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        inp_shape = (2, 2, 3, 3)
        x = tensor(
            np.arange(36, dtype=np.int32).reshape(inp_shape),
        )
        out = F.flatten(x, 2)
        print(x.numpy().shape)
        print(out.numpy().shape)

    Outputs:

    .. testoutput::

        (2, 2, 3, 3)
        (2, 2, 9)

    """
    target_shape = tuple(inp.shape[i] for i in range(start_axis)) + (-1,)
    if end_axis != -1:
        target_shape += (*inp.shape[end_axis + 1 :],)
    return inp.reshape(*target_shape)


def expand_dims(inp: Tensor, axis: Union[int, Sequence[int]]) -> Tensor:
    r"""
    Adds dimension before given axis.

    :param inp: input tensor.
    :param axis: place of new axes.
    :return: output tensor.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = tensor([1, 2])
        out = F.expand_dims(x, 0)
        print(out.shape)

    Outputs:

    .. testoutput::

        (1, 2)

    """
    Param = builtin.AxisAddRemove.Param

    def get_axes():
        try:
            return [int(axis)]
        except (TypeError, ValueError):
            pass
        return list(map(int, axis))

    axis = get_axes()
    ndim = inp.ndim + len(axis)
    axis = sorted(i + ndim if i < 0 else i for i in axis)

    param = Param(*map(builtin.AxisAddRemove.AxisDesc.make_add, axis))
    op = builtin.AxisAddRemove(param=param)
    (result,) = apply(op, inp)
    return result


def squeeze(inp: Tensor, axis: Optional[Union[int, Sequence[int]]] = None) -> Tensor:
    r"""
    Removes dimension of shape 1.

    :param inp: input tensor.
    :param axis: place of axis to be removed.
    :return: output tensor.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = tensor(np.array([1, 2], dtype=np.int32).reshape(1, 1, 2, 1))
        out = F.squeeze(x, 3)
        print(out.shape)

    Outputs:

    .. testoutput::

        (1, 1, 2)

    """
    return _remove_axis(inp, axis)


def linspace(
    start: Union[int, float, Tensor],
    stop: Union[int, float, Tensor],
    num: Union[int, Tensor],
    dtype="float32",
    device: Optional[CompNode] = None,
) -> Tensor:
    r"""Returns equally spaced numbers over a specified interval.

    :param start: starting value of the squence, shoule be scalar.
    :param stop: last value of the squence, shoule be scalar.
    :param num: number of values to generate.
    :param dtype: result data type.
    :return: generated tensor.

    Examples:

    .. testcode::

        import numpy as np
        import megengine.functional as F

        a = F.linspace(3,10,5)
        print(a.numpy())

    Outputs:

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
    r"""Returns a tensor with values from start to end with adjacent interval step.

    :param start: starting value of the squence, shoule be scalar.
    :param end: ending value of the squence, shoule be scalar.
    :param step: gap between each pair of adjacent values. Default: 1
    :param dtype: result data type.
    :return: generated tensor.

    Examples:

    .. testcode::

        import numpy as np
        import megengine.functional as F

        a = F.arange(5)
        print(a.numpy())

    Outputs:

    Outputs:

    .. testoutput::

        [0. 1. 2. 3. 4.]

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
    Returns split tensor to tensor list as offsets and shapes described,
            only used for ``parampack``.

    :param inp: input tensor.
    :param offsets: offsets of outputs, length of `2 * n`,
            while n is tensor nums you want to split,
            format `[begin0, end0, begin1, end1]`.
    :param shapes: tensor shapes of outputs.
    :return: splitted tensors.

    Examples:

    .. testcode::

        import numpy as np
        import megengine.functional as F
        from megengine import tensor

        a = tensor(np.ones((10,), np.int32))
        b, c = F.param_pack_split(a, [0, 1, 1, 10], [(1,), (3, 3)])
        print(b.numpy())
        print(c.numpy())

    Outputs:

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
    Returns concated tensor, only used for ``parampack``.

    :param inps: input tensors.
    :param offsets: device value of offsets.
    :param offsets_val: offsets of inputs, length of `2 * n`,
            format `[begin0, end0, begin1, end1]`.
    :return: concated tensor.

    Examples:

    .. testcode::

        import numpy as np
        import megengine.functional as F
        from megengine import tensor

        a = tensor(np.ones((1,), np.int32))
        b = tensor(np.ones((3, 3), np.int32))
        offsets_val = [0, 1, 1, 10]
        offsets = tensor(offsets_val, np.int32)
        c = F.param_pack_concat([a, b], offsets, offsets_val)
        print(c.numpy())

    Outputs:

    .. testoutput::

        [1 1 1 1 1 1 1 1 1 1]

    """
    op = builtin.ParamPackConcat()
    op.offsets = offsets_val
    return apply(op, *inps, offsets)[0]
