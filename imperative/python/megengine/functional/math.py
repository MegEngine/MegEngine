# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import collections
import math
from functools import lru_cache
from typing import Optional, Sequence, Tuple, Union

from ..core import _config
from ..core._imperative_rt.core2 import apply, dtype_promotion
from ..core._imperative_rt.ops import SubgraphBuilder as _SubgraphBuilder
from ..core._trace_option import use_symbolic_shape
from ..core.ops import builtin
from ..core.ops.builtin import BatchNorm, Elemwise, GetVarShape, Reduce, TypeCvt
from ..core.ops.special import Const
from ..core.tensor import amp
from ..core.tensor.utils import _normalize_axis, cast_tensors, setscalar, subgraph
from ..jit import exclude_from_trace
from ..tensor import Tensor
from .debug_param import get_execution_strategy
from .elemwise import clip, minimum
from .tensor import broadcast_to, concat, expand_dims, squeeze

__all__ = [
    "argmax",
    "argmin",
    "argsort",
    "dot",
    "isinf",
    "isnan",
    "matinv",
    "matmul",
    "max",
    "mean",
    "min",
    "norm",
    "normalize",
    "prod",
    "sign",
    "sort",
    "std",
    "sum",
    "svd",
    "topk",
    "var",
]


def isnan(inp: Tensor) -> Tensor:
    r"""Returns a new tensor representing if each element is ``NaN`` or not.

    Args:
        inp: input tensor.

    Returns:
        result tensor.

    Examples:

        .. testcode::

            from megengine import tensor
            import megengine.functional as F

            x = tensor([1, float("nan"), 0])
            print(F.isnan(x).numpy())

        Outputs:

        .. testoutput::

            [False  True False]
    """
    return inp != inp


def isinf(inp: Tensor) -> Tensor:
    r"""Returns a new tensor representing if each element is ``Inf`` or not.

    Args:
        inp: input tensor.

    Returns:
        result tensor.

    Examples:

        .. testcode::

            from megengine import tensor
            import megengine.functional as F

            x = tensor([1, float("inf"), 0])
            print(F.isinf(x).numpy())

        Outputs:

        .. testoutput::

            [False  True False]
    """
    return abs(inp).astype("float32") == float("inf")


def sign(inp: Tensor):
    r"""Returns a new tensor representing the sign of each element in input tensor.

    Args:
        inp: Tensor:

    Returns:
        the sign of input tensor.

    Examples:

        .. testcode::

            from megengine import tensor
            import megengine.functional as F

            x = tensor([1, -1, 0])
            print(F.sign(x).numpy())

        Outputs:

        .. testoutput::

            [ 1 -1  0]
    """
    return (inp > 0).astype(inp.dtype) - (inp < 0).astype(inp.dtype)


def sum(
    inp: Tensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
) -> Tensor:
    r"""Returns the sum of input tensor along given axis. If axis is a list of dimensions,
    reduce over all of them.

    Args:
        inp: input tensor.
        axis: dimension to reduce. If None, all dimensions will be reduced.
            Default: None
        keepdims: whether the output tensor has axis retained or not.
            Default: False

    Returns:
        output tensor.

    Examples:

        .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            x = tensor(np.arange(1, 7, dtype=np.int32).reshape(2, 3))
            out = F.sum(x)
            print(out.numpy())

        Outputs:

        .. testoutput::

            21
    """
    return inp.sum(axis=axis, keepdims=keepdims)


def prod(
    inp: Tensor, axis: Optional[Union[int, Sequence[int]]] = None, keepdims=False
) -> Tensor:
    r"""Returns the product of input tensor along given axis. If axis is a list of dimensions,
    reduce over all of them.

    Args:
        inp: input tensor.
        axis: dimension to reduce. If None, all dimensions will be reduced. Default: None
        keepdims: whether the output tensor has axis retained or not. Default: False

    Returns:
        output tensor.

    Examples:

        .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            x = tensor(np.arange(1, 7, dtype=np.int32).reshape(2, 3))
            out = F.prod(x)
            print(out.numpy())

        Outputs:

        .. testoutput::

            720
    """
    return inp.prod(axis=axis, keepdims=keepdims)


def mean(
    inp: Tensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
) -> Tensor:
    r"""Returns the mean value of input tensor along
    given axis. If axis is a list of dimensions,
    reduce over all of them.

    Args:
        inp: input tensor.
        axis: dimension to reduce. If None, all dimensions will be reduced. Default: None
        keepdims: whether the output tensor has axis retained or not. Default: False

    Returns:
        output tensor.

    Examples:

        .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            x = tensor(np.arange(1, 7, dtype=np.int32).reshape(2, 3))
            out = F.mean(x)
            print(out.numpy())

        Outputs:

        .. testoutput::

            3.5
    """
    return inp.mean(axis=axis, keepdims=keepdims)


def var(
    inp: Tensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
) -> Tensor:
    r"""Returns the variance value of input tensor along
    given axis. If axis is a list of dimensions,
    reduce over all of them.

    Args:
        inp: input tensor.
        axis: dimension to reduce. If None, all dimensions will be reduced. Default: None
        keepdims: whether the output tensor has axis retained or not. Default: False

    Returns:
        output tensor.

    Examples:

        .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            data = tensor(np.arange(1, 7, dtype=np.float32).reshape(2, 3))
            out = F.var(data)
            print(out.numpy().round(decimals=4))

        Outputs:

        .. testoutput::

            2.9167
    """
    if axis is None:
        m = mean(inp, axis=axis, keepdims=False)
    else:
        m = mean(inp, axis=axis, keepdims=True)
    v = inp - m
    return mean(v ** 2, axis=axis, keepdims=keepdims)


def std(
    inp: Tensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
) -> Tensor:
    r"""Returns the standard deviation of input tensor along
    given axis. If axis is a list of dimensions,
    reduce over all of them.

    Args:
        inp: input tensor.
        axis: dimension to reduce. If None, all dimensions will be reduced. Default: None
        keepdims: whether the output tensor has axis retained or not. Default: False

    Returns:
        output tensor.

    Examples:

        .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            data = tensor(np.arange(1, 7, dtype=np.float32).reshape(2, 3))
            out = F.std(data, axis=1)
            print(out.numpy().round(decimals=4))

        Outputs:

        .. testoutput::

            [0.8165 0.8165]
    """
    return var(inp, axis=axis, keepdims=keepdims) ** 0.5


def min(
    inp: Tensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
) -> Tensor:
    r"""Returns the min value of input tensor along
    given axis. If axis is a list of dimensions,
    reduce over all of them.

    Args:
        inp: input tensor.
        axis: dimension to reduce. If None, all dimensions will be reduced. Default: None
        keepdims: whether the output tensor has axis retained or not. Default: False

    Returns:
        output tensor.

    Examples:

        .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            x = tensor(np.arange(1, 7, dtype=np.int32).reshape(2,3))
            out = F.min(x)
            print(out.numpy())

        Outputs:

        .. testoutput::

            1
    """
    return inp.min(axis=axis, keepdims=keepdims)


def max(
    inp: Tensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
) -> Tensor:
    r"""Returns the max value of the input tensor along
    given axis. If axis is a list of dimensions,
    reduce over all of them.

    Args:
        inp: input tensor.
        axis: dimension to reduce. If None, all dimensions will be reduced. Default: None
        keepdims: whether the output tensor has axis retained or not. Default: False

    Returns:
        output tensor.

    Examples:

        .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            x = tensor(np.arange(1, 7, dtype=np.int32).reshape(2,3))
            out = F.max(x)
            print(out.numpy())

        Outputs:

        .. testoutput::

            6
    """
    return inp.max(axis=axis, keepdims=keepdims)


def norm(
    inp: Tensor, ord: float = None, axis: int = None, keepdims=False,
):
    r"""Calculates ``p``-norm of input tensor along
    given axis.

    Args:
        inp: input tensor.
        ord: power of value applied to inp. Default: 2
        axis: dimension to reduce. If None, input must be a vector. Default: None
        keepdims: whether the output tensor has axis retained or not. Default: False

    Returns:
        output tensor.

    Examples:

        .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            x = tensor(np.arange(-3, 3, dtype=np.float32))
            out = F.norm(x)
            print(out.numpy().round(decimals=4))

        Outputs:

        .. testoutput::

            4.3589
    """
    if axis is None:
        if inp.ndim != 1:
            raise TypeError("axis is required unless input is a vector")
    if ord is None:
        ord = 2
    if ord == 0:
        return sum(inp != 0, axis=axis, keepdims=keepdims)
    if ord == math.inf:
        return max(abs(inp))
    if ord == -math.inf:
        return min(abs(inp))
    return sum(abs(inp) ** ord, axis=axis, keepdims=keepdims) ** (1.0 / ord)


def argmin(
    inp: Tensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
) -> Tensor:
    r"""Returns the indices of the minimum values along
    given axis. If axis is a list of dimensions,
    reduce over all of them.

    Args:
        inp: input tensor.
        axis: dimension to reduce. If None, all dimensions will be reduced. Default: None
        keepdims: whether the output tensor has axis retained or not. Default: False

    Returns:
        output tensor.

    Examples:

        .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            x = tensor(np.arange(1, 7, dtype=np.int32).reshape(2,3))
            out = F.argmin(x)
            print(out.numpy())

        Outputs:

        .. testoutput::

            0
    """
    if axis is None:
        assert not keepdims, "can not set axis=None and keepdims=True"
        inp = inp.flatten()
        axis = 0

    axis = _normalize_axis(inp.ndim, axis, reverse=True)
    if isinstance(axis, collections.abc.Iterable):

        for ai in axis:
            op = builtin.Argmin(axis=ai)
            (inp,) = apply(op, inp)

            if not keepdims:
                inp = squeeze(inp, ai)

        return inp

    op = builtin.Argmin(axis=axis)
    (result,) = apply(op, inp)
    if not keepdims:
        result = squeeze(result, axis)
    return result


def argmax(
    inp: Tensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
) -> Tensor:
    r"""Returns the indices of the maximum values along
    given axis. If axis is a list of dimensions,
    reduce over all of them.

    Args:
        inp: input tensor.
        axis: dimension to reduce. If None, all dimensions will be reduced. Default: None
        keepdims: whether the output tensor has axis retained or not. Default: False

    Returns:
        output tensor.

    Examples:

        .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            x = tensor(np.arange(1, 7, dtype=np.int32).reshape(2,3))
            out = F.argmax(x)
            print(out.numpy())

        Outputs:

        .. testoutput::

            5
    """
    if axis is None:
        assert not keepdims, "can not set axis=None and keepdims=True"
        inp = inp.flatten()
        axis = 0
    axis = _normalize_axis(inp.ndim, axis, reverse=True)

    if isinstance(axis, collections.abc.Iterable):

        for ai in axis:
            op = builtin.Argmax(axis=ai)
            (inp,) = apply(op, inp)

            if not keepdims:
                inp = squeeze(inp, ai)

        return inp

    op = builtin.Argmax(axis=axis)
    (result,) = apply(op, inp)
    if not keepdims:
        result = squeeze(result, axis)
    return result


def normalize(
    inp: Tensor, ord: float = None, axis: int = None, eps: float = 1e-12,
) -> Tensor:
    r"""Performs :math:`L_p` normalization of input tensor along
    given axis.

    For a tensor of shape :math:`(n_0, ..., n_{dim}, ..., n_k)`, each
    :math:`n_{dim}` -element vector :math:`v` along dimension :attr:`axis` is transformed as:

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}.

    Args:
        inp: input tensor.
        ord: power of value applied to input tensor. Default: 2
        axis: dimension to reduce.If None, input must be a vector. Default: None
        eps: a small value to avoid division by zero. Default: 1e-12

    Returns:
        normalized output tensor.
    """
    if axis is None:
        return inp / clip(norm(inp, ord, axis), lower=eps)
    else:
        return inp / clip(norm(inp, ord, axis, keepdims=True), lower=eps)


def argsort(inp: Tensor, descending: bool = False) -> Tensor:
    r"""Returns the indices that would sort the input tensor.

    Args:
        inp: input tensor. If it's 2d, the result would be array of indices show how to sort each row in the input tensor.
        descending: sort in descending order, where the largest comes first. Default: False
        inp: Tensor:
        descending: bool:

    Returns:
        indices of int32 indicates how to sort the input.

    Examples:

        .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            x = tensor(np.array([1,2], dtype=np.float32))
            indices = F.argsort(x)
            print(indices.numpy())

        Outputs:

        .. testoutput::

            [0 1]
    """
    assert len(inp.shape) <= 2, "Input should be 1d or 2d"
    if descending:
        order = "descending"
    else:
        order = "ascending"

    op = builtin.Argsort(order=order)
    if len(inp.shape) == 1:
        inp = inp.reshape(1, -1)
        _, result = apply(op, inp)
        return result[0]
    _, result = apply(op, inp)
    return result


def sort(inp: Tensor, descending: bool = False) -> Tuple[Tensor, Tensor]:
    r"""Returns sorted tensor and the indices would sort the input tensor.

    Args:
        inp: input tensor. If it's 2d, the result would be sorted by row.
        descending: sort in descending order, where the largest comes first. Default: False

    Returns:
        tuple of two tensors `(sorted_tensor, indices_of_int32)`.

    Examples:

        .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            x = tensor(np.array([1,2], dtype=np.float32))
            out, indices = F.sort(x)
            print(out.numpy())

        Outputs:

        .. testoutput::

            [1. 2.]
    """
    assert len(inp.shape) <= 2, "Input should be 1d or 2d"
    if descending:
        order = "descending"
    else:
        order = "ascending"

    op = builtin.Argsort(order=order)
    if len(inp.shape) == 1:
        inp = inp.reshape(1, -1)
        tns, ind = apply(op, inp)
        return tns[0], ind[0]
    tns, ind = apply(op, inp)
    return tns, ind


def topk(
    inp: Tensor,
    k: int,
    descending: bool = False,
    kth_only: bool = False,
    no_sort: bool = False,
) -> Tuple[Tensor, Tensor]:
    r"""Selects the ``Top-K`` (by default) smallest elements of 2d matrix by row.

    Args:
        inp: input tensor. If input tensor is 2d, each row will be sorted.
        k: number of elements needed.
        descending: if True, return the largest elements instead. Default: False
        kth_only: if True, only the k-th element will be returned. Default: False
        no_sort: if True, the returned elements can be unordered. Default: False

    Returns:
        tuple of two tensors ``(topk_tensor, indices_of_int32)``

    Examples:

        .. testcode::

            import numpy as np
            from megengine import tensor
            import  megengine.functional as F

            x = tensor(np.array([2, 4, 6, 8, 7, 5, 3, 1], dtype=np.float32))
            top, indices = F.topk(x, 5)
            print(top.numpy(), indices.numpy())

        Outputs:

        .. testoutput::

            [1. 2. 3. 4. 5.] [7 0 6 1 5]
    """
    if descending:
        k = -k

    if kth_only:
        mode = "kth_only"
    elif no_sort:
        mode = "value_idx_nosort"
    else:
        mode = "value_idx_sorted"
    op = builtin.TopK(mode=mode)

    if not isinstance(k, Tensor):
        (k,) = Const(k, dtype="int32", device=inp.device)()

    if len(inp.shape) == 1:
        if kth_only:
            (tns,) = apply(op, expand_dims(inp, 0), k)
            # FIXME:
            # could use a dedicated kernel
            # gradient may be routed to other indices if k-th value is not unique
            ind = argmax((tns == inp).astype("int8"))
            tns = squeeze(tns, 0)
        else:
            tns, ind = apply(op, expand_dims(inp, 0), k)
            tns = squeeze(tns, 0)
            ind = squeeze(ind, 0)
    else:
        if kth_only:
            (tns,) = apply(op, inp, k)
            # FIXME: same as above
            ind = argmax((expand_dims(tns, 1) == inp).astype("int8"), 1)
        else:
            tns, ind = apply(op, inp, k)

    return tns, ind


def matinv(inp: Tensor) -> Tensor:
    r"""Computes the inverse of a batch of matrices; input must has shape [..., n, n].

    Args:
        inp: input tensor.

    Returns:
        output tensor.


    Examples:

        .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            data = tensor([[1.0, 0.0], [1.0, 1.0]])
            out = F.matinv(data)
            print(out.numpy())

        Outputs:

        .. testoutput::

            [[ 1.  0.]
             [-1.  1.]]
    """

    (result,) = apply(builtin.MatrixInverse(), inp)
    return result


class _Hashable:
    def __init__(self, value) -> None:
        self.value = value

    def __hash__(self) -> int:
        return hash(str(self.value))

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, _Hashable):
            return False
        return self.value == o.value


@lru_cache(maxsize=None)
def _get_extentedMatrixMulOp(
    device, dtype, dim1, dim2, transpose_a, transpose_b, compute_mode, format, strategy,
):
    @subgraph("extentedMatrixMulOp", dtype, device, 2, gopt_level=2)
    def extentedMatrixMulOp(inputs, f, c):
        assert len(inputs) == 2
        inp1, inp2 = inputs
        _dim1, _dim2 = dim1, dim2

        def build_shape_head(shape, idx=-1):
            # shape[:idx]
            return f(
                builtin.Subtensor(items=[[0, False, True, False, False]]),
                shape,
                c(idx, "int32"),
            )

        def build_shape_tail(shape, idx=-1):
            # shape[idx:]
            return f(
                builtin.Subtensor(items=[[0, True, False, False, False]]),
                shape,
                c(idx, "int32"),
            )

        remove_row, remove_col = False, False
        if _dim1 == 1:
            _dim1 = 2
            remove_row = True
        if _dim2 == 1:
            _dim2 = 2
            remove_col = True

        if remove_row:
            inp1 = f(builtin.AddAxis(axis=[0,]), inp1)
        if remove_col:
            inp2 = f(builtin.AddAxis(axis=[1,]), inp2)

        shape1 = f(GetVarShape(), inp1)
        shape2 = f(GetVarShape(), inp2)
        if _dim1 > 2:
            inp1 = f(
                builtin.Reshape(),
                inp1,
                f(
                    builtin.Concat(axis=0, comp_node=device),
                    f(builtin.Reduce(mode="product", axis=0), build_shape_head(shape1)),
                    build_shape_tail(shape1),
                ),
            )
        if _dim2 > 2:
            inp2 = f(
                builtin.Reshape(),
                inp2,
                f(
                    builtin.Concat(axis=0, comp_node=device),
                    f(builtin.Reduce(mode="product", axis=0), build_shape_head(shape2)),
                    build_shape_tail(shape2),
                ),
            )
        op = builtin.MatrixMul(
            transposeA=transpose_a,
            transposeB=transpose_b,
            compute_mode=compute_mode,
            format=format,
            strategy=strategy.value,
        )
        result = f(op, inp1, inp2)
        result_shape = f(GetVarShape(), result)
        if _dim1 > 2:
            result = f(
                builtin.Reshape(),
                result,
                f(
                    builtin.Concat(axis=0, comp_node=device),
                    build_shape_head(shape1),
                    build_shape_tail(result_shape),
                ),
            )
        if _dim2 > 2:
            result = f(
                builtin.Reshape(),
                result,
                f(
                    builtin.Concat(axis=0, comp_node=device),
                    build_shape_head(shape2),
                    build_shape_tail(result_shape),
                ),
            )
        maxdim = _dim1 if _dim1 > _dim2 else _dim2
        if remove_row:
            result = f(builtin.RemoveAxis(axis=[maxdim - 2]), result)
        if remove_col:
            result = f(builtin.RemoveAxis(axis=[maxdim - 1]), result)
        return (result,), (True,)

    return extentedMatrixMulOp


@lru_cache(maxsize=None)
def _get_extentedBatchedMatrixMulOp(
    device, dtype, dim1, dim2, transpose_a, transpose_b, compute_mode, format, strategy,
):
    @subgraph("extentedBatchedMatrixMulOp", dtype, device, 2, gopt_level=2)
    def extentedBatchedMatrixMulOp(inputs, f, c):
        assert len(inputs) == 2
        inp1, inp2 = inputs
        _dim1, _dim2 = dim1, dim2

        def build_shape_head(shape, idx=-2):
            # shape[:idx]
            return f(
                builtin.Subtensor(items=[[0, False, True, False, False]]),
                shape,
                c(idx, "int32"),
            )

        def build_shape_tail(shape, idx=-2):
            # shape[idx:]
            return f(
                builtin.Subtensor(items=[[0, True, False, False, False]]),
                shape,
                c(idx, "int32"),
            )

        remove_row, remove_col = False, False
        if _dim1 == 1:
            _dim1 = 2
            remove_row = True
        if _dim2 == 1:
            _dim2 = 2
            remove_col = True

        if remove_row:
            inp1 = f(builtin.AddAxis(axis=[0,]), inp1)
        if remove_col:
            inp2 = f(builtin.AddAxis(axis=[1,]), inp2)
        shape1 = f(GetVarShape(), inp1)
        shape2 = f(GetVarShape(), inp2)
        maxdim = _dim1 if _dim1 > _dim2 else _dim2
        if _dim1 > _dim2:
            # broadcast
            shape2 = f(
                builtin.Concat(axis=0, comp_node=device),
                build_shape_head(shape1, idx=-_dim2),  # shape1[:-_dim2]
                shape2,
            )
            inp2 = f(builtin.Broadcast(), inp2, shape2)
            batch_shape = build_shape_head(shape1)
        if _dim2 > _dim1:
            # broadcast
            shape1 = f(
                builtin.Concat(axis=0, comp_node=device),
                build_shape_head(shape2, idx=-_dim1),  # shape2[:-_dim1]
                shape1,
            )
            inp1 = f(builtin.Broadcast(), inp1, shape1)
            batch_shape = build_shape_head(shape2)
        if _dim1 == _dim2:
            batch_shape = build_shape_head(shape1)

        # compress inputs to 3d
        if maxdim > 3:
            inp1 = f(
                builtin.Reshape(),
                inp1,
                f(
                    builtin.Concat(axis=0, comp_node=device),
                    f(builtin.Reduce(mode="product", axis=0), batch_shape),
                    build_shape_tail(shape1),
                ),
            )
            inp2 = f(
                builtin.Reshape(),
                inp2,
                f(
                    builtin.Concat(axis=0, comp_node=device),
                    f(builtin.Reduce(mode="product", axis=0), batch_shape),
                    build_shape_tail(shape2),
                ),
            )
        op = builtin.BatchedMatrixMul(
            transposeA=transpose_a,
            transposeB=transpose_b,
            compute_mode=compute_mode,
            format=format,
            strategy=strategy.value,
        )
        result = f(op, inp1, inp2)

        if maxdim > 3:
            result = f(
                builtin.Reshape(),
                result,
                f(
                    builtin.Concat(axis=0, comp_node=device),
                    batch_shape,
                    build_shape_tail(f(GetVarShape(), result)),
                ),
            )
        if remove_row:
            result = f(builtin.RemoveAxis(axis=[maxdim - 2]), result)
        if remove_col:
            result = f(builtin.RemoveAxis(axis=[maxdim - 1]), result)
        return (result,), (True,)

    return extentedBatchedMatrixMulOp


def matmul(
    inp1: Tensor,
    inp2: Tensor,
    transpose_a=False,
    transpose_b=False,
    compute_mode="default",
    format="default",
) -> Tensor:
    r"""Performs a matrix multiplication of the matrices ``inp1`` and ``inp2``.

    With different inputs dim, this function behaves differently:

    * Both 1-D tensor, simply forward to ``dot``.
    * Both 2-D tensor, normal matrix multiplication.
    * If one input tensor is 1-D, matrix vector multiplication.
    * If at least one tensor are 3-dimensional or >3-dimensional, the other tensor should have dim >= 2,
      the batched matrix-matrix is returned, and the tensor with smaller dimension will be broadcasted.
      For example:

      * inp1: `(n, k, m)`, inp2: `(n, m, p)`, return: `(n, k, p)`
      * inp1: `(n, k, m)`, inp2: `(m, p)`, return: `(n, k, p)`
      * inp1: `(n, j, k, m)`, inp2: `(n, j, m, p)`, return: `(n, j, k, p)`

    Args:
        inp1: first matrix to be multiplied.
        inp2: second matrix to be multiplied.

    Returns:
        output tensor.

    Examples:

        .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            data1 = tensor(np.arange(0, 6, dtype=np.float32).reshape(2, 3))
            data2 = tensor(np.arange(0, 6, dtype=np.float32).reshape(3, 2))
            out = F.matmul(data1, data2)
            print(out.numpy())

        Outputs:

        .. testoutput::

            [[10. 13.]
             [28. 40.]]
    """
    if amp._enabled:
        compute_mode = "float32"
        inp1, inp2 = cast_tensors(inp1, inp2)
    else:
        dtype = dtype_promotion(inp1, inp2)
        if inp1.dtype != dtype:
            inp1 = inp1.astype(dtype)
        if inp2.dtype != dtype:
            inp2 = inp2.astype(dtype)

    dim1, dim2 = inp1.ndim, inp2.ndim
    assert dim1 > 0 and dim2 > 0
    maxdim = dim1 if dim1 > dim2 else dim2
    compute_mode = _config._get_actual_op_param(compute_mode, _config.__compute_mode)
    if dim1 == 1 and dim2 == 1:  # dispatch to Dot
        return dot(inp1, inp2)
    elif maxdim <= 2 or dim2 <= 2:  # dispath to MatrixMul
        extentedMatrixMulOp = _get_extentedMatrixMulOp(
            inp1.device,
            inp1.dtype,
            dim1,
            dim2,
            transpose_a,
            transpose_b,
            compute_mode,
            format,
            strategy=_Hashable(get_execution_strategy()),
        )
        (result,) = apply(extentedMatrixMulOp(), inp1, inp2)
        return result
    else:  # dispath to BatchedMatrixMul
        extentedBatchedMatrixMulOp = _get_extentedBatchedMatrixMulOp(
            inp1.device,
            inp1.dtype,
            dim1,
            dim2,
            transpose_a,
            transpose_b,
            compute_mode,
            format,
            strategy=_Hashable(get_execution_strategy()),
        )
        (result,) = apply(extentedBatchedMatrixMulOp(), inp1, inp2)
        return result


def dot(inp1: Tensor, inp2: Tensor) -> Tensor:
    r"""Computes dot-product of two vectors ``inp1`` and ``inp2``.
    inputs must be 1-dimensional or scalar. A scalar input is automatically broadcasted.
    Refer to :func:`~.matmul` for more general usage.

    Args:
        inp1: first vector.
        inp2: second vector.

    Returns:
        output value.

    Examples:

        .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            data1 = tensor(np.arange(0, 6, dtype=np.float32))
            data2 = tensor(np.arange(0, 6, dtype=np.float32))
            out = F.dot(data1, data2)
            print(out.numpy())

        Outputs:

        .. testoutput::

            55.
    """
    op = builtin.Dot()
    assert (
        inp1.ndim <= 1 and inp2.ndim <= 1
    ), "Input tensors for dot must be 1-dimensional or scalar"
    (result,) = apply(op, inp1, inp2)
    setscalar(result)
    return result


def svd(inp: Tensor, full_matrices=False, compute_uv=True) -> Tensor:
    r"""Computes the singular value decompositions of input matrix.

    Args:
        inp: input matrix, must has shape `[..., M, N]`.

    Returns:
        output matrices, `(U, sigma, V)`.

    Examples:

        .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            x = tensor(np.arange(0, 6, dtype=np.float32).reshape(2,3))
            _, y, _ = F.svd(x)
            print(y.numpy().round(decimals=3))

        Outputs:

        .. testoutput::

            [7.348 1.   ]
    """
    op = builtin.SVD(full_matrices=full_matrices, compute_uv=compute_uv)
    U, sigma, V = apply(op, inp)
    return U, sigma, V


def _check_non_finite(inp: Tensor) -> Tensor:
    r"""Check whether input contains infinite or nan value.

    Args:
        inp: a tensor to be checked.

    Returns:
        a int32 scalar tensor, 0 for False and 1 for True.
    """
    op = builtin.CheckNonFinite()
    (oup,) = apply(op, inp.reshape(-1).astype("float32"))
    oup._setscalar()
    return oup
