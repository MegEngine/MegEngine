# -*- coding: utf-8 -*-
import collections
import math
from typing import Iterable, Optional, Sequence, Tuple, Union

from ..core._imperative_rt.core2 import Const, apply
from ..core._imperative_rt.ops import SubgraphBuilder as _SubgraphBuilder
from ..core.ops import builtin
from ..core.tensor.array_method import _matmul
from ..core.tensor.utils import _normalize_axis
from ..tensor import Tensor
from ..utils.deprecation import deprecated_kwargs_default
from .elemwise import clip
from .tensor import expand_dims, squeeze

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
        >>> x = Tensor([1, float("nan"), 0])
        >>> F.isnan(x).numpy()
        array([False,  True, False])
    """
    return inp != inp


def isinf(inp: Tensor) -> Tensor:
    r"""Returns a new tensor representing if each element is ``Inf`` or not.

    Args:
        inp: input tensor.

    Returns:
        result tensor.

    Examples:
        >>> x = Tensor([1, float("inf"), 0])
        >>> F.isinf(x).numpy()
        array([False,  True, False])
    """
    return abs(inp).astype("float32") == float("inf")


def sign(inp: Tensor):
    r"""Returns a new tensor representing the sign of each element in input tensor.

    Args:
        inp: Tensor:

    Returns:
        the sign of input tensor.

    Examples:
        >>> x = Tensor([1, -1, 0])
        >>> F.sign(x)
        Tensor([ 1 -1  0], dtype=int32, device=xpux:0)
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
        >>> import numpy as np
        >>> x = Tensor(np.arange(1, 7, dtype=np.int32).reshape(2, 3))
        >>> F.sum(x)
        Tensor(21, dtype=int32, device=xpux:0)
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
        >>> import numpy as np
        >>> x = Tensor(np.arange(1, 7, dtype=np.int32).reshape(2, 3))
        >>> F.prod(x)
        Tensor(720, dtype=int32, device=xpux:0)
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
        >>> import numpy as np
        >>> x = Tensor(np.arange(1, 7, dtype=np.int32).reshape(2, 3))
        >>> out = F.mean(x)
        >>> out.numpy()
        array(3.5, dtype=float32)
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
        >>> import numpy as np
        >>> data = Tensor(np.arange(1, 7, dtype=np.float32).reshape(2, 3))
        >>> out = F.var(data)
        >>> out.numpy().round(decimals=4)
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
        >>> import numpy as np
        >>> data = Tensor(np.arange(1, 7, dtype=np.float32).reshape(2, 3))
        >>> out = F.std(data, axis=1)
        >>> out.numpy().round(decimals=4)
        array([0.8165, 0.8165], dtype=float32)
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
        >>> import numpy as np
        >>> x = Tensor(np.arange(1, 7, dtype=np.int32).reshape(2,3))
        >>> F.min(x)
        Tensor(1, dtype=int32, device=xpux:0)
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
        >>> import numpy as np
        >>> x = Tensor(np.arange(1, 7, dtype=np.int32).reshape(2,3))
        >>> F.max(x)
        Tensor(6, dtype=int32, device=xpux:0)
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
        >>> import numpy as np
        >>> x = Tensor(np.arange(-3, 3, dtype=np.float32))
        >>> out = F.norm(x)
        >>> out.numpy().round(decimals=4)
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
        >>> import numpy as np
        >>> x = Tensor(np.arange(1, 7, dtype=np.int32).reshape(2,3))
        >>> F.argmin(x)
        Tensor(0, dtype=int32, device=xpux:0)
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
        >>> import numpy as np
        >>> x = Tensor(np.arange(1, 7, dtype=np.int32).reshape(2,3))
        >>> F.argmax(x)
        Tensor(5, dtype=int32, device=xpux:0)
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
        >>> import numpy as np
        >>> x = Tensor(np.array([1,2], dtype=np.float32))
        >>> F.argsort(x)
        Tensor([0 1], dtype=int32, device=xpux:0)
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
        >>> import numpy as np
        >>> x = Tensor(np.array([1,2], dtype=np.float32))
        >>> out, indices = F.sort(x)
        >>> out.numpy()
        array([1., 2.], dtype=float32)
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


@deprecated_kwargs_default("1.12", "descending", 3)
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
        >>> import numpy as np
        >>> x = Tensor(np.array([2, 4, 6, 8, 7, 5, 3, 1], dtype=np.float32))
        >>> top, indices = F.topk(x, 5, descending=False)
        >>> print(top.numpy(), indices.numpy())
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
        k = Const(k, "int32", inp.device, None)

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
        >>> import numpy as np
        >>> data = Tensor([[1.0, 0.0], [1.0, 1.0]])
        >>> out = F.matinv(data)
        >>> out.numpy()
        array([[ 1.,  0.],
               [-1.,  1.]], dtype=float32)
    """

    (result,) = apply(builtin.MatrixInverse(), inp)
    return result


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
        >>> import numpy as np
        >>> data1 = Tensor(np.arange(0, 6, dtype=np.float32).reshape(2, 3))
        >>> data2 = Tensor(np.arange(0, 6, dtype=np.float32).reshape(3, 2))
        >>> out = F.matmul(data1, data2)
        >>> out.numpy()
        array([[10., 13.],
               [28., 40.]], dtype=float32)
    """
    return _matmul(inp1, inp2, transpose_a, transpose_b, compute_mode, format)


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
        >>> import numpy as np
        >>> data1 = Tensor(np.arange(0, 6, dtype=np.float32))
        >>> data2 = Tensor(np.arange(0, 6, dtype=np.float32))
        >>> out = F.dot(data1, data2)
        >>> out.numpy()
        array(55., dtype=float32)
    """
    op = builtin.Dot()
    assert (
        inp1.ndim <= 1 and inp2.ndim <= 1
    ), "Input tensors for dot must be 1-dimensional or scalar"
    (result,) = apply(op, inp1, inp2)
    return result


def svd(inp: Tensor, full_matrices=False, compute_uv=True) -> Tensor:
    r"""Returns a singular value decomposition ``A = USVh`` of a matrix (or a stack of matrices) ``x`` , where ``U`` is a matrix (or a stack of matrices) with orthonormal columns, ``S`` is a vector of non-negative numbers (or stack of vectors), and ``Vh`` is a matrix (or a stack of matrices) with orthonormal rows.

    Args:
        x (Tensor): A input real tensor having the shape ``(..., M, N)`` with ``x.ndim >= 2`` .
        full_matrices (bool, optional): If ``False`` , ``U`` and ``Vh`` have the shapes  ``(..., M, K)`` and ``(..., K, N)`` , respectively, where ``K = min(M, N)`` . If ``True`` , the shapes  are ``(..., M, M)`` and ``(..., N, N)`` , respectively. Default: ``False`` . 
        compute_uv (bool, optional): Whether or not to compute ``U`` and ``Vh`` in addition to ``S`` . Default: ``True`` .

    Note:
        * naive does not support ``full_matrices`` and ``compute_uv`` as ``True`` .

    Returns:
        Returns a tuple ( ``U`` , ``S`` , ``Vh`` ), which are  SVD factors ``U`` , ``S``, ``Vh`` of  input matrix ``x``. ( ``U`` , ``Vh`` only returned when ``compute_uv`` is True). 
            ``U`` contains matrices orthonormal columns (i.e., the columns are left singular vectors). If ``full_matrices`` is ``True`` , the array must have shape ``(..., M, M)`` . If ``full_matrices`` is ``False`` , the array must have shape ``(..., M, K)`` , where ``K = min(M, N)`` .

    Examples:
        >>> import numpy as np
        >>> x = Tensor(np.random.randn(9, 6))
        >>> y = Tensor(np.random.randn(2, 7, 8, 3))
        >>> U, S, Vh = F.svd(x, full_matrices=False)
        >>> print(U._tuple_shape, S._tuple_shape, Vh._tuple_shape)
        (9, 6) (6,) (6, 6)
        >>> u, s, vh = F.svd(y, full_matrices=False)
        >>> print(u._tuple_shape, s._tuple_shape, vh._tuple_shape)
        (2, 7, 8, 3) (2, 7, 3) (2, 7, 3, 3)
    """
    op = builtin.SVD(full_matrices=full_matrices, compute_uv=compute_uv)
    U, S, Vh = apply(op, inp)
    return U, S, Vh


def _check_non_finite(inps: Iterable[Tensor], scale=1.0) -> Tensor:
    r"""Check whether input contains infinite or nan value.

    Args:
        inp: a tensor to be checked.

    Returns:
        a int32 scalar tensor, 0 for False and 1 for True.
    """
    op = builtin.CheckNonFinite(scale=scale)
    oups = apply(op, *inps)
    out = oups[-1]
    for i in range(len(inps)):
        inps[i]._reset(oups[i])

    return out
