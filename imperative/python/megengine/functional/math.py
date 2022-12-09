# -*- coding: utf-8 -*-
import collections
import math
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from ..core._imperative_rt.core2 import Const, apply
from ..core._imperative_rt.ops import SubgraphBuilder as _SubgraphBuilder
from ..core.ops import builtin
from ..core.tensor.array_method import _elwise, _matmul
from ..core.tensor.utils import _normalize_axis
from ..tensor import Tensor
from ..utils.deprecation import deprecated_kwargs_default
from .elemwise import _elemwise_multi_type, clip
from .tensor import broadcast_to, expand_dims, squeeze

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


# TODO: Should be moved to elemwise - logical functions


def isnan(inp: Tensor) -> Tensor:
    r"""Element-wise ``NaN`` check.
    
    Tests each element :math:`x_i` of the input tensor :math:`x` to determine whether the element is ``NaN``.

    Args:
        inp: input tensor. Should have a numeric data type.

    Returns:
        a tensor containing test results.
        An element out is ``True`` if :math:`x_i` is ``NaN`` and ``False`` otherwise.
        The returned array should have a data type of bool.

    Examples:
        
        >>> F.isnan(Tensor(1))
        Tensor(False, dtype=bool, device=xpux:0)

        .. TODO: Remove these comments when _elemwise_multi_type support scalar input
        .. >>> F.isnan(Tensor(float("nan")))
        .. Tensor(True, dtype=bool, device=xpux:0)

        Element-wise isnan:

        >>> x = Tensor([1, float("nan"), 0])
        >>> F.isnan(x)
        Tensor([False  True False], dtype=bool, device=xpux:0)
    """
    if not np.issubdtype(inp.dtype, np.floating):
        return broadcast_to(Tensor(False), inp.shape)
    return _elemwise_multi_type(inp, mode="isnan", dtype="bool")


def isinf(inp: Tensor) -> Tensor:
    r"""Element-wise ``infinity`` check.
    
    Tests each element :math:`x_i` of the input tensor :math:`x` to determine
    whether the element is if equal to positive or negative infinity.

    Args:
        inp: input tensor. Should have a numeric data type.

    Returns:
        a tensor containing test results.
        An element out is ``True`` if :math:`x_i` is either positive or negative infinity and ``False`` otherwise.
        The returned array should have a data type of bool.

    Examples:

        >>> F.isinf(Tensor(1))
        Tensor(False, dtype=bool, device=xpux:0)

        .. TODO: Remove these comments when _elemwise_multi_type support scalar input
        .. >>> F.isinf(Tensor(float("inf")))
        .. Tensor(True, dtype=bool, device=xpux:0)

        Element-wise isinf: 

        >>> x = Tensor([1, float("inf"), 0])
        >>> F.isinf(x)
        Tensor([False  True False], dtype=bool, device=xpux:0)
    """
    if not np.issubdtype(inp.dtype, np.floating):
        return broadcast_to(Tensor(False), inp.shape)
    return _elemwise_multi_type(inp, mode="isinf", dtype="bool")


# TODO: Should be moved to elemwise - arithmetic operations


def sign(x: Tensor):
    r"""Element-wise sign.

    Returns an indication of the sign of a number for each element :math:`x_i` of the input tensor :math:`x`.

    Args:
        inp: input tensor. Should have a numeric data type.

    Returns:
        a tensor containing the evaluated result for each element in :math:`x`.
        The returned array must have the same data type as :math:`x`.

    Examples:

        Element-wise sign:

        >>> x = Tensor([1, -1, 0])
        >>> F.sign(x)
        Tensor([ 1 -1  0], dtype=int32, device=xpux:0)
    """
    return _elwise(x, mode=builtin.Elemwise.Mode.SIGN)


def sum(
    inp: Tensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
) -> Tensor:
    r"""Calculates the sum of tensor elements over a given axis (or axes).

    Args:
        inp: input tensor. Should have a numeric data type.
        axis: axis or axes along which sums must be computed.
            By default, the sum must be computed over the entire tensor.
            If a sequence of integers, sums must be computed over multiple axes.
        keepdims: if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions,
            and, accordingly, the result must be compatible with the input tensor (see :ref:`broadcasting-rule`).
            Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result.

    Returns:
        if the sum was computed over the entire tensor, a zero-dimensional tensor containing the sum;
        otherwise, a tensor containing the sums.
        The returned tensor must have a data type determined by :ref:`dtype-promotion`.

    .. admonition:: Special Cases

       Let ``N`` equal the number of elements over which to compute the sum.

       * If ``N`` is 0, the sum is ``0`` (i.e., the empty sum).
       * If :math:`x_i` is ``NaN``, the sum is ``NaN`` (i.e., ``NaN`` values propagate).

    .. warning::

       If the accumulator is too small, overflow occurs:

       >>> x = F.ones(128, dtype="int8")
       >>> F.sum(x)
       Tensor(-128, dtype=int8, device=xpux:0)

    Examples:
        
        The sum of an empty tensor is the neutral element 0:

        >>> F.sum(Tensor([]))
        Tensor(0.0, device=xpux:0)

        Normal case:

        >>> F.sum(Tensor([1, 2, 3]))
        Tensor(6, dtype=int32, device=xpux:0)
        >>> F.sum(Tensor([0.5, 1.5]))
        Tensor(2.0, device=xpux:0)

        Along an axis:

        >>> F.sum(Tensor([[1, 2, 3], [4, 5, 6]]), axis=0)
        Tensor([5 7 9], dtype=int32, device=xpux:0)
        >>> F.sum(Tensor([[1, 2, 3], [4, 5, 6]]), axis=1)
        Tensor([ 6 15], dtype=int32, device=xpux:0)

    """
    return inp.sum(axis=axis, keepdims=keepdims)


def prod(
    inp: Tensor, axis: Optional[Union[int, Sequence[int]]] = None, keepdims=False
) -> Tensor:
    r"""Calculates the product of tensor elements over a given axis (or axes).

    Args:
        inp: input tensor. Should have a numeric data type.
        axis: axis or axes along which products must be computed.
            By default, the product must be computed over the entire tensor.
            If a sequence of integers, products must be computed over multiple axes.
        keepdims: if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions, 
            and, accordingly, the result must be compatible with the input tensor (see :ref:`broadcasting-rule`).
            Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result.

    Returns:
        if the product was computed over the entire tensor, a zero-dimensional tensor containing the products; 
        otherwise, a non-zero-dimensional tensor containing the products.
        The returned tensor must have a data type determined by :ref:`dtype-promotion`.

    .. admonition:: Special Cases

       Let ``N`` equal the number of elements over which to compute the product.

       * If ``N`` is 0, the product is ``1`` (i.e., the empty product).
       * If :math:`x_i` is ``NaN``, the product is ``NaN`` (i.e., ``NaN`` values propagate).

    .. warning::
    
       Arithmetic is modular when using integer types, and no error is raised on overflow:

       >>> x = Tensor([536870910, 536870910, 536870910, 536870910])
       >>> F.prod(x)
       Tensor(16, dtype=int32, device=xpux:0)

    Examples:

        The product of an empty tensor is the neutral element 1:

        >>> F.prod(Tensor([]))
        Tensor(1.0, device=xpux:0)

        Normal case:

        >>> F.prod(Tensor([1, 2, 3]))
        Tensor(6, dtype=int32, device=xpux:0)
        >>> F.prod(Tensor([0.5, 1.5]))
        Tensor(0.75, device=xpux:0)

        Along an axis:

        >>> F.prod(Tensor([[1, 2, 3], [4, 5, 6]]), axis=0)
        Tensor([ 4 10 18], dtype=int32, device=xpux:0)
        >>> F.prod(Tensor([[1, 2, 3], [4, 5, 6]]), axis=1)
        Tensor([  6 120], dtype=int32, device=xpux:0)

    """
    return inp.prod(axis=axis, keepdims=keepdims)


def mean(
    inp: Tensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
) -> Tensor:
    r"""Calculates the mean of tensor elements over a given axis (or axes).

    Args:
        inp: input tensor. Should have a numeric data type.
        axis: axis or axes along which means must be computed.
            By default, the mean must be computed over the entire tensor.
            If a sequence of integers, means must be computed over multiple axes.
        keepdims: if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions,
            and, accordingly, the result must be compatible with the input tensor (see :ref:`broadcasting-rule`).
            Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result.

    Returns:
        if the mean was computed over the entire tensor, a zero-dimensional tensor containing the mean;
        otherwise, a non-zero-dimensional tensor containing the means.
        The returned tensor must have a data type determined by :ref:`dtype-promotion`.

    .. admonition:: Special Cases

       Let ``N`` equal the number of elements over which to compute the mean.

       * If ``N`` is 0, the mean is ``NaN``.
       * If :math:`x_i` is ``NaN``, the mean is ``NaN`` (i.e., ``NaN`` values propagate).

    Examples:
        >>> F.mean(Tensor([1, 2, 3]))
        Tensor(2.0, device=xpux:0)

        >>> import numpy as np
        >>> F.mean(Tensor([1, np.nan, 3]))
        Tensor(nan, device=xpux:0)

        Along an axis:

        >>> F.mean(Tensor([[1, 2, 3], [4, 5, 6]]), axis=0)
        Tensor([2.5 3.5 4.5], device=xpux:0)
        >>> F.mean(Tensor([[1, 2, 3], [4, 5, 6]]), axis=1)
        Tensor([2. 5.], device=xpux:0)

    """
    return inp.mean(axis=axis, keepdims=keepdims)


def var(
    inp: Tensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
) -> Tensor:
    r"""Calculates the variance of tensor elements over a given axis (or axes).

    Args:
        inp: input tensor. Should have a numeric data type.
        axis: axis or axes along which variances must be computed.
            By default, the variance must be computed over the entire tensor.
            If a sequence of integers, variances must be computed over multiple axes.
        keepdims: if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions,
            and, accordingly, the result must be compatible with the input tensor (see :ref:`broadcasting-rule`).
            Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result.

    Returns:
        if the variance was computed over the entire tensor, a zero-dimensional tensor containing the variance;
        otherwise, a non-zero-dimensional tensor containing the variances.
        The returned tensor must have a data type determined by :ref:`dtype-promotion`.

    .. note::

       The variance is the average of the squared deviations from the mean, 
       i.e., ``var = mean(x)``, where ``x = abs(a - a.mean())**2``.

    Examples:
        >>> x = Tensor([[1, 2], [3, 4]])
        >>> F.var(x)
        Tensor(1.25, device=xpux:0)

        >>> x = Tensor([[14, 8, 11, 10], [7, 9, 10, 11], [10, 15, 5, 10]])
        >>> F.var(x)
        Tensor(6.8333335, device=xpux:0)

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
    r"""Calculates the standard deviation of tensor elements over a given axis (or axes).

    Args:
        inp: input tensor. Should have a numeric data type.
        axis: axis or axes along which standard deviations must be computed.
            By default, the standard deviation must be computed over the entire tensor.
            If a sequence of integers, standard deviations must be computed over multiple axes.
        keepdims: if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions,
            and, accordingly, the result must be compatible with the input tensor (see :ref:`broadcasting-rule`).
            Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result.

    Returns:
        if the standard deviation was computed over the entire tensor, a zero-dimensional tensor containing the standard deviation;
        otherwise, a non-zero-dimensional tensor containing the standard deviations.

    .. note::

       The standard deviation is the square root of the average of the squared deviations from the mean,
       i.e., ``std = sqrt(mean(x))``, where ``x = abs(a - a.mean())**2``.

    Examples:
        >>> x = Tensor([[1, 2], [3, 4]])
        >>> F.std(x)
        Tensor(1.118034, device=xpux:0)

        >>> x = Tensor([[14, 8, 11, 10], [7, 9, 10, 11], [10, 15, 5, 10]])
        >>> F.std(x)
        Tensor(2.6140645, device=xpux:0)
        
    """
    return var(inp, axis=axis, keepdims=keepdims) ** 0.5


def min(
    inp: Tensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
) -> Tensor:
    r"""Calculates the minimum of tensor elements over a given axis (or axes).

    Args:
        inp: input tensor. Should have a numeric data type.
        axis: axis or axes along which minimums must be computed.
            By default, the minimum must be computed over the entire tensor.
            If a sequence of integers, minimums must be computed over multiple axes.
        keepdims: if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions,
            and, accordingly, the result must be compatible with the input tensor (see :ref:`broadcasting-rule`).
            Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result.
    
    Returns:
        if the minimum was computed over the entire tensor, a zero-dimensional tensor containing the minimum;
        otherwise, a non-zero-dimensional tensor containing the minimums.

    .. admonition:: Special Cases

       If :math:`x_i` is ``NaN``, the minimum is ``NaN`` (i.e., ``NaN`` values propagate).

    Examples:

        >>> x = Tensor([[1, 2], [3, 4]])
        >>> F.min(x)
        Tensor(1, dtype=int32, device=xpux:0)

        Along an axis:

        >>> F.min(x, axis=0)
        Tensor([1 2], dtype=int32, device=xpux:0)
        >>> F.min(x, axis=1)
        Tensor([1 3], dtype=int32, device=xpux:0)

    """
    return inp.min(axis=axis, keepdims=keepdims)


def max(
    inp: Tensor,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
) -> Tensor:
    r"""Calculates the maximum of tensor elements over a given axis (or axes).

    Args:
        inp: input tensor. Should have a numeric data type.
        axis: axis or axes along which maximums must be computed.
            By default, the maximum must be computed over the entire tensor.
            If a sequence of integers, maximums must be computed over multiple axes.
        keepdims: if ``True``, the reduced axes (dimensions) must be included in the result as singleton dimensions,
            and, accordingly, the result must be compatible with the input tensor (see :ref:`broadcasting-rule`).
            Otherwise, if ``False``, the reduced axes (dimensions) must not be included in the result.

    Returns:
        if the maximum was computed over the entire tensor, a zero-dimensional tensor containing the maximum;
        otherwise, a non-zero-dimensional tensor containing the maximums.

    .. admonition:: Special Cases

       If :math:`x_i` is ``NaN``, the maximum is ``NaN`` (i.e., ``NaN`` values propagate).

    Examples:

        >>> x = Tensor([[1, 2], [3, 4]])
        >>> F.max(x)
        Tensor(4, dtype=int32, device=xpux:0)

        Along an axis:

        >>> F.max(x, axis=0)
        Tensor([3 4], dtype=int32, device=xpux:0)
        >>> F.max(x, axis=1)
        Tensor([2 4], dtype=int32, device=xpux:0)
    """
    return inp.max(axis=axis, keepdims=keepdims)


# searching functions


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


# sorting functions


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


def topk(
    inp: Tensor,
    k: int,
    descending: bool = True,
    kth_only: bool = False,
    no_sort: bool = False,
) -> Tuple[Tensor, Tensor]:
    r"""Selects the ``Top-K`` (by default) smallest elements of 2d matrix by row.

    Args:
        inp: input tensor. If input tensor is 2d, each row will be sorted.
        k: number of elements needed.
        descending: if True, return the largest elements. Default: True
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
        k = Const(k, "int32", inp.device)

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


# linear algebra functions


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
    return _matmul(inp1, inp2, transpose_a, transpose_b, compute_mode)


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
    r"""Computes the singular value decomposition of a matrix (or a stack of matrices) ``inp``.

    Let :math:`X` be the input matrix (or a stack of input matrices), the output should satisfies:

    .. math::
        X = U * diag(S) * Vh

    where ``U`` is a matrix (or stack of vectors) with orthonormal columns, ``S`` is a vector of 
    non-negative numbers (or stack of vectors), and ``Vh`` is a matrix (or a stack of matrices) 
    with orthonormal rows.

    Args:
        inp (Tensor): A input real tensor having the shape ``(..., M, N)`` with ``inp.ndim >= 2`` .
        full_matrices (bool, optional): If ``False`` , ``U`` and ``Vh`` have the shapes  ``(..., M, K)``
            and ``(..., K, N)`` , respectively, where ``K = min(M, N)`` . If ``True`` , the shapes 
            are ``(..., M, M)`` and ``(..., N, N)`` , respectively. Default: ``False`` . 
        compute_uv (bool, optional): Whether or not to compute ``U`` and ``Vh`` in addition to ``S`` . Default: ``True`` .

    Note:
        * naive does not support ``full_matrices`` and ``compute_uv`` as ``True`` .

    Returns:
        Returns a tuple ( ``U`` , ``S`` , ``Vh`` ), which are SVD factors ``U`` , ``S``, ``Vh`` of  input matrix ``inp``. 
        ( ``U`` , ``Vh`` only returned when ``compute_uv`` is True). ``U`` contains matrices orthonormal columns 
        (i.e., the columns are left singular vectors). If ``full_matrices`` is ``True`` , the array must have shape 
        ``(..., M, M)`` . If ``full_matrices`` is ``False`` , the array must have shape ``(..., M, K)`` , where ``K = min(M, N)`` .

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


def norm(
    inp: Tensor, ord: float = None, axis: int = None, keepdims=False,
):
    r"""Calculates the norm of tensor elements over a given axis.

    This function is able to return different matrix norms, 
    or one of an infinite number of vector norms (described below), depending on the value of the ord parameter.

    Args:
        inp: input tensor. Should have a numeric data type.
        ord: Order of the norm (see table under Notes). If not specified, the default is 2.
        axis: Axis along which to compute vector norms.
            If axis is an integer, it specifies the axis of inp along which to compute the vector norms.
        keepdims: If this is set to ``True``, 
            the axes which are normed over are left in the result as dimensions with size one.

    Returns:
        Norm of the matrix or vector(s).

    .. note:: 

        Now the following norms can be calculated:

        * inf: norm-:math:`\infty` (maximum of absolute values).
        * -inf: norm-:math:`-\infty` (minimum of absolute values).
        * 2: 2-norm (largest singluar value).

        The Frobenius norm is given by to ``sum(abs(x)**ord)**(1./ord)``:

        .. math::

           \|A\|_F=\left[\sum_{i, j} a b s\left(a_{i, j}\right)^2\right]^{1 / 2}

    .. seealso:: :func:`numpy.linalg.norm` / :func:`~.functional.normalize`

    Examples:

        >>> import math
        >>> x = Tensor([1, 2, 3])
        >>> F.norm(x, ord=math.inf)
        Tensor(3, dtype=int32, device=xpux:0)
        >>> F.norm(x, ord=-math.inf)
        Tensor(1, dtype=int32, device=xpux:0)

        >>> x = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> F.norm(x, ord=2, axis=0)
        Tensor([4.1231 5.3852 6.7082], device=xpux:0)
        >>> F.norm(x, ord=2, axis=1)
        Tensor([3.7417 8.775 ], device=xpux:0)

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


def normalize(
    inp: Tensor, ord: float = None, axis: int = None, eps: float = 1e-12,
) -> Tensor:
    r"""Performs :math:`L_p` normalization of input tensor along given axis.

    For a tensor of shape :math:`(n_0, ..., n_{dim}, ..., n_k)`, 
    each :math:`n_{dim}` -element vector :math:`v` along dimension :attr:`axis` is transformed as:

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}.

    Args:
        inp: input tensor.
        ord: power of value applied to input tensor.
        axis: dimension to reduce.If None, input must be a vector.
        eps: a small value to avoid division by zero.

    Returns:
        normalized output tensor.

    seealso:: :func:`numpy.linalg.norm` / :func:`~.functional.norm`

    Examples:

        >>> x = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> F.normalize(x, ord=2, axis=0)
        Tensor([[0.2425 0.3714 0.4472]
         [0.9701 0.9285 0.8944]], device=xpux:0)
        >>> F.normalize(x, ord=2, axis=1)
        Tensor([[0.2673 0.5345 0.8018]
         [0.4558 0.5698 0.6838]], device=xpux:0)
    """
    if axis is None:
        return inp / clip(norm(inp, ord, axis), lower=eps)
    else:
        return inp / clip(norm(inp, ord, axis, keepdims=True), lower=eps)


def _check_non_finite(inps: Iterable[Tensor], scale=1.0) -> Tensor:
    r"""Check whether input contains infinite or nan value.

    Args:
        inps: tensors to be checked.

    Returns:
        a int32 scalar tensor, 0 for False and 1 for True.
    """
    if isinstance(inps, Tensor):
        inps = [inps]
    op = builtin.CheckNonFinite(scale=scale)
    oups = apply(op, *inps)
    out = oups[-1]
    for i in range(len(inps)):
        inps[i]._reset(oups[i])

    return out
