# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Optional

import megengine._internal as mgb

from ..core import Tensor, wrap_io_tensor
from .elemwise import clamp


@wrap_io_tensor
def sum(inp: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    r"""Returns the sum of each row of the ``inp`` tensor in the given ``axis``.

    :param inp: The input tensor.
    :param axis: The dimension to reduce. If None, all the dimensions will be reduced.
        Default: None
    :param keepdims: Whether the output tensor has ``axis`` retained or not.
        Default: False
    :return: The output tensor

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        data = tensor(np.arange(1, 7, dtype=np.int32).reshape(2, 3))
        out = F.sum(data)
        print(out.numpy())

    .. testoutput::

        [21]

    """
    return mgb.opr.reduce_(inp, "SUM", axis, keepdims)


@wrap_io_tensor
def prod(inp: Tensor, axis: Optional[int] = None, keepdims=False) -> Tensor:
    r"""
    Returns the element product of input tensor along given *axis*.

    :param inp: The input tensor
    :param axis: The dimension to reduce. If None, all the dimensions will be reduced. Default: ``None``
    :param keepdims: Whether the output tensor has *axis* retained or not. Default: ``False``
    :return: The output tensor

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        data = tensor(np.arange(1, 7, dtype=np.int32).reshape(2, 3))
        out = F.prod(data)
        print(out.numpy())

    Outputs:

    .. testoutput::

        [720]

    """
    return mgb.opr.reduce_(inp, "PRODUCT", axis, keepdims)


@wrap_io_tensor
def mean(inp: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """Returns the mean value of each row of the ``inp`` tensor in
    the given ``axis``. If axis is a list of dimensions,
    reduce over all of them.

    :param inp: The input tensor
    :param axis: The dimension to reduce. If None, all the dimensions will be reduced. Default: None
    :param keepdims: Whether the output tensor has ``axis`` retained or not. Default: False

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        data = tensor(np.arange(1, 7, dtype=np.int32).reshape(2, 3))
        out = F.mean(data)
        print(out.numpy())

    .. testoutput::

        [3.5]

    """
    return mgb.opr.mean(inp, axis, keepdims)


@wrap_io_tensor
def min(inp: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    r"""
    Returns the min value of input tensor along given *axis*.

    :param inp: The input tensor
    :param axis: The dimension to reduce. If None, all the dimensions will be reduced. Default: None
    :param keepdims: Whether the output tensor has *axis* retained or not. Default: False
    :return: The output tensor

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = tensor(np.arange(1, 7, dtype=np.int32).reshape(2,3))
        y = F.min(x)
        print(y.numpy())

    Outputs:

    .. testoutput::

        [1]

    """
    return mgb.opr.reduce_(inp, "MIN", axis, keepdims)


@wrap_io_tensor
def max(inp: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    r"""Returns the max value of the input tensor along given *axis*.

    :param inp: The input tensor
    :param axis: The dimension to reduce. If None, all the dimensions will be reduced. Default: None
    :param keepdims: Whether the output tensor has *axis* retained or not. Default: False
    :return: The output tensor

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = tensor(np.arange(1, 7, dtype=np.int32).reshape(2,3))
        y = F.max(x)
        print(y.numpy())

    .. testoutput::

        [6]

    """
    return mgb.opr.reduce_(inp, "MAX", axis, keepdims)


@wrap_io_tensor
def sqrt(inp: Tensor) -> Tensor:
    """
    Return a new tensor with the square-root of the elements of ``inp``

    :param inp: The input tensor
    :return: The computed tensor

    Examples:

    .. testcode::

        import numpy as np
        import megengine as mge
        import megengine.functional as F

        data = mge.tensor(np.arange(0, 6, dtype=np.float32).reshape(2, 3))
        out = F.sqrt(data)
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[0.      1.     1.4142]
         [1.7321  2.     2.2361 ]]

    """

    return mgb.opr.sqrt(inp)


def norm(inp: Tensor, p: int = 2, axis: Optional[int] = None, keepdims=False):
    """Calculate ``p``-norm of input tensor along certain axis.

    :param inp: The input tensor
    :param p: power of value ``p`` applied to ``inp``. Default: 2
    :param axis: The dimension to reduce. If None, all the dimensions will be reduced. Default: None
    :param keepdims: Whether the output tensor has ``axis`` retained or not. Default: False
    :return: The output tensor

    """
    if axis is None:
        inp = inp.reshape(-1)
    return (inp ** p).sum(axis=axis, keepdims=keepdims) ** (1.0 / p)


@wrap_io_tensor
def argmin(inp: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    r"""Returns the indices of the minimum values along an axis

    :param inp: The input tensor
    :param axis: The dimension to reduce. If None, all the dimensions will be reduced. Default: None
    :param keepdims: Whether the output tensor has *axis* retained or not. Default: False
    :return: The output tensor

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = tensor(np.arange(1, 7, dtype=np.int32).reshape(2,3))
        y = F.argmin(x)
        print(y.numpy())

    .. testoutput::

        [0]

    """
    return mgb.opr.argmin(inp, axis, keepdims)


@wrap_io_tensor
def argmax(inp: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    r"""Returns the indices of the maximum values along an axis

    :param inp: The input tensor
    :param axis: The dimension to reduce. If None, all the dimensions will be reduced. Default: None
    :param keepdims: Whether the output tensor has *axis* retained or not. Default: False
    :return: The output tensor

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = tensor(np.arange(1, 7, dtype=np.int32).reshape(2,3))
        y = F.argmax(x)
        print(y.numpy())

    .. testoutput::

        [5]

    """
    return mgb.opr.argmax(inp, axis, keepdims)


def normalize(
    inp: Tensor, p: int = 2, axis: Optional[int] = None, eps: float = 1e-12
) -> Tensor:
    r"""Perform :math:`L_p` normalization of input tensor along certain axis.

    For a tensor :attr:`inp` of shape :math:`(n_0, ..., n_{dim}, ..., n_k)`, each
    :math:`n_{dim}` -element vector :math:`v` along dimension :attr:`axis` is transformed as:

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}.

    :param inp: the input tensor
    :param p: power of value ``p`` applied to ``inp``. Default: 2
    :param axis: The dimension to reduce. If None, all the dimensions will be reduced
        to calculate the norm. Default: None
    :param eps: a small value to avoid division by zero. Default: 1e-12
    :return: the normalized output tensor

    """
    if axis is None:
        return inp / clamp(norm(inp, p), lower=eps)
    else:
        return inp / clamp(norm(inp, p, axis, keepdims=True), lower=eps)
