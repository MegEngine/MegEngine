# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# pylint: disable=unused-argument,invalid-name,redefined-builtin,arguments-out-of-order
import functools

import megengine._internal as mgb

from ..core.tensor import Tensor, wrap_io_tensor

__all__ = [
    "abs",
    "arccos",
    "add",
    "arcsin",
    "ceil",
    "cos",
    "divide",
    "equal",
    "exp",
    "greater",
    "greater_equal",
    "floor",
    "less",
    "less_equal",
    "log",
    "maximum",
    "minimum",
    "mod",
    "multiply",
    "power",
    "relu",
    "round",
    "sigmoid",
    "sin",
    "subtract",
    "tanh",
]


def _elemwise(mode):  # DONT export
    """Decorator helps to wrap megbrain element-wise oprs"""

    def elemwise_decorator(func):
        @functools.wraps(func)
        @wrap_io_tensor
        def elemwise_func(*inputs) -> Tensor:
            return mgb.opr.elemwise(*inputs, mode=mode)

        return elemwise_func

    return elemwise_decorator


@_elemwise("ABS")
def abs(x):
    """Calculate the absolute value element-wise."""


@_elemwise("ACOS")
def arccos(x):
    """Inverse cosine, element-wise."""


@_elemwise("ADD")
def add(x, y):
    """Element-wise addition."""


@_elemwise("ASIN")
def arcsin(x):
    """Inverse sine, element-wise."""


@_elemwise("CEIL")
def ceil(x):
    """Return the ceil of the input, element-wise."""


@_elemwise("COS")
def cos(x):
    """Cosine, element-wise."""


@_elemwise("TRUE_DIV")
def divide(x, y):
    """Return (x / y) element-wise."""


@_elemwise("EQ")
def equal(x, y):
    """Return (x == y) element-wise."""


@_elemwise("EXP")
def exp(x):
    """Calculate the exponential element-wise"""


@_elemwise("FLOOR")
def floor(x):
    """Return the floor of the input, element-wise"""


def greater(x, y):
    """Return (x > y) element-wise."""
    return less(y, x)


def greater_equal(x, y):
    """Return (x >= y) element-wise"""
    return less_equal(y, x)


@_elemwise("LT")
def less(x, y):
    """Return (x < y) element-wise."""


@_elemwise("LEQ")
def less_equal(x, y):
    """Return (x =< y) element-wise."""


@_elemwise("LOG")
def log(x):
    """Natural logarithm (base `e`), element-wise."""


@_elemwise("MAX")
def maximum(x, y):
    """Element-wise maximum of array elements."""


@_elemwise("MIN")
def minimum(x, y):
    """Element-wise minimum of array elements."""


@_elemwise("MOD")
def mod(x, y):
    """Return element-wise remainder of division."""


@_elemwise("MUL")
def multiply(x, y):
    """Element-wise multiplication."""


@_elemwise("POW")
def power(x, y):
    """First tensor elements raised to powers from second tensor (x ** y), element-wise."""


@_elemwise("RELU")
def relu(x):
    """Return `max(x, 0)` element-wise."""


@_elemwise("ROUND")
def round(x):
    """Round tensor to int element-wise."""


@_elemwise("SIGMOID")
def sigmoid(x):
    """Return 1 / ( 1 + exp( -x ) ) element-wise."""


@_elemwise("SIN")
def sin(x):
    """Sine, element-wise."""


@_elemwise("SUB")
def subtract(x, y):
    """Subtract arguments element-wise"""


@_elemwise("TANH")
def tanh(x):
    """Compute hyperbolic tangent element-wise."""


@wrap_io_tensor
def clamp(inp: Tensor, lower=None, upper=None) -> Tensor:
    r"""
    Clamp all elements in :attr:`inp` into the range `[` :attr:`lower`, :attr:`upper` `]` and return
    a resulting tensor:

    .. math::
        y_i = \begin{cases}
            \text{lower} & \text{if } x_i < \text{lower} \\
            x_i & \text{if } \text{lower} \leq x_i \leq \text{upper} \\
            \text{upper} & \text{if } x_i > \text{upper}
        \end{cases}

    :param inp: the input tensor.
    :param lower: lower-bound of the range to be clamped to
    :param upper: upper-bound of the range to be clamped to

    Example:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F
        a = tensor(np.arange(5).astype(np.int32))

        print(F.clamp(a, 2, 4).numpy())

        print(F.clamp(a, lower=3).numpy())

        print(F.clamp(a, upper=3).numpy())

    .. testoutput::

        [2 2 2 3 4]
        [3 3 3 3 4]
        [0 1 2 3 3]

    """
    assert lower or upper, "At least one of 'lower' or 'upper' must not be None"
    if lower:
        if upper:
            assert lower <= upper, "clamp lower bound is bigger that upper bound"
            return minimum(maximum(inp, lower), upper)
        else:
            return maximum(inp, lower)
    else:
        return minimum(inp, upper)
