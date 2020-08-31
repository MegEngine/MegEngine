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

from ..core.ops import builtin
from ..core.tensor import megbrain_graph, utils
from ..core.tensor.core import apply
from ..device import get_default_device
from ..tensor import Tensor

__all__ = [
    "abs",
    "add",
    "acos",
    "asin",
    "atan",
    "atan2",
    "asinh",
    "acosh",
    "atanh",
    "bitwise_and",  # TODO
    "bitwise_not",  # TODO
    "bitwise_or",  # TODO
    "bitwise_xor",  # TODO
    "ceil",
    "clamp",
    "cos",
    "cosh",
    "div",
    "eq",
    "exp",
    "expm1",
    "floor",
    "floor_div",
    "gt",
    "ge",
    "hswish",
    "hsigmoid",
    "left_shift",
    "lt",
    "le",
    "log",
    "log1p",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
    "mod",
    "mul",
    "neg",
    "ne",
    "pow",
    "relu",
    "relu6",
    "right_shift",
    "round",
    "sigmoid",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "sub",
    "tan",
    "tanh",
    "fast_tanh",
]


def _elwise(*args, mode):
    op = builtin.Elemwise(mode=mode)
    tensor_args = list(
        filter(lambda x: isinstance(x, (Tensor, megbrain_graph.VarNode)), args)
    )
    if len(tensor_args) == 0:
        dtype = utils.dtype_promotion(args)
        first_arg = Tensor(args[0], dtype=dtype, device=get_default_device())
        args = utils.convert_inputs(first_arg, *args[1:])
    else:
        args = utils.convert_inputs(*args)
    if mode in ("true_div", "exp", "pow", "log", "expm1", "log1p"):
        args = tuple(map(lambda x: x.astype("float32"), args))
    (result,) = apply(op, *args)
    return result


def _logical(*args, mode):
    op = builtin.CondExecPredLogical(mode=mode)
    args = utils.convert_inputs(*args)
    (result,) = apply(op, *args)
    return result


def _elemwise_multi_type(*args, mode, **kwargs):
    op = builtin.ElemwiseMultiType(mode=mode, **kwargs)
    args = utils.convert_inputs(*args)
    (result,) = apply(op, *args)
    return result


# math operations


def add(x, y):
    """Element-wise addition.
    At least one operand should be tensor.
    same for sub/mul/div/floor_div/pow/mod/atan2/eq/ne/lt/le/gt/ge/maximum/minmium.
    """
    return _elwise(x, y, mode="add")


def sub(x, y):
    """Element-wise subtract."""
    return _elwise(x, y, mode="sub")


def mul(x, y):
    """Element-wise multiplication."""
    return _elwise(x, y, mode="mul")


def div(x, y):
    """Element-wise (x / y)."""
    return _elwise(x, y, mode="true_div")


def floor_div(x, y):
    """Element-wise floor(x / y)."""
    return _elwise(x, y, mode="floor_divide")


def neg(x):
    """Element-wise negation."""
    return _elwise(x, mode="negate")


def pow(x, y):
    """Element-wise power."""
    return _elwise(x, y, mode="pow")


def mod(x, y):
    """Element-wise remainder of division."""
    return _elwise(x, y, mode="mod")


def abs(x):
    """Element-wise absolute value."""
    return _elwise(x, mode="abs")


def exp(x):
    """Element-wise exponential."""
    return _elwise(x, mode="exp")


def expm1(x):
    """Element-wise exp(x)-1."""
    return _elwise(x, mode="expm1")


def log(x):
    """Element-wise logarithm (base `e`)."""
    return _elwise(x, mode="log")


def log1p(x):
    """Element-wise log(x+1) (base `e`)."""
    return _elwise(x, mode="log1p")


def sqrt(inp: Tensor) -> Tensor:
    """
    Return a new tensor with the square-root of the elements of ``inp``.
    For negative value, return nan.

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
    return inp ** 0.5


def square(inp: Tensor) -> Tensor:
    """
    Return a new tensor with the square of the elements of ``inp``

    :param inp: The input tensor
    :return: The computed tensor

    Examples:

    .. testcode::

        import numpy as np
        import megengine as mge
        import megengine.functional as F

        data = mge.tensor(np.arange(0, 6, dtype=np.float32).reshape(2, 3))
        out = F.square(data)
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[0.      1.     4.]
         [9.      16.    25.]]

    """
    return inp ** 2


def round(x):
    """Round tensor to int element-wise."""
    return _elwise(x, mode="round")


def ceil(x):
    """Return the ceil of the input, element-wise."""
    return _elwise(x, mode="ceil")


def floor(x):
    """Calculate the floor element-wise"""
    return _elwise(x, mode="floor")


# trigonometric functions


def cos(x):
    """Cosine, element-wise."""
    return _elwise(x, mode="cos")


def sin(x):
    """Sine, element-wise."""
    return _elwise(x, mode="sin")


def tan(x):
    return sin(x) / cos(x)


def acos(x):
    """Inverse cosine, element-wise."""
    return _elwise(x, mode="acos")


def asin(x):
    """Inverse sine, element-wise."""
    return _elwise(x, mode="asin")


def atan(x):
    return _elwise(x, 1, mode="atan2")


def atan2(y, x):
    return _elwise(y, x, mode="atan2")


def cosh(x):
    r"""Compute element-wise hyperbolic cosine."""
    return 0.5 * (exp(x) + exp(-x))


def sinh(x):
    r"""Compute element-wise hyperbolic sine."""
    u = expm1(x)
    return 0.5 * u / (u + 1) * (u + 2)


def tanh(x):
    r"""Compute element-wise hyperbolic tangent."""
    return _elwise(x, mode="tanh")


def asinh(x):
    r"""Compute element-wise inverse hyperbolic sine."""
    return log(x + (x ** 2 + 1) ** 0.5)


def acosh(x):
    r"""Compute element-wise inverse hyperbolic cosine."""
    return log(x + (x ** 2 - 1) ** 0.5)


def atanh(x):
    r"""Compute element-wise inverse hyperbolic tangent."""
    return log1p(2 * x / (1 - x)) / 2


def fast_tanh(x):
    r"""Compute element-wise fast tanh; this is an approximation:

    .. math::
        \text{fast_tanh}(x) = x * (27. + x * x) / (27. + 9. * x * x)
    """
    return _elwise(x, mode="fast_tanh")


# bit-twiddling functions


def left_shift(x, y):
    return _elwise(x, y, mode="shl")


def right_shift(x, y):
    return _elwise(x, y, mode="shl")


def bitwise_and(x, y):
    raise NotImplementedError


def bitwise_not(x):
    raise NotImplementedError


def bitwise_or(x, y):
    raise NotImplementedError


def bitwise_xor(x, y):
    raise NotImplementedError


# logical functions


def logical_and(x, y):
    return _elwise(x, y, mode="AND")


def logical_not(x):
    return _elwise(x, mode="NOT")


def logical_or(x, y):
    return _elwise(x, y, mode="OR")


def logical_xor(x, y):
    return _elwise(x, y, mode="XOR")


# comparison functions


def eq(x, y):
    """Return (x == y) element-wise."""
    return _elwise(x, y, mode="eq")


def ne(x, y):
    return x != y


def lt(x, y):
    """Return (x < y) element-wise."""
    return _elwise(x, y, mode="lt")


def le(x, y):
    """Return (x =< y) element-wise."""
    return _elwise(x, y, mode="leq")


def gt(x, y):
    """Return (x > y) element-wise."""
    return _elwise(y, x, mode="lt")


def ge(x, y):
    """Return (x >= y) element-wise"""
    return _elwise(y, x, mode="leq")


def hswish(x):
    """Return x * relu6(x + 3) / 6 element-wise"""
    return _elwise(x, mode="h_swish")


def hsigmoid(x):
    """Return relu6(x + 3) / 6 element-wise"""
    return relu6(x + 3) / 6


def relu(x):
    """Return `max(x, 0)` element-wise."""
    return _elwise(x, mode="relu")


def relu6(x):
    """Return min(max(x, 0), 6) element-wise."""
    return minimum(maximum(x, 0), 6)


def sigmoid(x):
    """Return 1 / ( 1 + exp( -x ) ) element-wise."""
    return _elwise(x, mode="sigmoid")


def maximum(x, y):
    """Element-wise maximum of array elements."""
    return _elwise(x, y, mode="max")


def minimum(x, y):
    """Element-wise minimum of array elements."""
    return _elwise(x, y, mode="min")


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
    assert (
        lower is not None or upper is not None
    ), "At least one of 'lower' or 'upper' must not be None"
    if lower is not None:
        if upper is not None:
            assert lower <= upper, "clamp lower bound is bigger that upper bound"
            return minimum(maximum(inp, lower), upper)
        else:
            return maximum(inp, lower)
    else:
        return minimum(inp, upper)
