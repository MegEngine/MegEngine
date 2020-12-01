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

from ..core._imperative_rt.core2 import apply
from ..core.ops import builtin
from ..core.ops.builtin import Elemwise
from ..core.tensor import megbrain_graph, utils
from ..core.tensor.utils import isscalar, setscalar
from ..device import get_default_device
from ..jit.tracing import is_tracing
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
    "ceil",
    "clip",
    "cos",
    "cosh",
    "div",
    "equal",
    "exp",
    "expm1",
    "floor",
    "floor_div",
    "greater",
    "greater_equal",
    "hswish",
    "hsigmoid",
    "left_shift",
    "less",
    "less_equal",
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
    "not_equal",
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
]


def _elwise(*args, mode):
    op = builtin.Elemwise(mode)
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
    _isscalar = True
    for i in args:
        if isscalar(i) == False:
            _isscalar = False
            break
    (result,) = apply(op, *args)
    if _isscalar:
        setscalar(result)
    return result


def _elemwise_multi_type(*args, mode, **kwargs):
    op = builtin.ElemwiseMultiType(mode=mode, **kwargs)
    args = utils.convert_inputs(*args)
    (result,) = apply(op, *args)
    return result


# math operations


def add(x, y):
    """
    Element-wise `addition`.
    At least one operand should be tensor.

    Same for sub/mul/div/floor_div/pow/mod/atan2/equal/not_equal/less/less_equal/greater/greater_equal/maximum/minmium.

    :param x: input tensor.
    :return: computed tensor.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = tensor(np.arange(0, 6, dtype=np.float32).reshape(2, 3))
        y = tensor(np.arange(0, 6, dtype=np.float32).reshape(2, 3))
        out = F.add(x, y)
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[ 0.  2.  4.]
         [ 6.  8. 10.]]

    """
    return _elwise(x, y, mode=Elemwise.Mode.ADD)


def sub(x, y):
    """Element-wise `subtraction`."""
    return _elwise(x, y, mode=Elemwise.Mode.SUB)


def mul(x, y):
    """Element-wise `multiplication`."""
    return _elwise(x, y, mode=Elemwise.Mode.MUL)


def div(x, y):
    """Element-wise `(x / y)`."""
    return _elwise(x, y, mode=Elemwise.Mode.TRUE_DIV)


def floor_div(x, y):
    """Element-wise `floor(x / y)`."""
    return _elwise(x, y, mode=Elemwise.Mode.FLOOR_DIVIDE)


def neg(x):
    """Element-wise `negation`."""
    return _elwise(x, mode=Elemwise.Mode.NEGATE)


def pow(x, y):
    """Element-wise `power`."""
    return _elwise(x, y, mode=Elemwise.Mode.POW)


def mod(x, y):
    """Element-wise `remainder of division`."""
    return _elwise(x, y, mode=Elemwise.Mode.MOD)


def abs(x):
    """Element-wise `absolute value`."""
    return _elwise(x, mode=Elemwise.Mode.ABS)


def exp(x):
    """Element-wise `exponential`."""
    return _elwise(x, mode=Elemwise.Mode.EXP)


def expm1(x):
    """Element-wise `exp(x)-1`."""
    return _elwise(x, mode=Elemwise.Mode.EXPM1)


def log(x):
    """Element-wise `logarithm (base e)`."""
    return _elwise(x, mode=Elemwise.Mode.LOG)


def log1p(x):
    """Element-wise `log(x+1) (base e)`."""
    return _elwise(x, mode=Elemwise.Mode.LOG1P)


def sqrt(x: Tensor) -> Tensor:
    """
    Element-wise `sqrt`.
    Returns ``NaN`` for negative input value.

    :param x: input tensor.
    :return: computed tensor.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = tensor(np.arange(0, 6, dtype=np.float32).reshape(2, 3))
        out = F.sqrt(x)
        print(out.numpy().round(decimals=4))

    Outputs:

    .. testoutput::

        [[0.     1.     1.4142]
         [1.7321 2.     2.2361]]

    """
    return x ** 0.5


def square(x: Tensor) -> Tensor:
    """
    Returns a new tensor with the square of the elements of input tensor.

    :param inp: input tensor.
    :return: computed tensor.

    Examples:

    .. testcode::

        import numpy as np
        import megengine as mge
        import megengine.functional as F

        data = mge.tensor(np.arange(0, 6, dtype=np.float32).reshape(2, 3))
        out = F.square(data)
        print(out.numpy().round(decimals=4))

    Outputs:

    .. testoutput::

        [[ 0.  1.  4.]
         [ 9. 16. 25.]]

    """
    return x ** 2


def round(x):
    """Element-wise `rounding to int`."""
    return _elwise(x, mode=Elemwise.Mode.ROUND)


def ceil(x):
    """Element-wise `ceiling`."""
    return _elwise(x, mode=Elemwise.Mode.CEIL)


def floor(x):
    """Element-wise `floor`."""
    return _elwise(x, mode=Elemwise.Mode.FLOOR)


def maximum(x, y):
    """Element-wise `maximum of array elements`."""
    return _elwise(x, y, mode=Elemwise.Mode.MAX)


def minimum(x, y):
    """Element-wise `minimum of array elements`."""
    return _elwise(x, y, mode=Elemwise.Mode.MIN)


# trigonometric functions


def cos(x):
    """
    Element-wise `cosine`.

    :param x: input tensor.
    :return: computed tensor.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = tensor(np.arange(0, 6, dtype=np.float32).reshape(2, 3))
        out = F.cos(x)
        print(out.numpy().round(decimals=4))

    Outputs:

    .. testoutput::

        [[ 1.      0.5403 -0.4161]
         [-0.99   -0.6536  0.2837]]

    """
    return _elwise(x, mode=Elemwise.Mode.COS)


def sin(x):
    """Element-wise `sine`."""
    return _elwise(x, mode=Elemwise.Mode.SIN)


def tan(x):
    """Element-wise `tangent`."""
    return sin(x) / cos(x)


def acos(x):
    """Element-wise `inverse cosine`."""
    return _elwise(x, mode=Elemwise.Mode.ACOS)


def asin(x):
    """Element-wise `inverse sine`."""
    return _elwise(x, mode=Elemwise.Mode.ASIN)


def atan(x):
    """Element-wise `inverse tangent`."""
    return _elwise(x, 1, mode=Elemwise.Mode.ATAN2)


def atan2(y, x):
    """Element-wise `2-argument arctangent`."""
    return _elwise(y, x, mode=Elemwise.Mode.ATAN2)


def cosh(x):
    r"""Element-wise `hyperbolic cosine`."""
    return 0.5 * (exp(x) + exp(-x))


def sinh(x):
    r"""Element-wise `hyperbolic sine`."""
    u = expm1(x)
    return 0.5 * u / (u + 1) * (u + 2)


def tanh(x):
    r"""Element-wise `hyperbolic tangent`."""
    return _elwise(x, mode=Elemwise.Mode.TANH)


def asinh(x):
    r"""Element-wise `inverse hyperbolic sine`."""
    return log(x + (x ** 2 + 1) ** 0.5)


def acosh(x):
    r"""Element-wise `inverse hyperbolic cosine`."""
    return log(x + (x ** 2 - 1) ** 0.5)


def atanh(x):
    r"""Element-wise `inverse hyperbolic tangent`."""
    return log1p(2 * x / (1 - x)) / 2


# bit-twiddling functions


def left_shift(x, y):
    """
    Element-wise `bitwise binary: x << y`.

    :param x: input tensor, should be int.
    :param y: how many bits to be left-shifted.
    :return: computed tensor.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = tensor(np.arange(0, 6, dtype=np.int32).reshape(2, 3))
        out = F.left_shift(x, 2)
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[ 0  4  8]
         [12 16 20]]

    """
    return _elwise(x, y, mode=Elemwise.Mode.SHL)


def right_shift(x, y):
    """Element-wise `bitwise binary: x >> y`."""
    return _elwise(x, y, mode=Elemwise.Mode.SHR)


# logical functions


def logical_and(x, y):
    """Element-wise `logical and: x && y`."""
    return _elwise(x, y, mode=Elemwise.Mode.AND)


def logical_not(x):
    """Element-wise `logical not: ~x`."""
    return _elwise(x, mode=Elemwise.Mode.NOT)


def logical_or(x, y):
    """Element-wise `logical or: x || y`."""
    return _elwise(x, y, mode=Elemwise.Mode.OR)


def logical_xor(x, y):
    """Element-wise `logical xor: x ^ y`."""
    return _elwise(x, y, mode=Elemwise.Mode.XOR)


# comparison functions


def equal(x, y):
    """
    Element-wise `(x == y)`.

    :param x: input tensor 1.
    :param y: input tensor 2.
    :return: computed tensor.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = tensor(np.arange(0, 6, dtype=np.float32).reshape(2, 3))
        y = tensor(np.arange(0, 6, dtype=np.float32).reshape(2, 3))
        out = F.equal(x, y)
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[1. 1. 1.]
         [1. 1. 1.]]

    """
    return _elwise(x, y, mode=Elemwise.Mode.EQ)


def not_equal(x, y):
    """Element-wise `(x != y)`."""
    return x != y


def less(x, y):
    """Element-wise `(x < y)`."""
    return _elwise(x, y, mode=Elemwise.Mode.LT)


def less_equal(x, y):
    """Element-wise `(x <= y)`."""
    return _elwise(x, y, mode=Elemwise.Mode.LEQ)


def greater(x, y):
    """Element-wise `(x > y)`."""
    return _elwise(y, x, mode=Elemwise.Mode.LT)


def greater_equal(x, y):
    """Element-wise `(x >= y)`."""
    return _elwise(y, x, mode=Elemwise.Mode.LEQ)


# other functions


def hswish(x):
    """
    Element-wise `x * relu6(x + 3) / 6`.

    :param x: input tensor.
    :return: computed tensor.

    Example:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = tensor(np.arange(5).astype(np.float32))
        out = F.hswish(x)
        print(out.numpy().round(decimals=4))

    .. testoutput::

        [0.     0.6667 1.6667 3.     4.    ]

    """
    return _elwise(x, mode=Elemwise.Mode.H_SWISH)


def hsigmoid(x):
    """Element-wise `relu6(x + 3) / 6`."""
    return relu6(x + 3) / 6


def relu(x):
    """Element-wise `max(x, 0)`."""
    return _elwise(x, mode=Elemwise.Mode.RELU)


def relu6(x):
    """Element-wise `min(max(x, 0), 6)`."""
    return minimum(maximum(x, 0), 6)


def sigmoid(x):
    """Element-wise `1 / ( 1 + exp( -x ) )`."""
    return _elwise(x, mode=Elemwise.Mode.SIGMOID)


def clip(x: Tensor, lower=None, upper=None) -> Tensor:
    r"""
    Clamps all elements in input tensor into the range `[` :attr:`lower`, :attr:`upper` `]` and returns
    a resulting tensor:

    .. math::
        y_i = \begin{cases}
            \text{lower} & \text{if } x_i < \text{lower} \\
            x_i & \text{if } \text{lower} \leq x_i \leq \text{upper} \\
            \text{upper} & \text{if } x_i > \text{upper}
        \end{cases}

    :param x: input tensor.
    :param lower: lower-bound of the range to be clamped to.
    :param upper: upper-bound of the range to be clamped to.
    :return: output clamped tensor.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        a = tensor(np.arange(5).astype(np.int32))
        print(F.clip(a, 2, 4).numpy())
        print(F.clip(a, lower=3).numpy())
        print(F.clip(a, upper=3).numpy())

    Outputs:

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
            if not is_tracing():
                assert lower <= upper, "clip lower bound is bigger that upper bound"
            return minimum(maximum(x, lower), upper)
        else:
            return maximum(x, lower)
    else:
        return minimum(x, upper)
