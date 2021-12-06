# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# pylint: disable=unused-argument,invalid-name,redefined-builtin,arguments-out-of-order
import numpy as np

from ..core._imperative_rt.core2 import SymbolVar, apply
from ..core.ops import builtin
from ..core.ops.builtin import Elemwise
from ..core.tensor.array_method import _elwise
from ..core.tensor.utils import convert_inputs
from ..tensor import Tensor
from ..utils.deprecation import deprecated_func

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
    "left_shift",
    "less",
    "less_equal",
    "log",
    "log1p",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "logaddexp",
    "maximum",
    "minimum",
    "mod",
    "mul",
    "neg",
    "not_equal",
    "pow",
    "right_shift",
    "round",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "sub",
    "tan",
    "tanh",
]


def _elemwise_multi_type(*args, mode, **kwargs):
    op = builtin.ElemwiseMultiType(mode=mode, **kwargs)
    args = convert_inputs(*args)
    (result,) = apply(op, *args)
    return result


# math operations


def add(x, y):
    r"""Element-wise `addition`.

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
    r"""Element-wise `sub`.

    Examples:

        .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            x = tensor(np.arange(1, 7, dtype=np.float32).reshape(2, 3))
            y = tensor(np.arange(0, 6, dtype=np.float32).reshape(2, 3))
            out = F.sub(x, y)
            print(out.numpy())

        Outputs:

        .. testoutput::

            [[1. 1. 1.]
             [1. 1. 1.]]

    """
    return _elwise(x, y, mode=Elemwise.Mode.SUB)


def mul(x: Tensor, y: Tensor) -> Tensor:
    r"""Calculates the product for each element :math:`x_i` of the input tensor `x` with the respective element :math:`y_i` of the input tensor `y`.

    Note:
        * If either :math:`x_i` or :math:`y_i` is `NaN`, the result is `NaN`.
        * If :math:`x_i` is either `+infinity` or `-infinity` and :math:`y_i` is either `+0` or `-0`, the result is `NaN`.
        * If :math:`x_i` is either `+0` or `-0` and :math:`y_i` is either `+infinity` or `-infinity`, the result is `NaN`.
        * If :math:`x_i` and :math:`y_i` have different mathematical signs, the result has a negative mathematical sign, unless the result is `NaN`.
        * If :math:`x_i` is either `+infinity` or `-infinity` and :math:`y_i` is either `+infinity` or `-infinity`,
          the result is a signed infinity with the mathematical sign determined by the rule already stated above.
        * If :math:`x_i` is either `+infinity` or `-infinity` and :math:`y_i` is a nonzero finite number, 
          the result is a signed infinity with the mathematical sign determined by the rule already stated above.
        * If :math:`x_i` is a nonzero finite number and :math:`y_i` is either `+infinity` or `-infinity`, 
          the result is a signed infinity with the mathematical sign determined by the rule already stated above.
        * In the remaining cases, where neither `infinity` nor `NaN` is involved,
          the product must be computed and rounded to the nearest representable value according to IEEE 754-2019 and a supported rounding mode.
          If the magnitude is too large to represent, the result is an `infinity` of appropriate mathematical sign.
          If the magnitude is too small to represent, the result is a zero of appropriate mathematical sign.
        * Floating-point multiplication is not always associative due to finite precision.

    Args:
        x: first input tensor. Should have a numeric data type.
        y: second input tensor. Must be compatible with `x` (see :ref:`broadcasting-rule` ). Should have a numeric data type.

    Returns:
        A tensor containing the element-wise products. The returned array must have a data type determined by :ref:`dtype-promotion`.

    Examples:

        >>> F.mul(2, 3)
        Tensor(6, dtype=int32, device=xpux:0)

        >>> F.mul(2.0, 3.0)
        Tensor(6.0, device=xpux:0)

        >>> x = F.arange(6.0).reshape(2, 3))
        >>> y = F.arange(3.0)
        >>> F.mul(x, y)
        Tensor([[ 0.  1.  4.]
         [ 0.  4. 10.]], device=xpux:0)

        The `*` operator can be used as a shorthand for :func:`~.functional.mul` on tensors.

        >>> x = F.arange(6.0).reshape((2, 3))
        >>> y = F.arange(3.0)
        >>> x * y
        Tensor([[ 0.  1.  4.]
         [ 0.  4. 10.]], device=xpux:0)

    """
    return _elwise(x, y, mode=Elemwise.Mode.MUL)


def div(x, y):
    r"""Element-wise `(x / y)`."""
    return _elwise(x, y, mode=Elemwise.Mode.TRUE_DIV)


def floor_div(x, y):
    r"""Element-wise `floor(x / y)`."""
    return _elwise(x, y, mode=Elemwise.Mode.FLOOR_DIV)


def neg(x):
    r"""Element-wise `negation`."""
    return _elwise(x, mode=Elemwise.Mode.NEGATE)


def pow(x, y):
    r"""Element-wise `power`."""
    return _elwise(x, y, mode=Elemwise.Mode.POW)


def mod(x, y):
    r"""Element-wise `remainder of division`."""
    return _elwise(x, y, mode=Elemwise.Mode.MOD)


def abs(x):
    r"""Element-wise `absolute value`."""
    return _elwise(x, mode=Elemwise.Mode.ABS)


def exp(x):
    r"""Element-wise `exponential`."""
    return _elwise(x, mode=Elemwise.Mode.EXP)


def expm1(x):
    r"""Element-wise `exp(x)-1`."""
    return _elwise(x, mode=Elemwise.Mode.EXPM1)


def log(x):
    r"""Element-wise `logarithm (base e)`."""
    return _elwise(x, mode=Elemwise.Mode.LOG)


def log1p(x):
    r"""Element-wise `log(x+1) (base e)`."""
    return _elwise(x, mode=Elemwise.Mode.LOG1P)


def sqrt(x: Tensor) -> Tensor:
    r"""Element-wise `sqrt`.

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
    r"""Element-wise `square`.

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
    r"""Element-wise `rounding to int`."""
    return _elwise(x, mode=Elemwise.Mode.ROUND)


def ceil(x):
    r"""Element-wise `ceiling`."""
    return _elwise(x, mode=Elemwise.Mode.CEIL)


def floor(x):
    r"""Element-wise `floor`."""
    return _elwise(x, mode=Elemwise.Mode.FLOOR)


def maximum(x, y):
    r"""Element-wise `maximum of array elements`."""
    return _elwise(x, y, mode=Elemwise.Mode.MAX)


def minimum(x, y):
    r"""Element-wise `minimum of array elements`."""
    return _elwise(x, y, mode=Elemwise.Mode.MIN)


# trigonometric functions


def cos(x):
    r"""Element-wise `cosine`.

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
    r"""Element-wise `sine`."""
    return _elwise(x, mode=Elemwise.Mode.SIN)


def tan(x):
    r"""Element-wise `tangent`."""
    return sin(x) / cos(x)


def acos(x):
    r"""Element-wise `inverse cosine`."""
    return _elwise(x, mode=Elemwise.Mode.ACOS)


def asin(x):
    r"""Element-wise `inverse sine`."""
    return _elwise(x, mode=Elemwise.Mode.ASIN)


def atan(x):
    r"""Element-wise `inverse tangent`."""
    return _elwise(x, 1, mode=Elemwise.Mode.ATAN2)


def atan2(y, x):
    r"""Element-wise `2-argument arctangent`."""
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
    r"""Element-wise `bitwise binary: x << y`.

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
    r"""Element-wise `bitwise binary: x >> y`."""
    return _elwise(x, y, mode=Elemwise.Mode.SHR)


# logical functions


def logical_and(x, y):
    r"""Element-wise `logical and: x && y`."""
    return _elwise(x, y, mode=Elemwise.Mode.AND)


def logical_not(x):
    r"""Element-wise `logical not: ~x`."""
    return _elwise(x, mode=Elemwise.Mode.NOT)


def logical_or(x, y):
    r"""Element-wise `logical or: x || y`."""
    return _elwise(x, y, mode=Elemwise.Mode.OR)


def logical_xor(x, y):
    r"""Element-wise `logical xor: x ^ y`."""
    return _elwise(x, y, mode=Elemwise.Mode.XOR)


def logaddexp(x: Tensor, y: Tensor) -> Tensor:
    r"""Element-wise `numerically stable log(exp(x) + exp(y)`
    """
    return _elwise(x, y, mode=Elemwise.Mode.LOG_SUM_EXP)


# comparison functions


def equal(x, y):
    r"""Element-wise `(x == y)`.

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
    r"""Element-wise `(x != y)`."""
    return x != y


def less(x, y):
    r"""Element-wise `(x < y)`."""
    return _elwise(x, y, mode=Elemwise.Mode.LT)


def less_equal(x, y):
    r"""Element-wise `(x <= y)`."""
    return _elwise(x, y, mode=Elemwise.Mode.LEQ)


def greater(x, y):
    r"""Element-wise `(x > y)`."""
    return _elwise(y, x, mode=Elemwise.Mode.LT)


def greater_equal(x, y):
    r"""Element-wise `(x >= y)`."""
    return _elwise(y, x, mode=Elemwise.Mode.LEQ)


# other functions


def clip(x: Tensor, lower=None, upper=None) -> Tensor:
    r"""Clamps all elements in input tensor into the range ``[ lower, upper ]`` and returns
    a resulting tensor:

    .. math::

        y_i = \begin{cases}
            \text{lower} & \text{if } x_i < \text{lower} \\
            x_i & \text{if } \text{lower} \leq x_i \leq \text{upper} \\
            \text{upper} & \text{if } x_i > \text{upper}
        \end{cases}

    Args:
        x: input tensor.
        lower: lower-bound of the range to be clamped to.
        upper: upper-bound of the range to be clamped to.

    Returns:
        output clamped tensor.

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
            return minimum(maximum(x, lower), upper)
        else:
            return maximum(x, lower)
    else:
        return minimum(x, upper)


sigmoid = deprecated_func("1.3", "megengine.functional.nn", "sigmoid", True)
hsigmoid = deprecated_func("1.3", "megengine.functional.nn", "hsigmoid", True)
relu = deprecated_func("1.3", "megengine.functional.nn", "relu", True)
relu6 = deprecated_func("1.3", "megengine.functional.nn", "relu6", True)
hswish = deprecated_func("1.3", "megengine.functional.nn", "hswish", True)
