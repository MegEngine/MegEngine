# -*- coding: utf-8 -*-
# pylint: disable=unused-argument,invalid-name,redefined-builtin,arguments-out-of-order
import numpy as np

from ..core._imperative_rt.core2 import apply
from ..core.ops import builtin
from ..core.ops.builtin import Elemwise
from ..core.tensor.array_method import _elwise
from ..core.tensor.utils import convert_inputs
from ..tensor import Tensor
from ..utils.deprecation import deprecated_func
from .tensor_cache import get_scalar_one

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


def add(x: Tensor, y: Tensor) -> Tensor:
    r"""Calculates the sum for each element :math:`x_i` of the input tensor :math:`x` with the respective element :math:`y_i` of the input tensor :math:`y`.

    Args:
        x: first input tensor. Should have a numeric data type.
        y: second input tensor. Must be compatible with ``x`` (see :ref:`broadcasting-rule` ). Should have a numeric data type.

    Returns:
        A tensor containing the element-wise sums. The returned tensor must have a data type determined by :ref:`dtype-promotion`.

    .. admonition:: Special cases

       For floating-point operands,

       * If either :math:`x` or :math:`y` is ``NaN``, the result is ``NaN``.
       * If :math:`x` is ``+infinity`` and :math:`y` is ``-infinity``, the result is ``NaN``.
       * If :math:`x` is ``-infinity`` and :math:`y` is ``+infinity``, the result is ``NaN``.
       * If :math:`x` is ``+infinity`` and :math:`y` is ``+infinity``, the result is ``+infinity``.
       * If :math:`x` is ``-infinity`` and :math:`y` is ``-infinity``, the result is ``-infinity``.
       * If :math:`x` is ``+infinity`` and :math:`y` is a finite number, the result is ``+infinity``.
       * If :math:`x` is ``-infinity`` and :math:`y` is a finite number, the result is ``-infinity``.
       * If :math:`x` is a finite number and :math:`y` is ``+infinity``, the result is ``+infinity``.
       * If :math:`x` is a finite number and :math:`y` is ``-infinity``, the result is ``-infinity``.
       * If :math:`x` is ``-0`` and :math:`y` is ``-0``, the result is ``-0``.
       * If :math:`x` is ``-0`` and :math:`y` is ``+0``, the result is ``+0``.
       * If :math:`x` is ``+0`` and :math:`y` is ``-0``, the result is ``+0``.
       * If :math:`x` is ``+0`` and :math:`y` is ``+0``, the result is ``+0``.
       * If :math:`x` is either ``+0`` or ``-0`` and :math:`y` is a nonzero finite number, the result is :math:`y`.
       * If :math:`x` is a nonzero finite number and :math:`y` is either ``+0`` or ``-0``, the result is :math:`x`.
       * If :math:`x` is a nonzero finite number and :math:`y` is :math:`-x`, the result is ``+0``.
       * In the remaining cases, when neither ``infinity``, ``+0``, ``-0``, nor a ``NaN`` is involved, 
         and the operands have the same mathematical sign or have different magnitudes, 
         the sum must be computed and rounded to the nearest representable value according to 
         IEEE 754-2019 and a supported round mode. If the magnitude is too large to represent, 
         the operation overflows and the result is an infinity of appropriate mathematical sign.

    .. note::

       * Floating-point addition is a commutative operation, but not always associative.
       * The ``+`` operator can be used as a shorthand for ``add`` on tensors.

    Examples:
        >>> F.add(1.0, 4.0)
        Tensor(5.0, device=xpux:0)
        >>> x = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> y = Tensor([[1, 1, 1], [2, 2, 2]])
        >>> F.add(x, y)
        Tensor([[2 3 4]
         [6 7 8]], dtype=int32, device=xpux:0)
        >>> F.add(x, 1)
        Tensor([[2 3 4]
         [5 6 7]], dtype=int32, device=xpux:0)
    """
    return _elwise(x, y, mode=Elemwise.Mode.ADD)


def sub(x: Tensor, y: Tensor) -> Tensor:
    r"""Calculates the difference for each element :math:`x_i` of the input tensor :math:`x` with the respective element :math:`y` of the input tensor :math:`y`.
    The result of :math:`x_i - y_i` must be the same as :math:`x_i + (-y_i)` and must be governed by the same floating-point rules as addition.
    (See :func:`~.functional.add` ).

    Args:
        x: first input tensor. Should have a numeric data type.
        y: second input tensor. Must be compatible with ``x`` (see :ref:`broadcasting-rule` ). Should have a numeric data type.

    Returns:
        A tensor containing the element-wise differences. The returned tensor must have a data type determined by :ref:`dtype-promotion`.

    .. note::

       The ``-`` operator can be used as a shorthand for ``sub`` on Tensors.

    Examples:
       >>> F.sub(1.0, 4.0)
       Tensor(-3.0, device=xpux:0)
       >>> x = Tensor([[1, 2, 3], [4, 5, 6]])
       >>> y = Tensor([[1, 1, 1], [2, 2, 2]])
       >>> F.sub(x, y)
       Tensor([[0 1 2]
        [2 3 4]], dtype=int32, device=xpux:0)
       >>> F.sub(x, 1)
       Tensor([[0 1 2]
        [3 4 5]], dtype=int32, device=xpux:0)
    """
    return _elwise(x, y, mode=Elemwise.Mode.SUB)


def mul(x: Tensor, y: Tensor) -> Tensor:
    r"""Calculates the product for each element :math:`x_i` of the input tensor `x` with the respective element :math:`y_i` of the input tensor :math:`y`.

    Args:
        x: first input tensor. Should have a numeric data type.
        y: second input tensor. Must be compatible with ``x`` (see :ref:`broadcasting-rule` ). Should have a numeric data type.

    Returns:
        A tensor containing the element-wise products. The returned tensor must have a data type determined by :ref:`dtype-promotion`.

    .. admonition:: Special cases

       For floating-point operands,

       * If either :math:`x_i` or :math:`y_i` is ``NaN``, the result is ``NaN``.
       * If :math:`x_i` is either ``+infinity`` or ``-infinity`` and :math:`y_i` is either ``+0`` or ``-0``, the result is ``NaN``.
       * If :math:`x_i` is either ``+0`` or ``-0`` and :math:`y_i` is either ``+infinity`` or ``-infinity``, the result is ``NaN``.
       * If :math:`x_i` and :math:`y_i` have different mathematical signs, the result has a negative mathematical sign, unless the result is ``NaN``.
       * If :math:`x_i` is either ``+infinity`` or ``-infinity`` and :math:`y_i` is either ``+infinity`` or ``-infinity``,
         the result is a signed infinity with the mathematical sign determined by the rule already stated above.
       * If :math:`x_i` is either ``+infinity`` or ``-infinity`` and :math:`y_i` is a nonzero finite number,
         the result is a signed infinity with the mathematical sign determined by the rule already stated above.
       * If :math:`x_i` is a nonzero finite number and :math:`y_i` is either ``+infinity`` or ``-infinity``,
         the result is a signed infinity with the mathematical sign determined by the rule already stated above.
       * In the remaining cases, where neither ``infinity`` nor ``NaN`` is involved,
         the product must be computed and rounded to the nearest representable value according to IEEE 754-2019 and a supported rounding mode.
         If the magnitude is too large to represent, the result is an `infinity` of appropriate mathematical sign.
         If the magnitude is too small to represent, the result is a zero of appropriate mathematical sign.

    .. Note::

       * Floating-point multiplication is not always associative due to finite precision.
       * The ``*`` operator can be used as a shorthand for ``mul`` on tensors.

    Examples:
        >>> F.mul(1.0, 4.0)
        Tensor(4.0, device=xpux:0)
        >>> x = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> y = Tensor([[1, 1, 1], [2, 2, 2]])
        >>> F.mul(x, y)
        Tensor([[ 1  2  3]
         [ 8 10 12]], dtype=int32, device=xpux:0)
        >>> F.mul(x, 2)
        Tensor([[ 2  4  6]
         [ 8 10 12]], dtype=int32, device=xpux:0)
    """
    return _elwise(x, y, mode=Elemwise.Mode.MUL)


def div(x: Tensor, y: Tensor) -> Tensor:
    r"""Calculates the division for each element :math:`x_i` of the input tensor :math:`x` with the respective element :math:`y_i` of the input tensor :math:`y`.

    Args:
        x: dividend input tensor. Should have a numeric data type.
        y: divisor input tensor. Must be compatible with ``x``` (see :ref:`broadcasting-rule` ). Should have a numeric data type.

    Returns:
        A tensor containing the element-wise results. The returned tensor must have a data type determined by :ref:`dtype-promotion`.

    .. admonition:: Special cases

       For floating-point operands,

       * If either :math:`x` or :math:`y` is ``NaN``, the result is ``NaN``.
       * If :math:`x` is either ``+infinity`` or ``-infinity`` and :math:`y` is either ``+infinity`` or ``-infinity``, the result is ``NaN``.
       * If :math:`x` is either ``+0`` or ``-0`` and :math:`y` is either ``+0`` or ``-0``, the result is ``NaN``.
       * If :math:`x` is ``+0`` and :math:`y` is greater than ``0``, the result is ``+0``.
       * If :math:`x` is ``-0`` and :math:`y` is greater than ``0``, the result is ``-0``.
       * If :math:`x` is ``+0`` and :math:`y` is less than ``0``, the result is ``-0``.
       * If :math:`x` is ``-0`` and :math:`y` is less than ``0``, the result is ``+0``.
       * If :math:`x` is greater than ``0`` and :math:`y` is ``+0``, the result is ``+infinity``.
       * If :math:`x` is greater than ``0`` and :math:`y` is ``-0``, the result is ``-infinity``.
       * If :math:`x` is less than ``0`` and :math:`y` is ``+0``, the result is ``-infinity``.
       * If :math:`x` is less than ``0`` and :math:`y` is ``-0``, the result is ``+infinity``.
       * If :math:`x` is ``+infinity`` and :math:`y` is a positive (i.e., greater than ``0``) finite number, the result is ``+infinity``.
       * If :math:`x` is ``+infinity`` and :math:`y` is a negative (i.e., less than ``0``) finite number, the result is ``-infinity``.
       * If :math:`x` is ``-infinity`` and :math:`y` is a positive (i.e., greater than ``0``) finite number, the result is ``-infinity``.
       * If :math:`x` is ``-infinity`` and :math:`y` is a negative (i.e., less than ``0``) finite number, the result is ``+infinity``.
       * If :math:`x` is a positive (i.e., greater than ``0``) finite number and :math:`y` is ``+infinity``, the result is ``+0``.
       * If :math:`x` is a positive (i.e., greater than ``0``) finite number and :math:`y` is ``-infinity``, the result is ``-0``.
       * If :math:`x` is a negative (i.e., less than ``0``) finite number and :math:`y` is ``+infinity``, the result is ``-0``.
       * If :math:`x` is a negative (i.e., less than ``0``) finite number and :math:`y` is ``-infinity``, the result is ``+0``.
       * If :math:`x` and :math:`y` have the same mathematical sign and are both nonzero finite numbers, the result has a positive mathematical sign.
       * If :math:`x` and :math:`y` have different mathematical signs and are both nonzero finite numbers, the result has a negative mathematical sign.
       * In the remaining cases, where neither ``-infinity``, ``+0``, ``-0``, nor ``NaN`` is involved, the quotient must be computed and rounded to the nearest representable value according to IEEE 754-2019 and a supported rounding mode. If the magnitude is too large to represent, the operation overflows and the result is an infinity of appropriate mathematical sign. If the magnitude is too small to represent, the operation underflows and the result is a zero of appropriate mathematical sign.

    .. note::

       The ``/`` operator can be used as a shorthand for ``div`` on tensors.

    .. seealso::

       In Python, ``//`` is the floor division operator and ``/`` is the true division operator.
       See :func:`~.functional.floor_div`

    Examples:
        >>> F.div(1.0, 4.0)
        Tensor(0.25, device=xpux:0)
        >>> x = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> y = Tensor([[1, 1, 1], [2, 2, 2]])
        >>> F.div(x, y)
        Tensor([[1.  2.  3. ]
         [2.  2.5 3. ]], device=xpux:0)
        >>> F.div(x, 2)
        Tensor([[0.5 1.  1.5]
         [2.  2.5 3. ]], device=xpux:0)
    """
    return _elwise(x, y, mode=Elemwise.Mode.TRUE_DIV)


def floor_div(x: Tensor, y: Tensor) -> Tensor:
    r"""Rounds the result of dividing each element :math:`x_i` of the input tensor :math:`x` 
    by the respective element :math:`y_i` of the input tensor :math:`y` to the greatest 
    (i.e., closest to ``+infinity``) integer-value number that is not greater than the division result.

    Args:
        x: dividend input tensor. Should have a numeric data type.
        y: divisor input tensor. Must be compatible with ``x``` (see :ref:`broadcasting-rule` ). Should have a numeric data type.

    Returns:
        A tensor containing the element-wise results. The returned tensor must have a data type determined by :ref:`dtype-promotion`.

    .. admonition:: Special cases

       For floating-point operands,

       * If either :math:`x` or :math:`y` is ``NaN``, the result is ``NaN``.
       * If :math:`x` is either ``+infinity`` or ``-infinity`` and :math:`y` is either ``+infinity`` or ``-infinity``, the result is ``NaN``.
       * If :math:`x` is either ``+0`` or ``-0`` and :math:`y` is either ``+0`` or ``-0``, the result is ``NaN``.
       * If :math:`x` is ``+0`` and :math:`y` is greater than ``0``, the result is ``+0``.
       * If :math:`x` is ``-0`` and :math:`y` is greater than ``0``, the result is ``-0``.
       * If :math:`x` is ``+0`` and :math:`y` is less than ``0``, the result is ``-0``.
       * If :math:`x` is ``-0`` and :math:`y` is less than ``0``, the result is ``+0``.
       * If :math:`x` is greater than ``0`` and :math:`y` is ``+0``, the result is ``+infinity``.
       * If :math:`x` is greater than ``0`` and :math:`y` is ``-0``, the result is ``-infinity``.
       * If :math:`x` is less than ``0`` and :math:`y` is ``+0``, the result is ``-infinity``.
       * If :math:`x` is less than ``0`` and :math:`y` is ``-0``, the result is ``+infinity``.
       * If :math:`x` is ``+infinity`` and :math:`y` is a positive (i.e., greater than ``0``) finite number, the result is ``+infinity``.
       * If :math:`x` is ``+infinity`` and :math:`y` is a negative (i.e., less than ``0``) finite number, the result is ``-infinity``.
       * If :math:`x` is ``-infinity`` and :math:`y` is a positive (i.e., greater than ``0``) finite number, the result is ``-infinity``.
       * If :math:`x` is ``-infinity`` and :math:`y` is a negative (i.e., less than ``0``) finite number, the result is ``+infinity``.
       * If :math:`x` is a positive (i.e., greater than ``0``) finite number and :math:`y` is ``+infinity``, the result is ``+0``.
       * If :math:`x` is a positive (i.e., greater than ``0``) finite number and :math:`y` is ``-infinity``, the result is ``-0``.
       * If :math:`x` is a negative (i.e., less than ``0``) finite number and :math:`y` is ``+infinity``, the result is ``-0``.
       * If :math:`x` is a negative (i.e., less than ``0``) finite number and :math:`y` is ``-infinity``, the result is ``+0``.
       * If :math:`x` and :math:`y` have the same mathematical sign and are both nonzero finite numbers, the result has a positive mathematical sign.
       * If :math:`x` and :math:`y` have different mathematical signs and are both nonzero finite numbers, the result has a negative mathematical sign.
       * In the remaining cases, where neither ``-infinity``, ``+0``, ``-0``, nor ``NaN`` is involved, the quotient must be computed and rounded to the nearest representable value according to IEEE 754-2019 and a supported rounding mode. If the magnitude is too large to represent, the operation overflows and the result is an infinity of appropriate mathematical sign. If the magnitude is too small to represent, the operation underflows and the result is a zero of appropriate mathematical sign.

    .. note::

       The ``//`` operator can be used as a shorthand for ``floor_div`` on tensors.

    .. seealso::

       In Python, ``//`` is the floor division operator and ``/`` is the true division operator.
       See :func:`~.functional.div`

    Examples:
       >>> F.floor_div(5.0, 4.0)
       Tensor(1.0, device=xpux:0)
       >>> x = Tensor([[1, 2, 3], [4, 5, 6]])
       >>> y = Tensor([[1, 1, 1], [2, 2, 2]])
       >>> F.floor_div(x, y)
       Tensor([[1 2 3]
        [2 2 3]], dtype=int32, device=xpux:0)
       >>> F.floor_div(x, 2)
       Tensor([[0 1 1]
        [2 2 3]], dtype=int32, device=xpux:0)
    """
    return _elwise(x, y, mode=Elemwise.Mode.FLOOR_DIV)


def neg(x: Tensor) -> Tensor:
    r"""Computes the numerical negative of each element :math:`x_i` (i.e., :math:`y_i = -x_i` ) of the input tensor :math:`x`.

    Args:
        x: input tensor. Should have a numeric data type.

    Returns:
        A tensor containing the evaluated result for each element in :math:`x`.
        The returned tensor must have a data type determined by :ref:`dtype-promotion`.

    .. note::

       The unary ``-`` operator can be used as a shorthand for ``neg`` on tensors.

    Examples:
       >>> x = Tensor([1, -1])
       >>> F.neg(x)
       Tensor([-1  1], dtype=int32, device=xpux:0)
    """
    return _elwise(x, mode=Elemwise.Mode.NEGATE)


def pow(x: Tensor, y: Tensor) -> Tensor:
    r"""Calculates an implementation-dependent approximation of exponentiation by 
    raising each element :math:`x_i` (the base) of the input tensor :math:`x` to 
    the power of :math:`y_i` (the exponent), where :math:`y_i` is the corresponding element of the input tensor :math:`y`.

    Args:
        x: first input tensor whose elements correspond to the exponentiation base. Should have a numeric data type.
        y: second input tensor whose elements correspond to the exponentiation exponent. Must be compatible with `x` (see :ref:`broadcasting-rule` ). Should have a numeric data type.

    Returns:
        A tensor containing the element-wise results. The returned tensor must have a data type determined by :ref:`dtype-promotion`.

    .. note::

       The unary ``**`` operator can be used as a shorthand for ``pow`` on tensors.

    .. admonition:: Special cases

       For floating-point operands,

       * If :math:`x_i` is not equal to ``1`` and :math:`y_i` is ``NaN``, the result is ``NaN``.
       * If :math:`y_i` is ``+0``, the result is ``1``, even if ``x_i`` is ``NaN``.
       * If :math:`y_i` is ``-0``, the result is ``1``, even if ``x_i`` is ``NaN``.
       * If :math:`x_i` is ``NaN`` and ``y_i`` is not equal to ``0``, the result is ``NaN``.
       * If ``abs(x_i)`` is greater than ``1`` and ``y_i`` is ``+infinity``, the result is ``+infinity``.
       * If ``abs(x_i)`` is greater than ``1`` and ``y_i`` is ``-infinity``, the result is ``+0``.
       * If ``abs(x_i)`` is ``1`` and ``y_i`` is ``+infinity``, the result is ``1``.
       * If ``abs(x_i)`` is ``1`` and ``y_i`` is ``-infinity``, the result is ``1``.
       * If ``x_i`` is ``1`` and ``y_i`` is not ``NaN``, the result is ``1``.
       * If ``abs(x_i)`` is less than ``1`` and ``y_i`` is ``+infinity``, the result is ``+0``.
       * If ``abs(x_i)`` is less than ``1`` and ``y_i`` is ``-infinity``, the result is ``+infinity``.
       * If ``x_i`` is ``+infinity`` and ``y_i`` is greater than 0, the result is ``+infinity``.
       * If ``x_i`` is ``+infinity`` and ``y_i`` is less than 0, the result is ``+0``.
       * If ``x_i`` is ``-infinity``, ``y_i`` is greater than 0, and ``y_i`` is an odd integer value, the result is ``-infinity``.
       * If ``x_i`` is ``-infinity``, ``y_i`` is greater than 0, and ``y_i`` is not an odd integer value, the result is ``+infinity``.
       * If ``x_i`` is ``-infinity``, ``y_i`` is less than 0, and ``y_i`` is an odd integer value, the result is ``-0``.
       * If ``x_i`` is ``-infinity``, ``y_i`` is less than 0, and ``y_i`` is not an odd integer value, the result is ``+0``.
       * If ``x_i`` is ``+0`` and ``y_i`` is greater than 0, the result is ``+0``.
       * If ``x_i`` is ``+0`` and ``y_i`` is less than 0, the result is ``+infinity``.
       * If ``x_i`` is ``-0``, ``y_i`` is greater than 0, and ``y_i`` is an odd integer value, the result is ``-0``.
       * If ``x_i`` is ``-0``, ``y_i`` is greater than 0, and ``y_i`` is not an odd integer value, the result is ``+0``.
       * If ``x_i`` is ``-0``, ``y_i`` is less than 0, and ``y_i`` is an odd integer value, the result is ``-infinity``.
       * If ``x_i`` is ``-0``, ``y_i`` is less than 0, and ``y_i`` is not an odd integer value, the result is ``+infinity``.
       * If ``x_i`` is less than 0, ``x_i`` is a finite number, ``y_i`` is a finite number, and ``y_i`` is not an integer value, the result is ``NaN``.

    Examples:
        >>> F.pow(2.0, 3.0)
        Tensor(8.0, device=xpux:0)
        >>> x = Tensor([1, 2, 3, 4, 5])
        >>> y = Tensor([1, 2, 1, 2, 1])
        >>> F.pow(x, y)
        Tensor([ 1.  4.  3. 16.  5.], device=xpux:0)
        >>> F.pow(x, 2)
        Tensor([ 1.  4.  9. 16. 25.], device=xpux:0)
    """
    return _elwise(x, y, mode=Elemwise.Mode.POW)


def mod(x: Tensor, y: Tensor) -> Tensor:
    r"""Returns the remainder of division for each element ``x_i`` of the input tensor ``x``
    and the respective element ``y_i`` of the input tensor ``y``.

    .. note:: ``mod`` is an alias of ``remainder`` in NumPy.

    .. seealso:: :func:`~.div` / :func:`~.floor_div`

    Args:
        x: dividend input tensor. Should have a numeric data type.
        y: divisor input tensor. Must be compatible with ``x`` (see :ref:`broadcasting-rule` ). Should have a numeric data type.

    Returns:
        A tensor containing the element-wise results. The returned tensor must have a data type determined by :ref:`dtype-promotion`.

    Examples:
        >>> F.mod(8, 3)
        Tensor(2, dtype=int32, device=xpux:0)
        >>> x = Tensor([1, 2, 3, 4, 5])
        >>> y = Tensor([1, 2, 1, 2, 1])
        >>> F.mod(x, y)
        Tensor([0 0 0 0 0], dtype=int32, device=xpux:0)
        >>> F.mod(x, 3)
        Tensor([1 2 0 1 2], dtype=int32, device=xpux:0)
    """
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
    r"""Element-wise `sqrt`."""
    return x ** 0.5


def square(x: Tensor) -> Tensor:
    r"""Element-wise `square`."""
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
    r"""Element-wise `cosine`."""
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
    return _elwise(
        x,
        get_scalar_one("float32", x.device if isinstance(x, Tensor) else None),
        mode=Elemwise.Mode.ATAN2,
    )


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
    r"""Element-wise `bitwise binary: x << y`."""
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
    r"""Element-wise `numerically stable log(exp(x) + exp(y)`."""
    return _elwise(x, y, mode=Elemwise.Mode.LOG_SUM_EXP)


# comparison functions


def equal(x, y):
    r"""Element-wise `(x == y)`."""
    return x == y


def not_equal(x, y):
    r"""Element-wise `(x != y)`."""
    return x != y


def less(x, y):
    r"""Element-wise `(x < y)`."""
    return x < y


def less_equal(x, y):
    r"""Element-wise `(x <= y)`."""
    return x <= y


def greater(x, y):
    r"""Element-wise `(x > y)`."""
    return x > y


def greater_equal(x, y):
    r"""Element-wise `(x >= y)`."""
    return x >= y


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
        x: (Tensor): The input tensor.
        lower: (Numberic，optional): lower-bound of the range to be clamped to.
        upper: (Numberic，optional): upper-bound of the range to be clamped to.        

    Note:
        * If both `lower` and `upper` are None, raises an AssertionError.
        * If `lower` is bigger than `upper`, the result is same as `clip(Tensor(), upper, upper)`.

    Returns:
        output clamped tensor. The result must have a data type determined by :ref:`dtype-promotion`.

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
