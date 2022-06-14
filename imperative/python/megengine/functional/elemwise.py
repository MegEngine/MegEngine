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


# arithmetic operations


def add(x: Tensor, y: Tensor) -> Tensor:
    r"""Element-wise addition.
    
    Calculates the sum for each element :math:`x_i` of the input tensor :math:`x` with the respective element :math:`y_i` of the input tensor :math:`y`.

    Args:
        x: first input tensor. Should have a numeric data type.
        y: second input tensor. Must be compatible with :math:`x` (see :ref:`broadcasting-rule` ). Should have a numeric data type.

    Returns:
        A tensor containing the element-wise sums.
        The returned tensor must have a data type determined by :ref:`dtype-promotion`.

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

        Element-wise addition:

        >>> x = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> y = Tensor([[1, 1, 1], [2, 2, 2]])
        >>> F.add(x, y)
        Tensor([[2 3 4]
         [6 7 8]], dtype=int32, device=xpux:0)
        
        Broadcasting:

        >>> x = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> F.add(x, 1)
        Tensor([[2 3 4]
         [5 6 7]], dtype=int32, device=xpux:0)
    """
    return _elwise(x, y, mode=Elemwise.Mode.ADD)


def sub(x: Tensor, y: Tensor) -> Tensor:
    r"""Element-wise subtraction.
    
    Calculates the difference for each element :math:`x_i` of the input tensor :math:`x` with the respective element :math:`y` of the input tensor :math:`y`.
    The result of :math:`x_i - y_i` must be the same as :math:`x_i + (-y_i)` and must be governed by the same floating-point rules as addition.
    (See :func:`~.functional.add` ).

    Args:
        x: first input tensor. Should have a numeric data type.
        y: second input tensor. Must be compatible with :math:`x` (see :ref:`broadcasting-rule` ). Should have a numeric data type.

    Returns:
        A tensor containing the element-wise differences.
        The returned tensor must have a data type determined by :ref:`dtype-promotion`.

    .. note::

       The ``-`` operator can be used as a shorthand for ``sub`` on Tensors.

    Examples:
       >>> F.sub(1.0, 4.0)
       Tensor(-3.0, device=xpux:0)

       Element-wise subtraction:

       >>> x = Tensor([[1, 2, 3], [4, 5, 6]])
       >>> y = Tensor([[1, 1, 1], [2, 2, 2]])
       >>> F.sub(x, y)
       Tensor([[0 1 2]
        [2 3 4]], dtype=int32, device=xpux:0)

       Broadcasting:

       >>> x = Tensor([[1, 2, 3], [4, 5, 6]])
       >>> F.sub(x, 1)
       Tensor([[0 1 2]
        [3 4 5]], dtype=int32, device=xpux:0)
    """
    return _elwise(x, y, mode=Elemwise.Mode.SUB)


def mul(x: Tensor, y: Tensor) -> Tensor:
    r"""Element-wise multiplication.
    
    Calculates the product for each element :math:`x_i` of the input tensor `x` with the respective element :math:`y_i` of the input tensor :math:`y`.

    Args:
        x: first input tensor. Should have a numeric data type.
        y: second input tensor. Must be compatible with :math:`x` (see :ref:`broadcasting-rule` ). Should have a numeric data type.

    Returns:
        A tensor containing the element-wise products.
        The returned tensor must have a data type determined by :ref:`dtype-promotion`.

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

        Element-wise multiplication:

        >>> x = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> y = Tensor([[1, 1, 1], [2, 2, 2]])
        >>> F.mul(x, y)
        Tensor([[ 1  2  3]
         [ 8 10 12]], dtype=int32, device=xpux:0)
        
        Boradcasting:

        >>> x = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> F.mul(x, 2)
        Tensor([[ 2  4  6]
         [ 8 10 12]], dtype=int32, device=xpux:0)
    """
    return _elwise(x, y, mode=Elemwise.Mode.MUL)


def div(x: Tensor, y: Tensor) -> Tensor:
    r"""Element-wise division.
    
    Calculates the division for each element :math:`x_i` of the input tensor :math:`x` with the respective element :math:`y_i` of the input tensor :math:`y`.

    Args:
        x: dividend input tensor. Should have a numeric data type.
        y: divisor input tensor. Must be compatible with :math:`x`` (see :ref:`broadcasting-rule` ). Should have a numeric data type.

    Returns:
        A tensor containing the element-wise results.
        The returned tensor must have a data type determined by :ref:`dtype-promotion`.

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
       * In the remaining cases, where neither ``-infinity``, ``+0``, ``-0``, nor ``NaN`` is involved,
         the quotient must be computed and rounded to the nearest representable value according to IEEE 754-2019 and a supported rounding mode.
         If the magnitude is too large to represent, the operation overflows and the result is an infinity of appropriate mathematical sign.
         If the magnitude is too small to represent, the operation underflows and the result is a zero of appropriate mathematical sign.

    .. note::

       The ``/`` operator can be used as a shorthand for ``div`` on tensors.

    .. seealso::

       In Python, ``//`` is the floor division operator and ``/`` is the true division operator.
       See :func:`~.functional.floor_div`

    Examples:
        >>> F.div(1.0, 4.0)
        Tensor(0.25, device=xpux:0)

        Element-wise division:

        >>> x = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> y = Tensor([[1, 1, 1], [2, 2, 2]])
        >>> F.div(x, y)
        Tensor([[1.  2.  3. ]
         [2.  2.5 3. ]], device=xpux:0)

        Broadcasting:

        >>> x = Tensor([[1, 2, 3], [4, 5, 6]])
        >>> F.div(x, 2)
        Tensor([[0.5 1.  1.5]
         [2.  2.5 3. ]], device=xpux:0)
    """
    return _elwise(x, y, mode=Elemwise.Mode.TRUE_DIV)


def floor_div(x: Tensor, y: Tensor) -> Tensor:
    r"""Element-wise floor division.
    
    Rounds the result of dividing each element :math:`x_i` of the input tensor :math:`x` 
    by the respective element :math:`y_i` of the input tensor :math:`y` to the greatest 
    (i.e., closest to ``+infinity``) integer-value number that is not greater than the division result.

    Args:
        x: dividend input tensor. Should have a numeric data type.
        y: divisor input tensor. Must be compatible with :math:`x`` (see :ref:`broadcasting-rule` ). Should have a numeric data type.

    Returns:
        A tensor containing the element-wise results.
        The returned tensor must have a data type determined by :ref:`dtype-promotion`.

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
       * In the remaining cases, where neither ``-infinity``, ``+0``, ``-0``, nor ``NaN`` is involved, 
         the quotient must be computed and rounded to the nearest representable value according to IEEE 754-2019 and a supported rounding mode.
         If the magnitude is too large to represent, the operation overflows and the result is an infinity of appropriate mathematical sign.
         If the magnitude is too small to represent, the operation underflows and the result is a zero of appropriate mathematical sign.

    .. note::

       The ``//`` operator can be used as a shorthand for ``floor_div`` on tensors.

    .. seealso::

       In Python, ``//`` is the floor division operator and ``/`` is the true division operator.
       See :func:`~.functional.div`

    Examples:
       >>> F.floor_div(5.0, 4.0)
       Tensor(1.0, device=xpux:0)

       Element-wise floor division:

       >>> x = Tensor([[1, 2, 3], [4, 5, 6]])
       >>> y = Tensor([[1, 1, 1], [2, 2, 2]])
       >>> F.floor_div(x, y)
       Tensor([[1 2 3]
        [2 2 3]], dtype=int32, device=xpux:0)

       Broadcasting:

       >>> x = Tensor([[1, 2, 3], [4, 5, 6]])
       >>> F.floor_div(x, 2)
       Tensor([[0 1 1]
        [2 2 3]], dtype=int32, device=xpux:0)
    """
    return _elwise(x, y, mode=Elemwise.Mode.FLOOR_DIV)


def neg(x: Tensor) -> Tensor:
    r"""Element-wise negation.
    
    Computes the numerical negative of each element :math:`x_i` (i.e., :math:`y_i = -x_i` ) of the input tensor :math:`x`.

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
    r"""Element-wise power.
    
    Calculates an implementation-dependent approximation of exponentiation by 
    raising each element :math:`x_i` (the base) of the input tensor :math:`x` to 
    the power of :math:`y_i` (the exponent), where :math:`y_i` is the corresponding element of the input tensor :math:`y`.

    Args:
        x: first input tensor whose elements correspond to the exponentiation base. Should have a numeric data type.
        y: second input tensor whose elements correspond to the exponentiation exponent.
            Must be compatible with `x` (see :ref:`broadcasting-rule` ). Should have a numeric data type.

    Returns:
        A tensor containing the element-wise results.
        The returned tensor must have a data type determined by :ref:`dtype-promotion`.

    .. note::

       The unary ``**`` operator can be used as a shorthand for ``pow`` on tensors.

    .. admonition:: Special cases

       For floating-point operands,

       * If :math:`x_i` is not equal to ``1`` and :math:`y_i` is ``NaN``, the result is ``NaN``.
       * If :math:`y_i` is ``+0``, the result is ``1``, even if :math:`x_i` is ``NaN``.
       * If :math:`y_i` is ``-0``, the result is ``1``, even if :math:`x_i` is ``NaN``.
       * If :math:`x_i` is ``NaN`` and :math:`y_i` is not equal to ``0``, the result is ``NaN``.
       * If :math:`\abs{x_i}` is greater than ``1`` and :math:`y_i` is ``+infinity``, the result is ``+infinity``.
       * If :math:`\abs{x_i}` is greater than ``1`` and :math:`y_i` is ``-infinity``, the result is ``+0``.
       * If :math:`\abs{x_i}` is ``1`` and :math:`y_i` is ``+infinity``, the result is ``1``.
       * If :math:`\abs{x_i}` is ``1`` and :math:`y_i` is ``-infinity``, the result is ``1``.
       * If :math:`x_i` is ``1`` and :math:`y_i` is not ``NaN``, the result is ``1``.
       * If :math:`\abs{x_i}` is less than ``1`` and :math:`y_i` is ``+infinity``, the result is ``+0``.
       * If :math:`\abs{x_i}` is less than ``1`` and :math:`y_i` is ``-infinity``, the result is ``+infinity``.
       * If :math:`x_i` is ``+infinity`` and :math:`y_i` is greater than ``0``, the result is ``+infinity``.
       * If :math:`x_i` is ``+infinity`` and :math:`y_i` is less than ``0``, the result is ``+0``.
       * If :math:`x_i` is ``-infinity``, :math:`y_i` is greater than ``0``, and :math:`y_i` is an odd integer value, the result is ``-infinity``.
       * If :math:`x_i` is ``-infinity``, :math:`y_i` is greater than ``0``, and :math:`y_i` is not an odd integer value, the result is ``+infinity``.
       * If :math:`x_i` is ``-infinity``, :math:`y_i` is less than ``0``, and :math:`y_i` is an odd integer value, the result is ``-0``.
       * If :math:`x_i` is ``-infinity``, :math:`y_i` is less than ``0``, and :math:`y_i` is not an odd integer value, the result is ``+0``.
       * If :math:`x_i` is ``+0`` and :math:`y_i` is greater than ``0``, the result is ``+0``.
       * If :math:`x_i` is ``+0`` and :math:`y_i` is less than ``0``, the result is ``+infinity``.
       * If :math:`x_i` is ``-0``, :math:`y_i` is greater than ``0``, and :math:`y_i` is an odd integer value, the result is ``-0``.
       * If :math:`x_i` is ``-0``, :math:`y_i` is greater than ``0``, and :math:`y_i` is not an odd integer value, the result is ``+0``.
       * If :math:`x_i` is ``-0``, :math:`y_i` is less than ``0``, and :math:`y_i` is an odd integer value, the result is ``-infinity``.
       * If :math:`x_i` is ``-0``, :math:`y_i` is less than ``0``, and :math:`y_i` is not an odd integer value, the result is ``+infinity``.
       * If :math:`x_i` is less than 0, :math:`x_i` is a finite number, :math:`y_i` is a finite number, and :math:`y_i` is not an integer value, the result is ``NaN``.

    Examples:
        >>> F.pow(2.0, 3.0)
        Tensor(8.0, device=xpux:0)

        Element-wise power:

        >>> x = Tensor([1, 2, 3, 4, 5])
        >>> y = Tensor([1, 2, 1, 2, 1])
        >>> F.pow(x, y)
        Tensor([ 1.  4.  3. 16.  5.], device=xpux:0)

        Broadcasting:

        >>> F.pow(x, 2)
        Tensor([ 1.  4.  9. 16. 25.], device=xpux:0)
    """
    return _elwise(x, y, mode=Elemwise.Mode.POW)


def mod(x: Tensor, y: Tensor) -> Tensor:
    r"""Element-wise :math:`\operatorname{mod}(x, y)` function.
    
    Returns the remainder of division for each element :math:`x_i` of the input tensor :math:`x`
    and the respective element :math:`y_i` of the input tensor :math:`y`.

    .. note:: 
    
       * In general, similar to Python’s % operator, this function is not recommended for floating-point operands as semantics do not follow IEEE 754.
         That this function is specified to accept floating-point operands is primarily for reasons of backward compatibility.
       * ``mod`` is an alias of ``remainder`` in NumPy.

    .. seealso:: :func:`~.div` / :func:`~.floor_div`

    Args:
        x: dividend input tensor. Should have a numeric data type.
        y: divisor input tensor. Must be compatible with :math:`x` (see :ref:`broadcasting-rule` ). Should have a numeric data type.

    Returns:
        A tensor containing the element-wise results.
        The returned tensor must have a data type determined by :ref:`dtype-promotion`.

    .. admonition:: Special cases

       For floating-point operands,

       * If either :math:`x_i` or :math:`y_i` is ``NaN``, the result is ``NaN``.
       * If :math:`x_i` is either ``+infinity`` or ``-infinity`` and :math:`y_i` is either ``+infinity`` or ``-infinity``, the result is ``NaN``.
       * If :math:`x_i` is either ``+0`` or ``-0`` and :math:`y_i` is either ``+0`` or ``-0``, the result is ``NaN``.
       * If :math:`x_i` is ``+0`` and :math:`y_i` is greater than 0, the result is ``+0``.
       * If :math:`x_i` is ``-0`` and :math:`y_i` is greater than 0, the result is ``+0``.
       * If :math:`x_i` is ``+0`` and :math:`y_i` is less than 0, the result is ``-0``.
       * If :math:`x_i` is ``-0`` and :math:`y_i` is less than 0, the result is ``-0``.
       * If :math:`x_i` is greater than ``0`` and :math:`y_i` is ``+0``, the result is ``NaN``.
       * If :math:`x_i` is greater than ``0`` and :math:`y_i` is ``-0``, the result is ``NaN``.
       * If :math:`x_i` is less than ``0`` and :math:`y_i` is ``+0``, the result is ``NaN``.
       * If :math:`x_i` is less than ``0`` and :math:`y_i` is ``-0``, the result is ``NaN``.
       * If :math:`x_i` is ``+infinity`` and :math:`y_i` is a positive (i.e., greater than 0) finite number, the result is ``NaN``.
       * If :math:`x_i` is ``+infinity`` and :math:`y_i` is a negative (i.e., less than 0) finite number, the result is ``NaN``.
       * If :math:`x_i` is ``-infinity`` and :math:`y_i` is a positive (i.e., greater than 0) finite number, the result is ``NaN``.
       * If :math:`x_i` is ``-infinity`` and :math:`y_i` is a negative (i.e., less than 0) finite number, the result is ``NaN``.
       * If :math:`x_i` is a positive (i.e., greater than ``0``) finite number and :math:`y_i` is ``+infinity``, the result is :math:`x_i`. 
         (note: this result matches Python behavior.)
       * If :math:`x_i` is a positive (i.e., greater than ``0``) finite number and :math:`y_i` is ``-infinity``, the result is :math:`y_i`.
         (note: this result matches Python behavior.)
       * If :math:`x_i` is a negative (i.e., less than ``0``) finite number and :math:`y_i` is ``+infinity``, the result is :math:`y_i`.
         (note: this results matches Python behavior.)
       * If :math:`x_i` is a negative (i.e., less than ``0``) finite number and :math:`y_i` is ``-infinity``, the result is :math:`x_i`.
         (note: this result matches Python behavior.)
       * In the remaining cases, the result must match that of the Python ``%`` operator.

    Examples:
        >>> F.mod(8, 3)
        Tensor(2, dtype=int32, device=xpux:0)

        Element-wise mod:

        >>> x = Tensor([1, 2, 3, 4, 5])
        >>> y = Tensor([1, 2, 1, 2, 1])
        >>> F.mod(x, y)
        Tensor([0 0 0 0 0], dtype=int32, device=xpux:0)

        Broadcasting:

        >>> x = Tensor([1, 2, 3, 4, 5])
        >>> F.mod(x, 3)
        Tensor([1 2 0 1 2], dtype=int32, device=xpux:0)
    """
    return _elwise(x, y, mode=Elemwise.Mode.MOD)


def abs(x):
    r"""Element-wise :math:`\operatorname{abs}(x)` function.
    
    Calculates the absolute value for each element :math:`x_i` of the input tensor :math:`x`.
    (i.e., the element-wise result has the same magnitude as the respective element in x but has positive sign).

    Args:
        x: input tensor. Should have a numeric data type.

    Returns:

        a tensor containing the absolute value of each element in :math:`x`.
        The returned tensor must have the same data type as :math:`x`.
    
    .. admonition:: Special cases

       For floating-point operands,

       * If :math:`x_i` is ``NaN``, the result is ``NaN``.
       * If :math:`x_i` is ``-0``, the result is ``+0``.
       * If :math:`x_i` is ``-infinity``, the result is ``+infinity``.

    Examples:
        >>> F.abs(-2)
        Tensor(2, dtype=int32, device=xpux:0)

        Element-wise absolution:

        >>> x = Tensor([1, -2, 3, -4, 5])
        >>> F.abs(x)
        Tensor([1 2 3 4 5], dtype=int32, device=xpux:0)  
    """
    return _elwise(x, mode=Elemwise.Mode.ABS)


def exp(x):
    r"""Element-wise :math:`e^x` function.
    
    Calculates an approximation to the exponential function for each element :math:`x_i` of the input tensor :math:`x` 
    (:math:`e` raised to the power of :math:`x_i`, where :math:`e` is the base of the natural logarithm).

    This function has domain ``[-infinity, +infinity]`` and codomain ``[+0, +infinity]``.
    
    Args:
        x: input tensor. Should have a floating-point data type.

    Returns:
        a tensor containing the evaluated exponential function result for each element in :math:`x`.
        The returned tensor must have a floating-point data type determined by :ref:`dtype-promotion`.

    .. admonition:: Special cases

       For floating-point operands,

       * If :math:`x_i` is ``NaN``, the result is ``NaN``.
       * If :math:`x_i` is ``+0``, the result is ``1``.
       * If :math:`x_i` is ``-0``, the result is ``1``.
       * If :math:`x_i` is ``+infinity``, the result is ``+infinity``.
       * If :math:`x_i` is ``-infinity``, the result is ``+0``.

    Examples:

       >>> F.exp([0, F.log(2)])
       Tensor([1. 2.], device=xpux:0)
    
    """
    return _elwise(x, mode=Elemwise.Mode.EXP)


def expm1(x):
    r"""Element-wise :math:`e^x - 1` function.
    
    Calculate  the exponential of the elements minus 1 of input.

    This function has domain ``[-infinity, +infinity]`` and codomain ``[-1, +infinity]``.

    .. math::

       y_i = e^{x_i} - 1

    Args:
        x: input tensor. Should have a floating-point data type.

    Returns:
        a tensor containing the evaluated result for each element in :math:`x`.
        The returned tensor must have a floating-point data type determined by :ref:`dtype-promotion`.
    
    .. note::

       This function provides greater precision than :math:`\exp(x) - 1` for small values of x.
       See FDLIBM, or some other IEEE 754-2019 compliant mathematical library, for a potential reference implementation.

    Examples:

       >>> F.expm1(1e-10)
       Tensor(1e-10, device=xpux:0)
    
    """
    return _elwise(x, mode=Elemwise.Mode.EXPM1)


def log(x):
    r"""Element-wise :math:`\operatorname{log}(x)` function.
    
    Calculates an approximation to the natural (base :math:`e`) logarithm for each element :math:`x_i` of the input tensor :math:`x`.

    This function has domain ``[+0, +infinity]`` and codomain ``[-infinity, +infinity]``.
    
    Args:
        x: input tensor. Should have a floating-point data type.

    Returns:
        a tensor containing the evaluated natural logarithm result for each element in :math:`x`.
        The returned tensor must have a floating-point data type determined by :ref:`dtype-promotion`.
    
    .. admonition:: Special cases

       For floating-point operands,

       * If :math:`x_i` is ``NaN``, the result is ``NaN``.
       * If :math:`x_i` is less than ``0``, the result is ``NaN``.
       * If :math:`x_i` is either ``+0`` or ``-0``, the result is ``-infinity``.
       * If :math:`x_i` is ``1``, the result is ``+0``.
       * If :math:`x_i` is ``+infinity``, the result is ``+infinity``.

    Examples:

        >>> F.log([1, F.exp(1)])
        Tensor([0. 1.], device=xpux:0)
    
    """
    return _elwise(x, mode=Elemwise.Mode.LOG)


def log1p(x):
    r"""Element-wise :math:`\log(1 + x)` function.
    
    Calculates an approximation to :math:`\log(1 + x)`:

    .. math::

         y_i = \log(1 + x_i)

    where log refers to the natural (base :math:`e`) logarithm, 
    for each element :math:`x_i` of the input tensor :math:`x`.

    This function has domain ``[-1, +infinity]`` and codomain ``[-infinity, +infinity]``.

    Args:
        x: input tensor. Should have a floating-point data type.

    Returns:

        a tensor containing the evaluated result for each element in :math:`x`.
        The returned tensor must have a floating-point data type determined by :ref:`dtype-promotion`.
    
    .. note::

       This function is more accurate than :math:`\log(1+x)` for small values of input.
       See FDLIBM, or some other IEEE 754-2019 compliant mathematical library, for a potential reference implementation.
       

    .. admonition:: Special cases

       For floating-point operands,

       * If :math:`x_i` is ``NaN``, the result is ``NaN``.
       * If :math:`x_i` is less than ``-1``, the result is ``NaN``.
       * If :math:`x_i` is ``-1``, the result is ``-infinity``.
       * If :math:`x_i` is ``-0``, the result is ``-0``.
       * If :math:`x_i` is ``+0``, the result is ``+0``.
       * If :math:`x_i` is ``+infinity``, the result is ``+infinity``.

    Examples:

        >>> F.log(1e-10 + 1)
        Tensor(0.0, device=xpux:0)
        >>> F.log1p(1e-10)
        Tensor(1e-10, device=xpux:0)
    
    """
    return _elwise(x, mode=Elemwise.Mode.LOG1P)


def sqrt(x: Tensor) -> Tensor:
    r"""Element-wise :math:`\operatorname{sqrt}(x)` function.
    
    Calculates the square root for each element :math:`x_i` of the input tensor :math:`x`.
    After rounding, each result must be indistinguishable from the infinitely precise result (as required by IEEE 754).

    This function has domain ``[0, +infinity]`` and codomain ``[0, +infinity]``.
    
    Args:
        x: input tensor. Should have a floating-point data type.

    Returns:
        a tensor containing the evaluated square root result for each element in :math:`x`.
        The returned tensor must have a floating-point data type determined by :ref:`dtype-promotion`.
    
    .. admonition:: Special cases
    
       For floating-point operands,

       * If :math:`x_i` is ``NaN``, the result is ``NaN``.
       * If :math:`x_i` is less than ``0``, the result is ``NaN``.
       * If :math:`x_i` is ``+0``, the result is ``+0``.
       * If :math:`x_i` is ``-0``, the result is ``-0``.
       * If :math:`x_i` is ``+infinity``, the result is ``+infinity``.

    Examples:
        >>> F.sqrt(4)
        Tensor(2.0, device=xpux:0)

        Element-wise square root:

        >>> x = Tensor([1, 4, 9, 16])
        >>> F.sqrt(x)
        Tensor([1. 2. 3. 4.], device=xpux:0)

    """
    return _elwise(x, mode=Elemwise.Mode.SQRT)


def square(x: Tensor) -> Tensor:
    r"""Element-wise :math:`x^2` function.
    
    Calculates the square for each element :math:`x_i` of the input tensor :math:`x`.
    
    Args:
        x: input tensor. Should have a floating-point data type.

    Returns:
        a tensor containing the evaluated square root result for each element in :math:`x`.
        The returned tensor must have a floating-point data type determined by :ref:`dtype-promotion`.

    Examples:
        >>> F.square(2)
        Tensor(4, dtype=int32, device=xpux:0)

        Element-wise square:

        >>> x = Tensor([1, -2, -3, 4])
        >>> F.square(x)
        Tensor([ 1  4  9 16], dtype=int32, device=xpux:0)

    """
    return _elwise(x, mode=Elemwise.Mode.SQUARE)


def logaddexp(x: Tensor, y: Tensor) -> Tensor:
    r"""Element-wise :math:`\log(e^x + e^y)` function.

    This function is useful in statistics where the calculated probabilities of events may be so small
    as to exceed the range of normal floating point numbers. 
    In such cases the logarithm of the calculated probability is stored.
    This function allows adding probabilities stored in such a fashion.

    Args:
        x: input tensor. Should have a floating-point data type.
        y: input tensor. Must be compatible with :math:`x`` (see :ref:`broadcasting-rule` ). Should have a floating-point data type.

    Returns:
        a tensor containing the evaluated result for each element in :math:`x` and :math:`y`.
        The returned tensor must have a floating-point data type determined by :ref:`dtype-promotion`.

    Examples:
        >>> prob1 = F.log(1e-10)
        >>> prob2 = F.log(2e-10)
        >>> F.logaddexp(prob1, prob2)
        Tensor(-21.927238, device=xpux:0)
    """
    return _elwise(x, y, mode=Elemwise.Mode.LOG_SUM_EXP)


def round(x):
    r"""Element-wise :math:`\operatorname{round}(x)` function.
    
    Rounds each element :math:`x_i` of the input tebsor :math:`x` to the nearest integer-valued number.
    
    Args:
        x: input tensor. Should have a numeric data type.

    Returns:
        a tensor containing the rounded result for each element in :math:`x`.
        The returned tensor must have the same data type as :math:`x`.

    .. admonition:: Special cases

       If :math:`x_i` is already integer-valued, the result is :math:`x_i`.
    
       For floating-point operands,

       * If :math:`x_i` is ``+infinity``, the result is ``+infinity``.
       * If :math:`x_i` is ``-infinity``, the result is ``-infinity``.
       * If :math:`x_i` is ``+0``, the result is ``+0``.
       * If :math:`x_i` is ``-0``, the result is ``-0``.
       * If :math:`x_i` is NaN, the result is NaN.
       * If two integers are equally close to :math:`x_i`, the result is the even integer closest to :math:`x_i`.

    Examples:
        >>> F.round(1.5)
        Tensor(2.0, device=xpux:0)

        Element-wise rounding:

        >>> x = Tensor([1.5, 2.5, 3.5, 4.5])
        >>> F.round(x)
        Tensor([2. 3. 4. 5.], device=xpux:0)
    """
    return _elwise(x, mode=Elemwise.Mode.ROUND)


def ceil(x):
    r"""Element-wise :math:`\lceil x \rceil` function.
    
    Rounds each element :math:`x_i` of the input tensor :math:`x` to the smalles integer-valued number that is not less than :math:`x_i`.

    Args:
        x: input tensor. Should have a numeric data type.

    Returns:
        a tensor containing the rounded result for each element in :math:`x`.
        The returned tensor must have the same data type as :math:`x`.
    
    .. admonition:: Special cases

       If :math:`x_i` is already integer-valued, the result is :math:`x_i`.
    
       For floating-point operands,

       * If :math:`x_i` is ``+infinity``, the result is ``+infinity``.
       * If :math:`x_i` is ``-infinity``, the result is ``-infinity``.
       * If :math:`x_i` is ``+0``, the result is ``+0``.
       * If :math:`x_i` is ``-0``, the result is ``-0``.
       * If :math:`x_i` is NaN, the result is NaN.
    
    Examples:
        >>> F.ceil(1.5)
        Tensor(2.0, device=xpux:0)

        Element-wise ceiling:

        >>> x = Tensor([1.5, 2.5, 3.5, 4.5])
        >>> F.ceil(x)
        Tensor([2. 3. 4. 5.], device=xpux:0)

    """
    return _elwise(x, mode=Elemwise.Mode.CEIL)


def floor(x):
    r"""Element-wise :math:`\lfloor x \rfloor` function.
    
    Rounds each element :math:`x_i` of the input tensor :math:`x` to the greatest integer-valued number that is not greater than :math:`x_i`.

    Args:  
        x: input tensor. Should have a numeric data type.

    Returns:
        a tensor containing the rounded result for each element in :math:`x`.
        The returned tensor must have the same data type as :math:`x`.

    .. admonition:: Special cases

       If :math:`x_i` is already integer-valued, the result is :math:`x_i`.
    
       For floating-point operands,

       * If :math:`x_i` is ``+infinity``, the result is ``+infinity``.
       * If :math:`x_i` is ``-infinity``, the result is ``-infinity``.
       * If :math:`x_i` is ``+0``, the result is ``+0``.
       * If :math:`x_i` is ``-0``, the result is ``-0``.
       * If :math:`x_i` is NaN, the result is NaN.
        
    Examples:
        >>> F.floor(1.5)
        Tensor(1.0, device=xpux:0)

        Element-wise flooring:

        >>> x = Tensor([1.5, 2.5, 3.5, 4.5])
        >>> F.floor(x)
        Tensor([1. 2. 3. 4.], device=xpux:0)
    """
    return _elwise(x, mode=Elemwise.Mode.FLOOR)


def maximum(x, y):
    r"""Element-wise maximum of tensor elements.
    
    Compare two tensors and returns a new tensor containing the element-wise maxima.
    If one of the elements being compared is a ``NaN``, then that element is returned.
    If both elements are ``NaNs`` then the first is returned.

    Args:

        x: input tensor. Should have a numeric data type.
        y: input tensor. Should have the same data type as :math:`x`.

    Returns:

        a tensor containing the element-wise maxima.
        The returned tensor must have the same data type as :math:`x`.

    Examples:
        >>> F.maximum(1, 2)
        Tensor(2, dtype=int32, device=xpux:0)

        Element-wise maximum:

        >>> x = Tensor([1, 2, 3, 4])
        >>> y = Tensor([4, 3, 2, 1])
        >>> F.maximum(x, y)
        Tensor([4 3 3 4], dtype=int32, device=xpux:0)

        Broadcasting:

        >>> x = Tensor([1, 2, 3, 4])
        >>> F.maximum(x, 2)
        Tensor([2 2 3 4], dtype=int32, device=xpux:0)
    
    """
    return _elwise(x, y, mode=Elemwise.Mode.MAX)


def minimum(x, y):
    r"""Element-wise minimum of tensor elements.
    
    Compare two tensors and returns a new tensor containing the element-wise minima.
    If one of the elements being compared is a ``NaN``, then that element is returned.
    If both elements are ``NaNs`` then the first is returned.

    Args:

        x: input tensor. Should have a numeric data type.
        y: input tensor. Should have the same data type as :math:`x`.

    Returns:

        a tensor containing the element-wise minima.
        The returned tensor must have the same data type as :math:`x`.

    Examples:
        >>> F.minimum(1, 2)
        Tensor(1, dtype=int32, device=xpux:0)

        Element-wise minimum:

        >>> x = Tensor([1, 2, 3, 4])
        >>> y = Tensor([4, 3, 2, 1])
        >>> F.minimum(x, y)
        Tensor([1 2 2 1], dtype=int32, device=xpux:0)

        Broadcasting:

        >>> x = Tensor([1, 2, 3, 4])
        >>> F.minimum(x, 2)
        Tensor([1 2 2 2], dtype=int32, device=xpux:0)

    """
    return _elwise(x, y, mode=Elemwise.Mode.MIN)


def clip(x: Tensor, lower=None, upper=None) -> Tensor:
    r"""Element-wise clipping function.
    
    Clamps(limits) all elements :math:`x_i` of the input tensor :math:`x` into the range ``[ lower, upper ]``.
    For example, if a range of ``[0, 1]`` is specified, 
    values smaller than ``0`` become ``0``, and values larger than ``1`` become ``1``.

    .. math::

        y_i = \begin{cases}
            \text{lower} & \text{if } x_i < \text{lower} \\
            x_i & \text{if } \text{lower} \leq x_i \leq \text{upper} \\
            \text{upper} & \text{if } x_i > \text{upper}
        \end{cases}

    Equivalent to ``F.minimum(upper, np.maximum(x, upper))`` right now.

    Args:
        x: The input tensor.
        lower: lower-bound of the range to be clamped to. Should have a numeric data type.
        upper: upper-bound of the range to be clamped to. Should have a numeric data type.

    Note:
        * If both ``lower`` and ``upper`` are None, raises an ``AssertionError``.
        * If ``lower`` is None, equivalent to ``F.minimum(x, upper)``.
        * If ``upper`` is None, equivalent to ``F.maximum(x, lower)``.
        * If ``lower`` is bigger than ```upper``, the result is same as ``clip(Tensor(), upper, upper)``.

    Returns:
        output clamped tensor. The result must have a data type determined by :ref:`dtype-promotion`.

    """
    assert (
        lower is not None or upper is not None
    ), "At least one of 'lower' or 'upper' must not be None"
    if lower is not None:
        if upper is not None:
            return _elwise(x, lower, upper, mode=Elemwise.Mode.CLIP)
        else:
            return maximum(x, lower)
    else:
        return minimum(x, upper)


# trigonometric functions


def cos(x):
    r"""Element-wise :math:`\cos(x)` function.
    
    Calculates an approximation to the cosine for each element :math:`x_i` of the input tensor :math:`x`.
    Each element :math:`x_i` is assumed to be expressed in radians.

    This function has domain ``(-infinity, +infinity)`` and codomain ``[-1, +1]``.

    Args:
        x: input tensor whose elements are each expressed in radians. Should have a floating-point data type.

    Returns:
        a tensor containing the cosine of each element in :math:`x`.
        The returned tensor must have a floating-point data type determined by :ref:`dtype-promotion`.

    Examples:
        >>> F.cos(0)
        Tensor(1.0, device=xpux:0)

        Element-wise cosine:

        >>> import math
        >>> x = Tensor([0, math.pi/2, math.pi])
        >>> F.cos(x)
        Tensor([ 1. -0. -1.], device=xpux:0)
    """
    return _elwise(x, mode=Elemwise.Mode.COS)


def sin(x):
    r"""Element-wise :math:`\sin(x)` function.
    
    Calculates an approximation to the sine for each element :math:`x_i` of the input tensor :math:`x`.
    Each element :math:`x_i` is assumed to be expressed in radians.

    This function has domain ``(-infinity, +infinity)`` and codomain ``[-1, +1]``.

    Args:
        x: input tensor whose elements are each expressed in radians. Should have a floating-point data type.

    Returns:
        a tensor containing the sine of each element in :math:`x`.
        The returned tensor must have a floating-point data type determined by :ref:`dtype-promotion`.

    Examples:
        >>> F.sin(0)
        Tensor(0.0, device=xpux:0)

        Element-wise sine:

        >>> import math
        >>> x = Tensor([0, math.pi/2, math.pi])
        >>> F.sin(x)
        Tensor([ 0.  1. -0.], device=xpux:0)
    """
    return _elwise(x, mode=Elemwise.Mode.SIN)


def tan(x):
    r"""Element-wise :math:`\tan(x)` function.

    Calculates an approximation to the tangent for each element :math:`x_i` of the input tensor :math:`x`.
    Each element :math:`x_i` is assumed to be expressed in radians.

    This function has domain ``(-infinity, +infinity)`` and codomain ``(-infinity, +infinity)``.

    Args:
        x: input tensor whose elements are each expressed in radians. Should have a floating-point data type.

    Returns:
        a tensor containing the tangent of each element in :math:`x`.
        The returned tensor must have a floating-point data type determined by :ref:`dtype-promotion`.

    .. admonition:: Special cases

       For floating-point operands,

       * If :math:`x_i` is ``NaN``, the result is ``NaN``.
       * If :math:`x_i` is ``+0``, the result is ``+0``.
       * If :math:`x_i` is ``-0``, the result is ``-0``.
       * If :math:`x_i` is either ``+infinity`` or ``-infinity``, the result is ``NaN``.

    Examples:
        >>> F.tan(0)
        Tensor(0.0, device=xpux:0)

        Element-wise tangent:

        >>> import math
        >>> x = Tensor([0, math.pi/4, math.pi])
        >>> F.tan(x)
        Tensor([0. 1. 0.], device=xpux:0)

    """
    return _elwise(x, mode=Elemwise.Mode.TAN)


def acos(x):
    r"""Element-wise :math:`\arccos(x)` function.

    Calculates an approximation to the inverse cosine for each element :math:`x_i` of the input tensor :math:`x`.
    Each element-wise result is expressed in radians.

    This function has domain ``[-1, +1]`` and codomain ``[0, pi]``.

    The inverse of :math:`\cos` so that, if :math:`y = \cos(x)`, then :math:`x = \arccos(y)`.

    Args:
        x: input tensor. Should have a floating-point data type.

    Returns:
        a tensor containing the inverse cosine of each element in :math:`x`.
        The returned tensor must have a floating-point data type determined by :ref:`dtype-promotion`.

    .. admonition:: Special cases

       For floating-point operands,

       * If :math:`x_i` is ``NaN``, the result is ``NaN``.
       * If :math:`x_i` is greater than ``1``, the result is ``NaN``.
       * If :math:`x_i` is less than ``-1``, the result is ``NaN``.
       * If :math:`x_i` is ``1``, the result is ``+0``.

    Examples:
        >>> F.acos(1)
        Tensor(0.0, device=xpux:0)

        Element-wise inverse cosine:

        >>> import math
        >>> x = Tensor([0, 1, -1])
        >>> F.acos(x)
        Tensor([1.5708 0.     3.1416], device=xpux:0)

    """
    return _elwise(x, mode=Elemwise.Mode.ACOS)


def asin(x):
    r"""Element-wise :math:`\arcsin(x)` function.

    Calculates an approximation to the inverse sine for each element :math:`x_i` of the input tensor :math:`x`.
    Each element-wise result is expressed in radians.

    This function has domain ``[-1, +1]`` and codomain ``[-pi/2, pi/2]``.

    The inverse of :math:`\sin` so that, if :math:`y = \sin(x)`, then :math:`x = \arcsin(y)`.

    Args:
        x: input tensor. Should have a floating-point data type.

    Returns:
        a tensor containing the inverse sine of each element in :math:`x`.
        The returned tensor must have a floating-point data type determined by :ref:`dtype-promotion`.

    .. admonition:: Special cases

       For floating-point operands,
       
       * If :math:`x_i` is ``NaN``, the result is ``NaN``.
       * If :math:`x_i` is greater than ``1``, the result is ``NaN``.
       * If :math:`x_i` is less than ``-1``, the result is ``NaN``.
       * If :math:`x_i` is ``+0``, the result is ``+0``.
       * If :math:`x_i` is ``-0``, the result is ``-0``.

    Examples:

        >>> F.asin(0)
        Tensor(0.0, device=xpux:0)

        Element-wise inverse sine:

        >>> x = Tensor([0, 1, -1])
        >>> F.asin(x)
        Tensor([ 0.      1.5708 -1.5708], device=xpux:0)
    """
    return _elwise(x, mode=Elemwise.Mode.ASIN)


def atan(x):
    r"""Element-wise :math:`\arctan(x)` function.

    Calculates an approximation to the inverse tangent for each element :math:`x_i` of the input tensor :math:`x`.
    Each element-wise result is expressed in radians.

    This function has domain ``(-infinity, +infinity)`` and codomain ``[-pi/2, pi/2]``.

    The inverse of :math:`\tan` so that, if :math:`y = \tan(x)`, then :math:`x = \arctan(y)`.

    Args:
        x: input tensor. Should have a floating-point data type.

    Returns:
        a tensor containing the inverse tangent of each element in :math:`x`.
        The returned tensor must have a floating-point data type determined by :ref:`dtype-promotion`.

    .. admonition:: Special cases

       For floating-point operands,

       * If :math:`x_i` is ``NaN``, the result is ``NaN``.
       * If :math:`x_i` is ``+0``, the result is ``+0``.
       * If :math:`x_i` is ``-0``, the result is ``-0``.
       * If :math:`x_i` is ``+infinity``, the result is an approximation to ``+π/2``.
       * If :math:`x_i` is ``-infinity``, the result is an approximation to ``-π/2``.

    Examples:
        >>> F.atan(0)
        Tensor(0.0, device=xpux:0)

        Element-wise inverse tangent:

        >>> x = Tensor([0, 1, -1])
        >>> F.atan(x)
        Tensor([ 0.      0.7854 -0.7854], device=xpux:0)

    """
    return _elwise(
        x,
        get_scalar_one("float32", x.device if isinstance(x, Tensor) else None),
        mode=Elemwise.Mode.ATAN2,
    )


def atan2(y, x):
    r"""Element-wise :math:`\arctan(\frac{y}{x})` function.

    Calculates an approximation to the inverse tangent for each element :math:`x_i` of the input tensor :math:`x`
    with the respective element :math:`y_i` of the input tensor :math:`y`.

    Args:
        y: first input tensor whose elements correspond to the dividend. Should have a numeric data type.
        x: second input tensor whose elements correspond to the divisor.
            Must be compatible with `x` (see :ref:`broadcasting-rule` ). Should have a numeric data type.

    Returns:
        a tensor containing the inverse tangent of each element in :math:`y/x`.

    .. admonition:: Special cases

       ``atan2`` is identical to the ``atan2`` function of the underlying C library.
       The following special values are defined in the C standard:  

       For floating-point operands,

       * if :math:`y` is `+/-0`` and :math:`x` is ``+0``, the result is ``+/-0``.
       * if :math:`y` is ``+/-0`` and :math:`x` is ``-0``, the result is ``+/-π``.
       * if :math:`y` is greater than ``0`` and :math:`x` is ``+/-infinity``, the result is ``+0/+π``.
       * if :math:`y` is less than ``0`` and :math:`x` is ``+/-infinity``, the result is ``-0/-π``.
       * if :math:`y` is ``+/-infinity`and :math:`x` is ``+infinity``, tge result is ``+/-(π/4)``.
       * if :math:`y` is ``+/-infinity`and :math:`x` is ``-infinity``, tge result is ``+/-(3π/4)``.

       Note that ``+0`` and ``-0`` are distinct floating point numbers, as are ``+inf`` and ``-inf``.

    Examples:
        >>> F.atan2(0, 1)  # equals to atan(0)
        Tensor(0.0, device=xpux:0)

        Element-wise inverse tangent:

        >>> y = Tensor([0, 1, -1])
        >>> x = Tensor([1, 1, 1])
        >>> F.atan2(y, x)
        Tensor([ 0.      0.7854 -0.7854], device=xpux:0)

    """
    return _elwise(y, x, mode=Elemwise.Mode.ATAN2)


def cosh(x):
    r"""Element-wise :math:`\cosh(x)` function.

    Calculates the hyperbolic cosine for each element :math:`x_i` of the input tensor :math:`x`.

    Equivalent to:

    .. math:: 

       \frac {e^{x}+e^{-x}} {2}

    This function has domain ``[-infinity, +infinity]`` and codomain ``[-infinity, +infinity]``.

    Args:
        x: input tensor whose elements each represent a hyperbolic angle. Should have a floating-point data type.
    
    Returns:
        a tensor containing the hyperbolic cosine of each element in :math:`x`.
        The returned tensor must have a floating-point data type determined by :ref:`dtype-promotion`.

    .. admonition:: Special cases

       For floating-point operands,

       * If :math:`x_i` is ``NaN``, the result is ``NaN``.
       * If :math:`x_i` is ``+0``, the result is ``1``.
       * If :math:`x_i` is ``-0``, the result is ``1``.
       * If :math:`x_i` is ``+infinity``, the result is ``+infinity``.
       * If :math:`x_i` is ``-infinity``, the result is ``+infinity``.

    Examples:
        >>> F.cosh(0)
        Tensor(1.0, device=xpux:0)

        Element-wise hyperbolic cosine:

        >>> x = Tensor([0, 1, -1])
        >>> F.cosh(x)
        Tensor([1.     1.5431 1.5431], device=xpux:0)
    
    """
    return _elwise(x, mode=Elemwise.Mode.COSH)


def sinh(x):
    r"""Element-wise :math:`\sinh(x)` function.

    Calculates the hyperbolic sine for each element :math:`x_i` of the input tensor :math:`x`.

    Equivalent to:

    .. math::

       \frac {e^{x}-e^{-x}} {2}

    This function has domain ``[-infinity, +infinity]`` and codomain ``[-infinity, +infinity]``.

    Args:
        x: input tensor whose elements each represent a hyperbolic angle. Should have a floating-point data type.

    Returns:
        a tensor containing the hyperbolic sine of each element in :math:`x`.

    .. admonition:: Special cases

       For floating-point operands,

       * If :math:`x_i` is ``NaN``, the result is ``NaN``.
       * If :math:`x_i` is ``+0``, the result is ``+0``.
       * If :math:`x_i` is ``-0``, the result is ``-0``.
       * If :math:`x_i` is ``+infinity``, the result is ``+infinity``.
       * If :math:`x_i` is ``-infinity``, the result is ``+infinity``.

    Examples:
        >>> F.sinh(0)
        Tensor(0.0, device=xpux:0)

        Element-wise hyperbolic sine:

        >>> x = Tensor([0, 1, -1])
        >>> F.sinh(x)
        Tensor([ 0.      1.1752 -1.1752], device=xpux:0)
    
    """
    return _elwise(x, mode=Elemwise.Mode.SINH)


def tanh(x):
    r"""Element-wise :math:`\tanh(x)` function.

    Calculates the hyperbolic tangent for each element :math:`x_i` of the input tensor :math:`x`.

    Equivalent to:

    .. math::

       \frac {\sinh(x)} {\cosh(x)} =  \frac {e^{x}-e^{-x}} {e^{x}+e^{-x}}

    This function has domain ``[-infinity, +infinity]`` and codomain ``[-1, 1]``.

    Args:
        x: input tensor whose elements each represent a hyperbolic angle. Should have a floating-point data type.
    
    Returns:
        a tensor containing the hyperbolic tangent of each element in :math:`x`.
        The returned tensor must have a floating-point data type determined by :ref:`dtype-promotion`.

    .. admonition:: Special cases

       For floating-point operands,

       * If :math:`x_i` is ``NaN``, the result is ``NaN``.
       * If :math:`x_i` is ``+0``, the result is ``+0``.
       * If :math:`x_i` is ``-0``, the result is ``-0``.
       * If :math:`x_i` is ``+infinity``, the result is ``+1``.
       * If :math:`x_i` is ``-infinity``, the result is ``+1``.

    Examples:
        >>> F.tanh(0)
        Tensor(0.0, device=xpux:0)

        Element-wise hyperbolic tangent:

        >>> x = Tensor([0, 1, -1])
        >>> F.tanh(x)
        Tensor([ 0.      0.7616 -0.7616], device=xpux:0)
    """
    return _elwise(x, mode=Elemwise.Mode.TANH)


def asinh(x):
    r"""Element-wise :math:`\sinh^{-1}(x)` function.

    Calculates the inverse hyperbolic sine for each element :math:`x_i` of the input tensor :math:`x`.

    This function has domain ``[-infinity, +infinity]`` and codomain ``[-infinity, +infinity]``.

    .. admonition:: Special cases

       For floating-point operands,

       * If :math:`x_i` is ``NaN``, the result is ``NaN``.
       * If :math:`x_i` is ``+0``, the result is ``+0``.
       * If :math:`x_i` is ``-0``, the result is ``-0``.
       * If :math:`x_i` is ``+infinity``, the result is ``+infinity``.
       * If :math:`x_i` is ``-infinity``, the result is ``+infinity``.

    Args:
        x: input tensor whose elements each represent the area of a hyperbolic sector. Should have a floating-point data type.

    Returns:
        a tensor containing the inverse hyperbolic sine of each element in :math:`x`.
        The returned tensor must have a floating-point data type determined by :ref:`dtype-promotion`.

    Examples:
        >>> F.asinh(0)
        Tensor(0.0, device=xpux:0)

        Element-wise inverse hyperbolic sine:

        >>> x = Tensor([0, 1, -1])
        >>> F.asinh(x)
        Tensor([ 0.      0.8814 -0.8814], device=xpux:0)

    """
    return _elwise(x, mode=Elemwise.Mode.ASINH)


def acosh(x):
    r"""Element-wise :math:`\cosh^{-1}(x)` function.

    Calculates the inverse hyperbolic cosine for each element :math:`x_i` of the input tensor :math:`x`.

    This function has domain ``[1, +infinity]`` and codomain ``[0, +infinity]``.

    .. admonition:: Special cases

       For floating-point operands,

       * If :math:`x_i` is ``NaN``, the result is ``NaN``.
       * If :math:`x_i` is less than ``1``, the result is ``NaN``.
       * If :math:`x_i` is ``1``, the result is ``+0``.
       * If :math:`x_i` is ``+infinity``, the result is ``+infinity``.

    Args:
        x: input tensor whose elements each represent the area of a hyperbolic sector. Should have a floating-point data type.

    Returns:
        a tensor containing the inverse hyperbolic cosine of each element in :math:`x`.
        The returned tensor must have a floating-point data type determined by :ref:`dtype-promotion`.

    Examples:
        >>> F.acosh(1)
        Tensor(0.0, device=xpux:0)

        Element-wise inverse hyperbolic cosine:

        >>> x = Tensor([1, 2, 3])
        >>> F.acosh(x)
        Tensor([0.     1.317  1.7627], device=xpux:0)
    """
    return _elwise(x, mode=Elemwise.Mode.ACOSH)


def atanh(x):
    r"""Element-wise :math:`\tanh^{-1}(x)` function.

    Calculates the inverse hyperbolic tangent for each element :math:`x_i` of the input tensor :math:`x`.

    This function has domain ``[-1, +1]`` and codomain ``[-infinity, +infinity]``.

    .. admonition:: Special cases

       * If :math:`x_i` is ``NaN``, the result is ``NaN``.
       * If :math:`x_i` is less than ``-1``, the result is ``NaN``.
       * If :math:`x_i` is greater than ``1``, the result is ``Nan``.
       * If :math:`x_i` is ``+1``, the result is ``+infinity``.
       * If :math:`x_i` is ``-1``, the result is ``-infinity``.
       * If :math:`x_i` is ``+0``, the result is ``+0``.
       * If :math:`x_i` is ``-0``, the result is ``-0``.

    Args:
        x: input tensor whose elements each represent the area of a hyperbolic sector. Should have a floating-point data type.

    Returns:
        a tensor containing the inverse hyperbolic tangent of each element in :math:`x`.
        The returned tensor must have a floating-point data type determined by :ref:`dtype-promotion`.

    Examples:
        >>> F.atanh(0)
        Tensor(0.0, device=xpux:0)

        Element-wise inverse hyperbolic tangent:

        >>> x = Tensor([0, 0.5, -0.5])
        >>> F.atanh(x)
        Tensor([ 0.      0.5493 -0.5493], device=xpux:0)

    """
    return _elwise(x, mode=Elemwise.Mode.ATANH)


# bit-twiddling functions


def left_shift(x, y):
    r"""Element-wise left shift.

    Shifts the bits of each element :math:`x_i` of the input tensor :math:`x` to the left by appending :math:`y_i`
    (i.e., the respective element in the input tesnor :math:`y`) zeros to the right of :math:`x_i`.
    
    .. note::

       The ``<<`` operator can be used as a shorthand for ``left_shift`` on Tensors.

    Args:
        x: first input tensor. Should have an integer data type.
        y: second input tensor. Must be compatible with :math:`x` (see :ref:`broadcasting-rule` ).
            Should have an integer data type. Each element must be greater than or equal to ``0``.

    Returns:
        a tensor containing the result of the element-wise left shift operation.
        The returned tensor must have the a data type determined by :ref:`dtype-promotion`.

    Examples:
        >>> F.left_shift([1, 2, 3], 1)
        Tensor([2 4 6], dtype=int32, device=xpux:0)

        Element-wise left shift:

        >>> x = Tensor([1, 2, 3])
        >>> y = Tensor([1, 2, 3])
        >>> F.left_shift(x, y)
        Tensor([ 2  8 24], dtype=int32, device=xpux:0)

        Broadcasting:

        >>> F.left_shift(5, [1, 2, 3])
        Tensor([10 20 40], dtype=int32, device=xpux:0)
    """
    return _elwise(x, y, mode=Elemwise.Mode.SHL)


def right_shift(x, y):
    r"""Element-wise right shift.

    Shifts the bits of each element :math:`x_i` of the input tensor :math:`x` to the right
    according to the respective element :math:`y_i` of the input tensor :math:`y`.

    .. note::

       The ``>>`` operator can be used as a shorthand for ``right_shift`` on Tensors.

    .. note::

       This operation must be an arithmetic shift (i.e., sign-propagating)
       and thus equivalent to floor division by a power of two.

    Args:
        x: first input tensor. Should have an integer data type.
        y: second input tensor. Must be compatible with :math:`x` (see :ref:`broadcasting-rule` ).
            Should have an integer data type. Each element must be greater than or equal to ``0``.

    Returns:
        a tensor containing the result of the element-wise right shift operation.
        The returned tensor must have the a data type determined by :ref:`dtype-promotion`.
    
    Examples:
        >>> F.right_shift([2, 4, 8], 1)
        Tensor([1 2 4], dtype=int32, device=xpux:0)

        Element-wise left shift:

        >>> x = Tensor([2, 8, 24])
        >>> y = Tensor([1, 2, 3])
        >>> F.right_shift(x, y)
        Tensor([1 2 3], dtype=int32, device=xpux:0)

        Broadcasting:

        >>> F.right_shift([10, 20, 40], 2)
        Tensor([ 2  5 10], dtype=int32, device=xpux:0)
    """
    return _elwise(x, y, mode=Elemwise.Mode.SHR)


# logical functions


def logical_and(x, y):
    r"""Element-wise logical AND.

    Computes the logical AND for each element :math:`x_i` of the input tensor :math:`x`
    with the respective element :math:`y_i` of the input tensor :math:`y`.

    Args:
        x: first input tensor. Should have a boolean data type.
        y: second input tensor. Must be compatible with :math:`x` (see :ref:`broadcasting-rule` ).
            Should have a boolean data type.

    Returns:
        a tensor containing the result of the element-wise logical AND operation.
        The returned tensor must have a data type of ``bool``.

    .. seealso:: 

       :func:`~.logical_or`, :func:`~.logical_not`, :func:`~.logical_xor`

    Examples:
        >>> F.logical_and(True, False)
        Tensor(False, dtype=bool, device=xpux:0)

        Element-wise logical AND:

        >>> x = Tensor([True, False, True])
        >>> y = Tensor([False, False, True])
        >>> F.logical_and(x, y)
        Tensor([False False  True], dtype=bool, device=xpux:0)

        The ``&`` operator can be used as a shorthand for ``F.logical_and`` on boolean tensors.

        >>> x & y
        Tensor([False False  True], dtype=bool, device=xpux:0)
    """
    return _elwise(x, y, mode=Elemwise.Mode.AND)


def logical_not(x):
    r"""Element-wise logical NOT.

    Computes the logical NOT for each element :math:`x_i` of the input tensor :math:`x`.

    Args:
        x: input tensor. Should have a boolean data type.

    Returns:
        a tensor containing the result of the element-wise logical NOT operation.
        The returned tensor must have a data type of ``bool``.

    .. seealso:: 

       :func:`~.logical_and`, :func:`~.logical_or`, :func:`~.logical_xor`

    Examples:
        >>> F.logical_not(True)
        Tensor(False, dtype=bool, device=xpux:0)

        Element-wise logical NOT:

        >>> x = Tensor([True, False, True])
        >>> F.logical_not(x)
        Tensor([False  True False], dtype=bool, device=xpux:0)
    
        The ``~`` operator can be used as a shorthand for ``F.logical_and`` on boolean tensors.

        >>> ~x
        Tensor([False  True False], dtype=bool, device=xpux:0)
    """
    return _elwise(x, mode=Elemwise.Mode.NOT)


def logical_or(x, y):
    r"""Element-wise logical OR.

    Computes the logical OR for each element :math:`x_i` of the input tensor :math:`x`
    with the respective element :math:`y_i` of the input tensor :math:`y`.

    Args:
        x: first input tensor. Should have a boolean data type.

    Returns:
        a tensor containing the result of the element-wise logical OR operation.
        The returned tensor must have a data type of ``bool``.

    .. seealso::

       :func:`~.logical_and`, :func:`~.logical_not`, :func:`~.logical_xor`

    Examples:
        >>> F.logical_or(True, False)
        Tensor(True, dtype=bool, device=xpux:0)

        Element-wise logical OR:

        >>> x = Tensor([True, False, True])
        >>> y = Tensor([False, False, True])
        >>> F.logical_or(x, y)
        Tensor([ True False  True], dtype=bool, device=xpux:0)

        The ``|`` operator can be used as a shorthand for ``F.logical_or`` on boolean tensors.

        >>> x | y
        Tensor([ True False  True], dtype=bool, device=xpux:0)
    """
    return _elwise(x, y, mode=Elemwise.Mode.OR)


def logical_xor(x, y):
    r"""Element-wise logical XOR.

    Computes the logical XOR for each element :math:`x_i` of the input tensor :math:`x`
    with the respective element :math:`y_i` of the input tensor :math:`y`.

    Args:
        x: first input tensor. Should have a boolean data type.
        y: second input tensor. Must be compatible with :math:`x` (see :ref:`broadcasting-rule` ).

    Returns:
        a tensor containing the result of the element-wise logical XOR operation.
        The returned tensor must have a data type of ``bool``.

    .. seealso::

       :func:`~.logical_and`, :func:`~.logical_not`, :func:`~.logical_or`

    Examples:

        >>> F.logical_xor(True, False)
        Tensor(True, dtype=bool, device=xpux:0)

        Element-wise logical XOR:

        >>> x = Tensor([True, False, True])
        >>> y = Tensor([False, False, True])
        >>> F.logical_xor(x, y)
        Tensor([ True False False], dtype=bool, device=xpux:0)

        The ``^`` operator can be used as a shorthand for ``F.logical_xor`` on boolean tensors.

        >>> x ^ y
        Tensor([ True False False], dtype=bool, device=xpux:0)

    """
    return _elwise(x, y, mode=Elemwise.Mode.XOR)


# comparison functions


def equal(x, y):
    r"""Element-wise equality comparison.

    Computes the truth value of :math:`x_i == y_i` for each element :math:`x_i` of the input tensor :math:`x`
    with the respective element :math:`y_i` of the input tensor :math:`y`.

    Args:
        x: first input tensor. May have any data type.
        y: second input tensor. Must be compatible with :math:`x` (see :ref:`broadcasting-rule` ). May have any data type.

    Returns:
        a tensor containing the result of the element-wise results.
        The returned tensor must have a data type of ``bool``.

    .. seealso:: 

       :func:`~.not_equal`, :func:`~.greater_equal`, :func:`~.less_equal`, :func:`~.greater`, :func:`~.less`

    Examples:

        Element-wise equality comparison:

        >>> x = Tensor([1, 2, 3])
        >>> y = Tensor([1, 2, 4])
        >>> F.equal(x, y)
        Tensor([ True  True False], dtype=bool, device=xpux:0)

        The ``==`` operator can be used as a shorthand for ``F.equal`` on boolean tensors.

        >>> x == y
        Tensor([ True  True False], dtype=bool, device=xpux:0)
    """
    return x == y


def not_equal(x, y):
    r"""Element-wise inequality comparison.

    Computes the truth value of :math:`x_i != y_i` for each element :math:`x_i` of the input tensor :math:`x`
    with the respective element :math:`y_i` of the input tensor :math:`y`.

    Args:
        x: first input tensor. May have any data type.
        y: second input tensor. Must be compatible with :math:`x` (see :ref:`broadcasting-rule` ). May have any data type.

    Returns:
        a tensor containing the result of the element-wise results.
        The returned tensor must have a data type of ``bool``.

    .. seealso::

       :func:`~.equal`, :func:`~.greater_equal`, :func:`~.less_equal`, :func:`~.greater`, :func:`~.less`

    Examples:

        Element-wise inequality comparison:

        >>> x = Tensor([1, 2, 3])
        >>> y = Tensor([1, 2, 4])
        >>> F.not_equal(x, y)
        Tensor([False False  True], dtype=bool, device=xpux:0)

        The ``!=`` operator can be used as a shorthand for ``F.not_equal`` on boolean tensors.

        >>> x != y
        Tensor([False False  True], dtype=bool, device=xpux:0)
    """
    return x != y


def less(x, y):
    r"""Element-wise less-than comparison.

    Computes the truth value of :math:`x_i < y_i` for each element :math:`x_i` of the input tensor :math:`x`
    with the respective element :math:`y_i` of the input tensor :math:`y`.

    Args:
        x: first input tensor. May have any data type.
        y: second input tensor. Must be compatible with :math:`x` (see :ref:`broadcasting-rule` ). May have any data type.

    Returns:
        a tensor containing the result of the element-wise results.
        The returned tensor must have a data type of ``bool``.

    .. seealso::

       :func:`~.equal`, :func:`~.not_equal`, :func:`~.greater_equal`, :func:`~.less_equal`, :func:`~.greater`

    Examples:

        Element-wise less-than comparison:

        >>> x = Tensor([1, 2, 3])
        >>> y = Tensor([1, 2, 4])
        >>> F.less(x, y)
        Tensor([False False  True], dtype=bool, device=xpux:0)

        The ``<`` operator can be used as a shorthand for ``F.less`` on boolean tensors.

        >>> x < y
        Tensor([False False  True], dtype=bool, device=xpux:0)
    
    """
    return x < y


def less_equal(x, y):
    r"""Element-wise less-than-or-equal-to comparison.

    Computes the truth value of :math:`x_i <= y_i` for each element :math:`x_i` of the input tensor :math:`x`
    with the respective element :math:`y_i` of the input tensor :math:`y`.

    Args:
        x: first input tensor. May have any data type.
        y: second input tensor. Must be compatible with :math:`x` (see :ref:`broadcasting-rule` ). May have any data type.

    Returns:
        a tensor containing the result of the element-wise results.
        The returned tensor must have a data type of ``bool``.

    .. seealso::

       :func:`~.equal`, :func:`~.not_equal`, :func:`~.greater_equal`, :func:`~.less`, :func:`~.greater`

    Examples:

        Element-wise less-than-or-equal-to comparison:

        >>> x = Tensor([1, 2, 3])
        >>> y = Tensor([1, 2, 4])
        >>> F.less_equal(x, y)
        Tensor([ True  True  True], dtype=bool, device=xpux:0)

        The ``<=`` operator can be used as a shorthand for ``F.less_equal`` on boolean tensors.

        >>> x <= y
        Tensor([ True  True  True], dtype=bool, device=xpux:0)

    """
    return x <= y


def greater(x, y):
    r"""Element-wise greater-than comparison.

    Computes the truth value of :math:`x_i > y_i` for each element :math:`x_i` of the input tensor :math:`x`
    with the respective element :math:`y_i` of the input tensor :math:`y`.

    Args:
        x: first input tensor. May have any data type.
        y: second input tensor. Must be compatible with :math:`x` (see :ref:`broadcasting-rule` ). May have any data type.

    Returns:
        a tensor containing the result of the element-wise results.

    .. seealso::

       :func:`~.equal`, :func:`~.not_equal`, :func:`~.greater_equal`, :func:`~.less_equal`, :func:`~.less`

    Examples:

        Element-wise greater-than comparison:

        >>> x = Tensor([1, 2, 3])
        >>> y = Tensor([1, 2, 4])
        >>> F.greater(x, y)
        Tensor([False False False], dtype=bool, device=xpux:0)

        The ``>`` operator can be used as a shorthand for ``F.greater`` on boolean tensors.

        >>> x > y
        Tensor([False False False], dtype=bool, device=xpux:0)

    """
    return x > y


def greater_equal(x, y):
    r"""Element-wise greater-than-or-equal-to comparison.

    Computes the truth value of :math:`x_i >= y_i` for each element :math:`x_i` of the input tensor :math:`x`
    with the respective element :math:`y_i` of the input tensor :math:`y`.

    Args:
        x: first input tensor. May have any data type.
        y: second input tensor. Must be compatible with :math:`x` (see :ref:`broadcasting-rule` ). May have any data type.

    Returns:
        a tensor containing the result of the element-wise results.

    .. seealso::

       :func:`~.equal`, :func:`~.not_equal`, :func:`~.greater`, :func:`~.less_equal`, :func:`~.less`

    Examples:

        Element-wise greater-than-or-equal-to comparison:

        >>> x = Tensor([1, 2, 3])
        >>> y = Tensor([1, 2, 4])
        >>> F.greater_equal(x, y)
        Tensor([ True  True False], dtype=bool, device=xpux:0)

        The ``>=`` operator can be used as a shorthand for ``F.greater_equal`` on boolean tensors.

        >>> x >= y
        Tensor([ True  True False], dtype=bool, device=xpux:0)

    """
    return x >= y


sigmoid = deprecated_func("1.3", "megengine.functional.nn", "sigmoid", True)
hsigmoid = deprecated_func("1.3", "megengine.functional.nn", "hsigmoid", True)
relu = deprecated_func("1.3", "megengine.functional.nn", "relu", True)
relu6 = deprecated_func("1.3", "megengine.functional.nn", "relu6", True)
hswish = deprecated_func("1.3", "megengine.functional.nn", "hswish", True)
