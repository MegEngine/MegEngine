# -*- coding: utf-8 -*-
from ..core._imperative_rt.core2 import apply
from ..core._imperative_rt.core2 import sync as _sync
from ..core.ops.builtin import AssertEqual
from ..tensor import Tensor
from ..utils.deprecation import deprecated_func
from .elemwise import abs, maximum, minimum
from .tensor import ones, zeros

__all__ = ["topk_accuracy"]


def _assert_equal(
    expect: Tensor, actual: Tensor, *, maxerr: float = 0.0001, verbose: bool = False
):
    r"""Asserts two tensors equal and returns expected value (first input).
    It is a variant of python assert which is symbolically traceable (similar to ``numpy.testing.assert_equal``).
    If we want to verify the correctness of model, just ``assert`` its states and outputs.
    While sometimes we need to verify the correctness at different backends for *dumped* model
    (or in :class:`~jit.trace` context), and no python code could be executed in that case.
    Thus we have to use :func:`~functional.utils._assert_equal` instead.

    Args:
        expect: expected tensor value
        actual: tensor to check value
        maxerr: max allowed error; error is defined as the minimal of absolute and relative error
        verbose: whether to print maxerr to stdout during opr exec

    Examples:

        >>> x = Tensor([1, 2, 3], dtype="float32")
        >>> y = Tensor([1, 2, 3], dtype="float32")
        >>> F.utils._assert_equal(x, y, maxerr=0)
        Tensor([1. 2. 3.], device=xpux:0)

    """
    err = (
        abs(expect - actual)
        / maximum(
            minimum(abs(expect), abs(actual)),
            Tensor(1.0, dtype="float32", device=expect.device),
        )
    ).max()
    result = apply(AssertEqual(maxerr=maxerr, verbose=verbose), expect, actual, err)[0]
    _sync()  # sync interpreter to get exception
    return result


def _simulate_error():
    x1 = zeros(100)
    x2 = ones(100)
    (ret,) = apply(AssertEqual(maxerr=0, verbose=False), x1, x2, x2)
    return ret


topk_accuracy = deprecated_func(
    "1.3", "megengine.functional.metric", "topk_accuracy", True
)
copy = deprecated_func("1.3", "megengine.functional.tensor", "copy", True)
