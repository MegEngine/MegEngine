# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ..core._imperative_rt.core2 import apply
from ..core._imperative_rt.core2 import sync as _sync
from ..core.ops.builtin import AssertEqual
from ..tensor import Tensor
from ..utils.deprecation import deprecated_func
from .elemwise import abs, maximum, minimum

__all__ = ["topk_accuracy"]


def _assert_equal(
    expect: Tensor, actual: Tensor, *, maxerr: float = 0.0001, verbose: bool = False
):
    r"""
    Asserts two tensors equal and returns expected value (first input).
    It is a variant of python assert which is symbolically traceable (similar to ``numpy.testing.assert_equal``).
    If we want to verify the correctness of model, just ``assert`` its states and outputs.
    While sometimes we need to verify the correctness at different backends for *dumped* model
    (or in :class:`~jit.trace` context), and no python code could be executed in that case.
    Thus we have to use :func:`~functional.utils._assert_equal` instead.

    :param expect: expected tensor value
    :param actual: tensor to check value
    :param maxerr: max allowed error; error is defined as the minimal of absolute and relative error
    :param verbose: whether to print maxerr to stdout during opr exec
    :return: expected tensor

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = tensor([1, 2, 3], np.float32)
        y = tensor([1, 2, 3], np.float32)
        print(F.utils._assert_equal(x, y, maxerr=0).numpy())

    Outputs:

    .. testoutput::

        [1. 2. 3.]
    """
    err = (
        abs(expect - actual)
        / maximum(minimum(abs(expect), abs(actual)), Tensor(1.0, dtype="float32"))
    ).max()
    result = apply(AssertEqual(maxerr=maxerr, verbose=verbose), expect, actual, err)[0]
    _sync()  # sync interpreter to get exception
    return result


topk_accuracy = deprecated_func(
    "1.3", "megengine.functional.metric", "topk_accuracy", True
)
copy = deprecated_func("1.3", "megengine.functional.tensor", "copy", True)
