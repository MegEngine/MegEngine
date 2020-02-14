# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import functools
from typing import Optional, Tuple, Union

import megengine._internal as mgb

from ..core.tensor import Tensor, wrap_io_tensor

__all__ = ["argsort", "sort", "top_k"]


@wrap_io_tensor
def argsort(inp: Tensor, descending: bool = False) -> Tuple[Tensor, Tensor]:
    r"""
    Sort the target 2d matrix by row, return both the sorted tensor and indices.

    :param inp: The input tensor, if 2d, each row will be sorted
    :param descending: Sort in descending order, where the largest comes first. Default: ``False``
    :return: Tuple of two tensors (sorted_tensor, indices_of_int32)

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import  megengine.functional as F
        data = tensor(np.array([1,2], dtype=np.float32))
        sorted, indices = F.argsort(data)
        print(sorted.numpy(), indices.numpy())

    Outputs:

    .. testoutput::
        :options: +NUMBER

        [1. 2.] [0 1]

    """
    assert len(inp.imm_shape) <= 2, "Input should be 1d or 2d"
    if descending:
        order = mgb.opr_param_defs.Argsort.Order.DESCENDING
    else:
        order = mgb.opr_param_defs.Argsort.Order.ASCENDING
    if len(inp.imm_shape) == 1:
        inp = inp.reshape(1, -1)
        tns, ind = mgb.opr.argsort(inp, order=order)
        return tns[0], ind[0]
    return mgb.opr.argsort(inp, order=order)


@functools.wraps(argsort)
def sort(*args, **kwargs):
    return argsort(*args, **kwargs)


@wrap_io_tensor
def top_k(
    inp: Tensor,
    k: int,
    descending: bool = False,
    kth_only: bool = False,
    no_sort: bool = False,
) -> Tuple[Tensor, Tensor]:
    r"""
    Selected the Top-K (by default) smallest elements of 2d matrix by row.

    :param inp: The input tensor, if 2d, each row will be sorted
    :param k: The number of elements needed
    :param descending: If true, return the largest elements instead. Default: ``False``
    :param kth_only: If true, only the k-th element will be returned. Default: ``False``
    :param no_sort: If true, the returned elements can be unordered. Default: ``False``
    :return: Tuple of two tensors (topk_tensor, indices_of_int32)

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import  megengine.functional as F
        data = tensor(np.array([2, 4, 6, 8, 7, 5, 3, 1], dtype=np.float32))
        top, indices = F.top_k(data, 5)
        print(top.numpy(), indices.numpy())

    Outputs:

    .. testoutput::
        :options: +NUMBER

        [1. 2. 3. 4. 5.] [7 0 6 1 5]

    """
    assert len(inp.imm_shape) <= 2, "Input should be 1d or 2d"
    if kth_only:
        raise NotImplementedError(
            "TODO: would enconter:"
            "NotImplementedError: SymbolVar var could not be itered"
        )
    if descending:
        inp = -inp
    Mode = mgb.opr_param_defs.TopK.Mode
    if kth_only:
        mode = Mode.KTH_ONLY
    elif no_sort:
        mode = Mode.VALUE_IDX_NOSORT
    else:
        mode = Mode.VALUE_IDX_SORTED
    if len(inp.imm_shape) == 1:
        inp = inp.reshape(1, -1)
        tns, ind = mgb.opr.top_k(inp, k, mode=mode)
        tns = tns[0]
        ind = ind[0]
    else:
        tns, ind = mgb.opr.top_k(inp, k, mode=mode)
    if descending:
        tns = -tns
    return tns, ind
