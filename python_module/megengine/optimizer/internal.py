# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Union

import megengine._internal as mgb

from ..core.tensor import Tensor, tensor


def add_update_fastpath(
    dest: Tensor,
    delta: Tensor,
    *,
    alpha: Union[Tensor, float, int] = 1.0,
    beta: Union[Tensor, float, int] = 1.0,
    bias: Union[Tensor, float, int] = 0.0
):
    """a fast-path ONLY used to update parameters in optimzier, since it
    would bypass computing graph and launch dnn/add_update kernel directly,
    it is more efficient than functional/add_update.
    """

    if isinstance(beta, Tensor) or isinstance(alpha, Tensor):
        delta *= beta
        beta = 1.0
    if isinstance(alpha, Tensor):
        delta += (alpha - 1.0) * dest
        alpha = 1.0
    if isinstance(bias, Tensor):
        delta += bias
        bias = 0.0

    if not isinstance(delta, Tensor):
        delta = tensor(delta, device=dest.device, dtype=dest.dtype)

    def get_v(x):
        if x._Tensor__val is None:
            assert isinstance(x._Tensor__sym, mgb.SymbolVar)
            return x._Tensor__sym.eager_val
        else:
            assert isinstance(x._Tensor__val, mgb.SharedND)
            return x._Tensor__val

    mgb.mgb._add_update_fastpath(get_v(dest), get_v(delta), alpha, beta, bias)
    return dest
