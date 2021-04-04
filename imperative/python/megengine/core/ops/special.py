# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

from .._imperative_rt import make_const
from .._imperative_rt.core2 import SymbolVar, Tensor


class Const:
    def __init__(self, value=None, *, dtype=None, device=None):
        self.value = np.asarray(value, dtype=dtype)
        self.dtype = dtype
        self.device = device

    def __call__(self, *reference):
        from ...tensor import Tensor

        device = self.device

        if len(reference) != 0:
            reference = reference[0]
            assert isinstance(
                reference, (SymbolVar, Tensor)
            ), "Reference should be Tensor or VarNode"

            if device is None:
                device = reference.device

            if isinstance(reference, SymbolVar):
                cls = type(reference)
                rst = cls(make_const(reference.graph, self.value, device, self.dtype))
                return (rst,)

        return (Tensor(self.value, self.dtype, self.device, True),)
