# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

# from .._imperative_rt.core2 import Tensor
from ..tensor.core import OpBase, TensorBase, apply


class Const:
    def __init__(self, value=None, *, dtype=None, device=None):
        self.value = np.asarray(value, dtype=dtype)
        self.dtype = dtype
        self.device = device

    def __call__(self, *reference):
        from ...tensor import Tensor

        device = self.device
        if device is None:
            device = reference[0].device

        return (Tensor(self.value, self.dtype, self.device, True),)
