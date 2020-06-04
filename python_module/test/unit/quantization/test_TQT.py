# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np
import pytest

import megengine as mge
import megengine._internal as mgb
from megengine.core import tensor
from megengine.quantization.fake_quant import TQT_Function
from megengine.test import assertTensorClose


class numpy_TQT_Function:
    def __init__(self, lowerbound, upperbound):
        super().__init__()
        self.lowerbound = lowerbound
        self.upperbound = upperbound

    def forward(self, inp, scale):
        t = 2 ** scale
        # t = F.maximum(t, 1e-4)
        inp_scaled = inp / t
        inp_clipped = np.maximum(
            np.minimum(inp_scaled, self.upperbound), self.lowerbound
        )
        inp_rounded = np.round(inp_clipped)
        inp_flq = inp_rounded * t
        self.saved_tensors = (inp_scaled, inp_rounded, t)
        return inp_flq

    def backward(self, grad_inp_flq):
        (inp_scaled, inp_rounded, t) = self.saved_tensors
        mask_clip = (inp_scaled < -0.5 + self.lowerbound) + (
            inp_scaled > self.upperbound + 0.5
        )  # mask for accumulating the gradients of |data_scaled|>L
        mask_quant = np.abs(
            mask_clip - 1
        )  # mask for accumulating the gradients with |data_scaled|<=L
        grad_quant = (
            grad_inp_flq * mask_quant * (inp_rounded - inp_scaled)
        )  # gradient within |data_scaled|<=L
        grad_clip = (
            grad_inp_flq * mask_clip * inp_rounded
        )  # gradient with   | data_scaled|>L
        grad_s = grad_clip.sum() + grad_quant.sum()
        # dL/ds = dL/dt * t * ln(2)
        grad_s = grad_s * t * np.log(2)
        grad_inp = grad_inp_flq * mask_quant
        return grad_inp, grad_s


def test_TQT():
    f = TQT_Function(-127, 127)
    nf = numpy_TQT_Function(-127, 127)

    def check_inp(a, b, c, a_np, b_np, c_np):
        assertTensorClose(
            f.forward(a, b).numpy(), nf.forward(a_np, b_np).astype("float32")
        )
        c1, c2 = f.backward(c)
        c1_np, c2_np = nf.backward(c_np)
        assertTensorClose(c1.numpy(), c1_np.astype("float32"))
        assertTensorClose(c2.numpy(), c2_np.astype("float32"))

    a = tensor()
    b = tensor()
    a_np = np.random.random((4, 3)).astype("float32")
    b_np = np.random.random((1)).astype("float32")
    a.set_value(a_np)
    b.set_value(b_np)
    check_inp(a, b, b, a_np, b_np, b_np)
