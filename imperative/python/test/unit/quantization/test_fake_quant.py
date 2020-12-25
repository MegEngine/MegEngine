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
from megengine import tensor
from megengine.core.autodiff.grad import Function, Grad
from megengine.core.tensor.utils import make_shape_tuple
from megengine.quantization.fake_quant import TQT_Function
from megengine.quantization.internal_fake_quant import *
from megengine.quantization.utils import QuantMode, fake_quant_tensor


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
        np.testing.assert_allclose(
            f.forward(a, b).numpy(),
            nf.forward(a_np, b_np).astype("float32"),
            rtol=1e-6,
            atol=1e-6,
        )
        c1, c2 = f.backward(c)
        c1_np, c2_np = nf.backward(c_np)
        np.testing.assert_allclose(c1.numpy(), c1_np.astype("float32"), rtol=1e-6)
        np.testing.assert_allclose(c2.numpy(), c2_np.astype("float32"), rtol=5e-5)

    a_np = np.random.random((4, 3)).astype("float32")
    b_np = np.random.random((1)).astype("float32")
    a = tensor(a_np)
    b = tensor(b_np)
    check_inp(a, b, b, a_np, b_np, b_np)




def _save_to(self, name="grad"):
    def callback(tensor, grad):
        setattr(self, name, grad)

    return callback


class Round(Function):
    def forward(self, x):
        return F.round(x)

    def backward(self, output_grads):
        return output_grads


def fake_quant_tensor_gt(inp, scale, zero_point, qmin, qmax):
    oup = Round()(inp / scale) + zero_point
    oup = F.minimum(F.maximum(oup, qmin), qmax)
    oup = (oup - zero_point) * scale
    return oup


def test_fakequant():
    qmin = -126
    qmax = 129

    def run(zero_point, scale):
        q_dict = {}
        q_dict["mode"] = QuantMode.ASYMMERTIC
        q_dict["scale"] = scale
        q_dict["zero_point"] = zero_point
        inp_data = np.random.uniform(low=-512.0, high=512.0, size=(1, 32, 32, 32))
        inp = tensor(inp_data, dtype=np.float32)
        # test forward
        oup = fake_quant_tensor(inp, qmin, qmax, q_dict).numpy()
        oup_gt = fake_quant_tensor_gt(inp, scale, zero_point, qmin, qmax).numpy()
        assert np.allclose(oup, oup_gt)
        assert oup.shape == oup_gt.shape

        # test backward
        x = tensor(inp_data, dtype=np.float32)
        grad = Grad().wrt(x, callback=_save_to(x))
        y = fake_quant_tensor(x, qmin, qmax, q_dict)
        grad(y, tensor(F.ones_like(x)))

        x1 = tensor(inp_data, dtype=np.float32)
        grad = Grad().wrt(x1, callback=_save_to(x1))
        y1 = fake_quant_tensor_gt(x1, scale, zero_point, qmin, qmax)
        grad(y1, tensor(F.ones_like(x1)))

        assert np.allclose(x.grad.numpy(), x1.grad.numpy())
        assert make_shape_tuple(x.grad.shape) == make_shape_tuple(x1.grad.shape)

    zero_point = tensor([1.0], dtype=np.float32)
    scale = tensor([4.0], dtype=np.float32)
    run(zero_point, scale)

    zero_point = tensor(1.0 * np.ones((1, 32, 1, 1)), dtype=np.float32)
    scale = tensor(4.0 * np.ones((1, 32, 1, 1)), dtype=np.float32)
    run(zero_point, scale)
