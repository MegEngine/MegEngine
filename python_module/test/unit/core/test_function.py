# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

import megengine.functional as F
from megengine.core import Function, tensor
from megengine.test import assertTensorClose


def test_a_plus_b():
    data_shape = (1, 9, 2, 6)
    av = np.random.random(data_shape).astype(np.float32)
    bv = np.random.random(data_shape).astype(np.float32)
    a = tensor(av)
    b = tensor(bv)

    class MulFunc(Function):
        def forward(self, a, b):
            return a * b

        def backward(self, grad_o):
            return (grad_o * b * 2, grad_o * a * 3)

    c = MulFunc()(a, b).sum()
    assertTensorClose(c.numpy(), (av * bv).sum())
    assertTensorClose(F.grad(c, a, use_virtual_grad=False).numpy(), bv * 2)
    assertTensorClose(F.grad(c, b, use_virtual_grad=False).numpy(), av * 3)


def test_skip_invalid_grad():
    data_shape = (1, 9, 2, 6)
    av = np.random.random(data_shape).astype(np.float32)
    bv = np.random.random(data_shape).astype(np.float32)
    a = tensor(av)
    b = tensor(bv)
    cookie = tensor(np.random.random(data_shape).astype(np.float32))

    class EqWithFakeGrad(Function):
        def forward(self, a, b):
            return a == b

        def backward(self, grad_o):
            _ = grad_o
            return cookie, cookie

    c = EqWithFakeGrad()(a, b).sum()
    assertTensorClose(c.numpy(), (av == bv).sum().astype(np.float32))
    assertTensorClose(F.grad(c, a, use_virtual_grad=False).numpy(), cookie)
    assertTensorClose(F.grad(c, b, use_virtual_grad=False).numpy(), cookie)


def test_ste():
    class STE(Function):
        def forward(self, x):
            maxv, minv = x.max(), x.min()
            scale = F.maximum(maxv, -minv) / 127
            return F.round(x / scale) * scale

        def backward(self, grad_y):
            return grad_y

    data_shape = (1, 9, 2, 6)
    av = np.random.random(data_shape).astype(np.float32)
    a = tensor(av)
    q = STE()(a)
    q_2 = (q * 2.0).sum()
    assertTensorClose(
        F.grad(q_2, a, use_virtual_grad=False).numpy(),
        np.broadcast_to(np.array([2.0], dtype=np.float32), data_shape),
    )


def test_save_context():
    class Sigmoid(Function):
        def forward(self, x):
            y = 1 / (1 + F.exp(-x))
            self.save_for_backward(y)
            return y

        def backward(self, grad_y):
            (y,) = self.saved_tensors
            return grad_y * y * (1 - y)

    a = tensor(np.array([1926.0817], dtype=np.float32))
    s = Sigmoid()(a)
    s2 = F.sigmoid(a)
    assertTensorClose(s.numpy(), s2.numpy())
    assertTensorClose(
        F.grad(s, a, use_virtual_grad=False).numpy(),
        F.grad(s2, a, use_virtual_grad=False).numpy(),
    )


def test_none_in_out_grad():
    class Test(Function):
        def forward(self, a, b):
            return a, b

        def backward(self, grad_a, grad_b):
            assert grad_b is None
            return (grad_a, 0)

    a = tensor(np.array([1.0], dtype=np.float32))
    b = tensor(np.array([2.0], dtype=np.float32))
    aa, bb = Test()(a, b)
    assertTensorClose(
        F.grad(aa, a, use_virtual_grad=False).numpy(), np.array([1.0], dtype=np.float32)
    )
    assertTensorClose(
        F.grad(aa, b, use_virtual_grad=False).numpy(), np.array([0.0], dtype=np.float32)
    )


def test_zero_grad():
    class StopGradient(Function):
        def forward(self, a):
            return a

        def backward(self, *_):
            return None

    a = tensor(np.array([1.0], dtype=np.float32))
    b = a * 3.0
    c = a * 4.0
    loss = StopGradient()(b) + c
    assertTensorClose(
        F.grad(loss, a, use_virtual_grad=False).numpy(),
        np.array([4.0], dtype=np.float32),
    )
