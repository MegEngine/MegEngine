# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import copy

import numpy as np

import megengine.autodiff as ad
import megengine.functional as F
import megengine.optimizer as optimizer
from megengine import Parameter
from megengine import Tensor as tensor
from megengine import tensor
from megengine.core.autodiff.grad import Function
from megengine.module import Module


def test_single_input():
    data_shape = (9, 2, 6)
    av = np.random.random(data_shape).astype(np.float32)

    class MulFunc(Function):
        def forward(self, a):
            self.a = a
            return a * 10

        def backward(self, grad_o):
            return grad_o * 10

    class Simple(Module):
        def __init__(self, a):
            super().__init__()
            self.a = Parameter(a, dtype=np.float32)
            self.layer1 = MulFunc()

        def forward(self):
            x = self.layer1(self.a)
            return x

    net = Simple(av)
    gm = ad.GradManager().attach(net.parameters())
    opt = optimizer.SGD(net.parameters(), lr=1.0)

    opt.clear_grad()
    with gm:
        loss = net()
        gm.backward(loss.sum())
    opt.step()

    np.testing.assert_almost_equal(loss.numpy(), (av * 10))
    np.testing.assert_almost_equal(net.a.numpy(), (av - 10))


def test_multi_input():
    data_shape = (9, 2, 6)
    av = np.random.random(data_shape).astype(np.float32)
    bv = np.random.random(data_shape).astype(np.float32)

    class MulFunc(Function):
        def forward(self, a, b):
            self.a = a
            self.b = b
            return a * b

        def backward(self, grad_o):
            return grad_o * self.b * 2, grad_o * self.a * 3

    class Simple(Module):
        def __init__(self, a, b):
            super().__init__()
            self.a = Parameter(a, dtype=np.float32)
            self.b = Parameter(b, dtype=np.float32)
            self.layer1 = MulFunc()

        def forward(self):
            x = self.layer1(self.a, self.b)
            return x

    net = Simple(av, bv)
    gm = ad.GradManager().attach(net.parameters())
    opt = optimizer.SGD(net.parameters(), lr=1.0)

    opt.clear_grad()
    with gm:
        loss = net()
        gm.backward(loss.sum())
    opt.step()

    np.testing.assert_almost_equal(loss.numpy(), (av * bv))
    np.testing.assert_almost_equal(net.a.numpy(), (av - 2 * bv))
    np.testing.assert_almost_equal(net.b.numpy(), (bv - 3 * av))


def test_multi_output():
    data_shape = (9, 2, 6)
    av = np.random.random(data_shape).astype(np.float32)
    bv = np.random.random(data_shape).astype(np.float32)

    class MulFunc(Function):
        def forward(self, a, b):
            self.a = a
            self.b = b
            return a * b, a + b

        def backward(self, grad_1, grad_2):
            return grad_1 * (self.b + 1), grad_2 * (self.a + 1)

    class Simple(Module):
        def __init__(self, a, b):
            super().__init__()
            self.a = Parameter(a, dtype=np.float32)
            self.b = Parameter(b, dtype=np.float32)
            self.layer1 = MulFunc()

        def forward(self):
            x, y = self.layer1(self.a, self.b)
            return x + y

    net = Simple(av, bv)
    gm = ad.GradManager().attach(net.parameters())
    opt = optimizer.SGD(net.parameters(), lr=1.0)

    opt.clear_grad()
    with gm:
        loss = net()
        gm.backward(loss.sum())
    opt.step()

    np.testing.assert_almost_equal(loss.numpy(), (av * bv + av + bv), decimal=6)
    np.testing.assert_almost_equal(net.a.numpy(), (av - bv - 1), decimal=6)
    np.testing.assert_almost_equal(net.b.numpy(), (bv - av - 1), decimal=6)


def test_skip_invalid_grad():
    data_shape = (1, 9, 2, 6)
    av = np.random.random(data_shape).astype(np.float32)
    bv = np.random.random(data_shape).astype(np.float32)
    c = np.random.random(data_shape).astype(np.float32)
    cookie = tensor(c)

    class EqWithFakeGrad(Function):
        def forward(self, a, b):
            return a + b

        def backward(self, grad_o):
            _ = grad_o
            return cookie, cookie

    class Simple(Module):
        def __init__(self, a, b):
            super().__init__()
            self.a = Parameter(a, dtype=np.float32)
            self.b = Parameter(b, dtype=np.float32)
            self.layer1 = EqWithFakeGrad()

        def forward(self):
            x = self.layer1(self.a, self.b)
            return x

    net = Simple(av, bv)
    optim = optimizer.SGD(net.parameters(), lr=1.0)
    gm = ad.GradManager().attach(net.parameters())
    optim.clear_grad()
    with gm:
        loss = net().sum()
        gm.backward(loss)
    optim.step()
    np.testing.assert_almost_equal(net.a.numpy(), av - c)
    np.testing.assert_almost_equal(net.b.numpy(), bv - c)


def test_ste():
    class STE(Function):
        def forward(self, x):
            maxv, minv = x.max(), x.min()
            scale = F.maximum(maxv, -minv) / 127
            return F.round(x / scale) * scale

        def backward(self, grad_y):
            return grad_y

    class Simple(Module):
        def __init__(self, a):
            super().__init__()
            self.a = Parameter(a, dtype=np.float32)
            self.layer1 = STE()

        def forward(self):
            x = self.layer1(self.a)
            x = (x * 2.0).sum()
            return x

    data_shape = (1, 9, 2, 6)
    av = np.random.random(data_shape).astype(np.float32)
    net = Simple(av)
    optim = optimizer.SGD(net.parameters(), lr=1.0)
    gm = ad.GradManager().attach(net.parameters())
    optim.clear_grad()

    with gm:
        loss = net()
        gm.backward(loss.sum())
    optim.step()

    np.testing.assert_almost_equal(
        net.a.numpy(),
        av - np.broadcast_to(np.array([2.0], dtype=np.float32), data_shape),
    )


def test_deepcopy():
    class Sigmoid(Function):
        def __init__(self, param):
            super().__init__()
            self.param = param

        def forward(self, x):
            y = 1 / (1 + F.exp(-x))
            self.save_for_backward(y)
            return y

        def backward(self, grad_y):
            (y,) = self.saved_tensors
            return grad_y * y * (1 - y)

    origin = Sigmoid(0)
    new = copy.deepcopy(Sigmoid(0))
    assert new.param == origin.param


def test_none_in_out_grad():
    class Test(Function):
        def forward(self, a, b):
            return a, b

        def backward(self, grad_a, grad_b):
            assert grad_b is None
            return (grad_a, None)

    class Simple(Module):
        def __init__(self, a, b):
            super().__init__()
            self.a = Parameter(a, dtype=np.float32)
            self.b = Parameter(b, dtype=np.float32)
            self.layer = Test()

        def forward(self):
            aa, bb = self.layer(self.a, self.b)
            return aa, bb

    a = tensor(np.array([1.0], dtype=np.float32))
    b = tensor(np.array([2.0], dtype=np.float32))
    net = Simple(a, b)
    optim = optimizer.SGD(net.parameters(), lr=1.0)
    gm = ad.GradManager().attach(net.parameters())
    optim.clear_grad()
    with gm:
        loss, _ = net()
        gm.backward(loss)
    optim.step()

    np.testing.assert_almost_equal(
        net.a.numpy(), np.array([1.0 - 1.0], dtype=np.float32)
    )
    np.testing.assert_almost_equal(
        net.b.numpy(), np.array([2.0 - 0.0], dtype=np.float32)
    )


def test_zero_grad():
    class StopGradient(Function):
        def forward(self, a):
            return a

        def backward(self, *_):
            return None

    class Simple(Module):
        def __init__(self, a):
            super().__init__()
            self.a = Parameter(a, dtype=np.float32)
            self.layer = StopGradient()

        def forward(self):
            b = self.a * 3.0
            c = self.a * 4.0
            return self.layer(b) + c

    a = tensor(np.array([1.0], dtype=np.float32))
    net = Simple(a)
    optim = optimizer.SGD(net.parameters(), lr=1.0)
    gm = ad.GradManager().attach(net.parameters())
    optim.clear_grad()

    with gm:
        loss = net()
        gm.backward(loss.sum())
    optim.step()
    np.testing.assert_almost_equal(
        net.a.numpy(), np.array([1.0 - 4.0], dtype=np.float32),
    )
