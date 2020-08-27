# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

import megengine.optimizer as optimizer
from megengine import Parameter
from megengine import Tensor as tensor
from megengine import tensor
from megengine.core.tensor.function import Function
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
    optim = optimizer.SGD(net.parameters(), lr=1.0)
    optim.zero_grad()

    with optim.record():
        loss = net()
        optim.backward(loss.sum())
    optim.step()

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
    optim = optimizer.SGD(net.parameters(), lr=1.0)
    optim.zero_grad()

    with optim.record():
        loss = net()
        optim.backward(loss.sum())
    optim.step()

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
    optim = optimizer.SGD(net.parameters(), lr=1.0)
    optim.zero_grad()

    with optim.record():
        loss = net()
        optim.backward(loss.sum())
    optim.step()

    np.testing.assert_almost_equal(loss.numpy(), (av * bv + av + bv), decimal=6)
    np.testing.assert_almost_equal(net.a.numpy(), (av - bv - 1), decimal=6)
    np.testing.assert_almost_equal(net.b.numpy(), (bv - av - 1), decimal=6)
