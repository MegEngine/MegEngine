import numpy as np

from megengine import Tensor
from megengine.experimental.traced_module import trace_module
from megengine.module import Module as M


class MyModule1(M):
    def forward(self, x):
        y = Tensor(x)
        y += 1
        x = x + 2
        return x, y


class MyModule2(M):
    def forward(self, x):
        y = Tensor([1, x, 1])
        y += 1
        x = x + 2
        return x, y


def test_trace_module():

    x = Tensor(1)
    m1 = MyModule1()
    tm1 = trace_module(m1, x)

    m2 = MyModule2()
    tm2 = trace_module(m2, x)
    inp = Tensor(2)
    gt = m1(inp)
    output = tm1(inp)
    for a, b in zip(output, gt):
        np.testing.assert_equal(a.numpy(), b.numpy())

    gt1 = m2(inp)
    output1 = tm2(inp)

    for a, b in zip(output1, gt1):
        np.testing.assert_equal(a.numpy(), b.numpy())
