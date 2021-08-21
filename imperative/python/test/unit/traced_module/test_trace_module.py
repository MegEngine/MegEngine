import numpy as np

import megengine.module as M
from megengine import Tensor
from megengine.experimental.traced_module import TracedModule, trace_module


class MyModule1(M.Module):
    def forward(self, x):
        y = Tensor(x)
        y += 1
        x = x + 2
        return x, y


class MyModule2(M.Module):
    def forward(self, x):
        y = Tensor([1, x, 1])
        y += 1
        x = x + 2
        return x, y


class MyModule3(M.Module):
    def __init__(self):
        super().__init__()
        self.modules = [
            M.Elemwise("ADD"),
            M.Elemwise("ADD"),
            {"a": M.Elemwise("ADD"), "b": M.Elemwise("ADD")},
        ]

    def forward(self, a, b):
        x = self.modules[0](a, b)
        y = self.modules[1](a, b)
        y = self.modules[2]["a"](x, y)
        y = self.modules[2]["b"](x, y)
        return y


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

    a, b = Tensor(1), Tensor(2)
    m3 = MyModule3()
    gt = m3(a, b)
    tm3 = trace_module(m3, a, b)
    out = tm3(a, b)
    np.testing.assert_equal(out.numpy(), gt.numpy())
    assert isinstance(tm3.modules.__dict__["0"], M.Elemwise)
    assert isinstance(tm3.modules.__dict__["2"], TracedModule)
    assert isinstance(tm3.modules.__dict__["2"].a, M.Elemwise)
