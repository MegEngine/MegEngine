from collections import OrderedDict

import numpy as np

import megengine.functional as F
import megengine.module as M
from megengine import Tensor
from megengine.traced_module import TracedModule, trace_module
from megengine.traced_module.expr import CallFunction


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
            OrderedDict([("a", M.Elemwise("ADD")), ("b", M.Elemwise("ADD"))]),
            M.Elemwise("RELU"),
            M.Elemwise("RELU"),
        ]

    def forward(self, a, b):
        x = self.modules[0](a, b)
        y = self.modules[1](a, b)
        assert list(self.modules[2].keys()) == ["a", "b"]
        for _, m in self.modules[2].items():
            y = m(x, y)
        for m in self.modules[3:]:
            y = m(y)
        return y


class MyModule4(M.Module):
    def __init__(self):
        super().__init__()
        self.add = F.add

    def forward(self, x, y):
        return self.add(x, y)


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
    assert isinstance(tm3.modules.__dict__["3"], M.Elemwise)

    m4 = MyModule4()
    tm4 = trace_module(m4, a, b)
    assert len(tm4.graph._exprs) == 1
    assert isinstance(tm4.graph._exprs[0], CallFunction)
