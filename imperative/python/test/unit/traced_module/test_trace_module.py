from collections import OrderedDict

import numpy as np

import megengine.functional as F
import megengine.module as M
from megengine import Tensor
from megengine.core._imperative_rt.core2 import apply
from megengine.core.ops import builtin
from megengine.module import Module
from megengine.traced_module import TracedModule, enable_expr_checker, trace_module
from megengine.traced_module.expr import Apply, CallFunction, CallMethod, Constant


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


class MyModule5(M.Module):
    def forward(self, x):
        a = x + x
        b = x * a
        b.name = "result"
        return b


def test_trace_module():
    enable_expr_checker()
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
    np.testing.assert_equal(tm4(a, b).numpy(), 3)
    np.testing.assert_equal(tm4(a, y=b).numpy(), 3)
    np.testing.assert_equal(tm4(x=a, y=b).numpy(), 3)

    tm4 = trace_module(m4, a, y=b)
    np.testing.assert_equal(tm4(a, b).numpy(), 3)
    np.testing.assert_equal(tm4(a, y=b).numpy(), 3)
    np.testing.assert_equal(tm4(x=a, y=b).numpy(), 3)

    tm4 = trace_module(m4, x=a, y=b)
    np.testing.assert_equal(tm4(a, b).numpy(), 3)
    np.testing.assert_equal(tm4(a, y=b).numpy(), 3)
    np.testing.assert_equal(tm4(x=a, y=b).numpy(), 3)

    tm5 = trace_module(tm4, a, b)
    np.testing.assert_equal(tm5(a, b).numpy(), 3)
    np.testing.assert_equal(tm5(a, y=b).numpy(), 3)
    np.testing.assert_equal(tm5(x=a, y=b).numpy(), 3)

    tm5 = trace_module(tm4, a, y=b)
    np.testing.assert_equal(tm5(a, b).numpy(), 3)
    np.testing.assert_equal(tm5(a, y=b).numpy(), 3)
    np.testing.assert_equal(tm5(x=a, y=b).numpy(), 3)

    tm5 = trace_module(tm4, x=a, y=b)
    np.testing.assert_equal(tm5(a, b).numpy(), 3)
    np.testing.assert_equal(tm5(a, y=b).numpy(), 3)
    np.testing.assert_equal(tm5(x=a, y=b).numpy(), 3)

    assert len(tm4.graph._exprs) == 1
    assert isinstance(tm4.graph._exprs[0], CallFunction)

    class MyModule5(Module):
        def __init__(self):
            super().__init__()
            self.m1 = tm4

        def forward(self, x, y):
            return self.m1(x, y)

    tm6 = trace_module(MyModule5(), a, b)
    assert tm6.m1.argspec is None
    assert tm6.m1._is_top is False


def test_trace_module_2():
    class Model(M.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            out = x.shape
            out = apply(builtin.Elemwise(mode="ADD"), out, Tensor(1))
            return out

    traced_model = trace_module(Model(), Tensor(([1,])))

    assert isinstance(traced_model.graph._exprs[0], Apply) and isinstance(
        traced_model.graph._exprs[0].opdef, builtin.GetVarShape
    )
    assert isinstance(traced_model.graph._exprs[1], Constant)
    assert isinstance(traced_model.graph._exprs[2], Apply) and isinstance(
        traced_model.graph._exprs[2].opdef, builtin.Elemwise
    )
    assert int(traced_model(Tensor([1, 2]))[0]) == 3


def test_rename():
    model = MyModule5()
    tm_model = trace_module(model, Tensor(1))
    assert isinstance(tm_model.graph.outputs[0].expr, CallMethod)
