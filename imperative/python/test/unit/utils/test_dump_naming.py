# -*- coding: utf-8 -*-
import io

import numpy as np
import pytest

import megengine.functional as F
import megengine.module as M
import megengine.utils.comp_graph_tools as cgtools
from megengine import Parameter, Tensor
from megengine.core.tensor import megbrain_graph as G
from megengine.jit.tracing import trace
from megengine.quantization.quantize import quantize, quantize_qat
from megengine.utils.naming import AutoNaming


def _dump_and_load(func, symbolic, keep_opr_name=True):
    AutoNaming.clear()
    func = trace(func, symbolic=symbolic, capture_as_const=True)
    x = Tensor(np.ones(shape=(2, 3)))
    func(x).numpy()
    file = io.BytesIO()
    func.dump(
        file,
        optimize_for_inference=False,
        arg_names=("x",),
        keep_opr_name=keep_opr_name,
        keep_var_name=2,
    )
    file.seek(0)
    outputs = G.load_graph(file).output_vars_list
    ops = cgtools.get_oprs_seq(outputs)
    return ops


@pytest.mark.parametrize("symbolic", [False, True])
def test_auto_naming(symbolic):
    class Simple(M.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def forward(self, x):
            return x + x

    m = Simple("simple")
    op = _dump_and_load(m, symbolic)[-1]
    assert op.name == "simple.ADD"
    assert op.outputs[0].name == "simple.ADD"


@pytest.mark.parametrize("symbolic", [False, True])
def test_user_named_tensor(symbolic):
    class Simple(M.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self.k = Parameter(1.0, name="k")

        def forward(self, x):
            x = x + x
            x.name = "o_x"
            return x

    m = Simple("simple")

    op = _dump_and_load(m, symbolic)[-1]
    assert op.name == "simple.ADD"
    assert op.outputs[0].name == "o_x"


@pytest.mark.parametrize("symbolic", [False, True])
def test_user_named_param(symbolic):
    class Simple(M.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self.k = Parameter(2.0, name="k")

        def forward(self, x):
            return self.k * x

    m = Simple("simple")

    op = _dump_and_load(m, symbolic)[-1]
    assert op.inputs[0].name == "x"
    assert op.inputs[1].name == "simple.k"


@pytest.mark.parametrize("symbolic", [False, True])
def test_without_module(symbolic):
    def f(x):
        return 2 * x

    op = _dump_and_load(f, symbolic)[-1]
    assert op.name == "MUL"


@pytest.mark.parametrize("symbolic", [False, True])
def test_ignore_top_module(symbolic):
    class Simple(M.Module):
        def forward(self, x):
            return x + x

    m = Simple()
    op = _dump_and_load(m, symbolic)[-1]
    assert op.name == "ADD"
    assert op.outputs[0].name == "ADD"


@pytest.mark.parametrize("symbolic", [False, True])
def test_with_submodule(symbolic):
    class Simple(M.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self.linear = M.Linear(3, 3)

        def forward(self, x):
            x = self.linear(x)
            return x

    m = Simple("simple")

    ops = _dump_and_load(m, symbolic)
    assert ops[-1].name == "simple.linear.ADD"
    assert ops[-2].name == "simple.linear.MatrixMul"
    assert ops[-1].outputs[0].name == "simple.linear.ADD"


@pytest.mark.parametrize("symbolic", [False, True])
def test_with_submodule_in_container(symbolic):
    class Simple(M.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self.l0 = [M.Linear(3, 3) for _ in range(2)]
            self.l1 = tuple(self.l0)
            self.l2 = dict(zip(["l2-0", "l2-1"], self.l0))

        def forward(self, x):
            for i in range(2):
                x = self.l0[i](x)
                x = self.l1[i](x)
                x = self.l2["l2-%d" % i](x)
            return x

    m = Simple("simple")

    ops = _dump_and_load(m, symbolic)
    assert ops[-1].outputs[0].name == "simple.l0.1.ADD[2]"
    assert ops[-1].name == "simple.l0.1.ADD[2]"
    assert ops[-2].name == "simple.l0.1.MatrixMul[2]"
    assert ops[-3].name == "simple.l0.1.ADD[1]"
    assert ops[-4].name == "simple.l0.1.MatrixMul[1]"
    assert ops[-5].name == "simple.l0.1.ADD[0]"
    assert ops[-6].name == "simple.l0.1.MatrixMul[0]"


@pytest.mark.parametrize("symbolic", [False, True])
def test_named_submodule(symbolic):
    class Simple(M.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self.linear = M.Linear(3, 3, name="x")

        def forward(self, x):
            x = self.linear(x)
            return x

    m = Simple("simple")

    ops = _dump_and_load(m, symbolic)
    assert ops[-1].name == "simple.x.ADD"
    assert ops[-2].name == "simple.x.MatrixMul"
    assert ops[-1].outputs[0].name == "simple.x.ADD"


@pytest.mark.parametrize("symbolic", [False, True])
def test_with_same_operators(symbolic):
    class Simple(M.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def forward(self, x):
            x = F.relu(x)
            x = F.relu(x)
            return x

    m = Simple("simple")

    ops = _dump_and_load(m, symbolic)
    assert ops[-1].name == "simple.RELU[1]"
    assert ops[-2].name == "simple.RELU[0]"


@pytest.mark.parametrize("symbolic", [False, True])
def test_not_keep_opr_name(symbolic):
    def f(x):
        return 2 * x

    op = _dump_and_load(f, symbolic, False)[-1]
    assert op.name == "MUL(x,const<2>[2])[4]"


@pytest.mark.parametrize("tensor_name, var_name", [("data", "data"), (None, "arg_0")])
def test_catch_input_name(tensor_name, var_name):
    def f(x):
        return 2 * x

    func = trace(f, symbolic=True, capture_as_const=True)
    x = Tensor(np.ones(shape=(2, 3)), name=tensor_name)
    func(x).numpy()
    file = io.BytesIO()
    func.dump(file, optimize_for_inference=False, keep_opr_name=True, keep_var_name=2)
    file.seek(0)
    outputs = G.load_graph(file).output_vars_list
    op = cgtools.get_oprs_seq(outputs)[-1]
    assert op.inputs[0].name == var_name


@pytest.mark.parametrize("symbolic", [False, True])
def test_quantized_module_auto_naming(symbolic):
    class Simple(M.Module):
        def __init__(self, name):
            super().__init__(name=name)
            self.quant = M.QuantStub()
            self.linear = M.Linear(3, 3, bias=True)
            self.dequant = M.DequantStub()

        def forward(self, x):
            out = self.quant(x)
            out = self.linear(out)
            out = self.dequant(out)
            return out

    m = Simple("simple")
    quantize_qat(m)
    quantize(m)
    m.eval()

    ops = _dump_and_load(m, symbolic)
    ops_name = (
        "x",
        "simple.quant.TypeCvt",
        "simple.linear.MatrixMul",
        "simple.linear.ADD",
        "simple.linear.TypeCvt",
        "simple.dequant.TypeCvt",
    )
    for op, name in zip(ops, ops_name):
        assert op.name == name


@pytest.mark.parametrize("symbolic", [False, True])
def test_quantized_module_user_naming(symbolic):
    class Simple(M.Module):
        def __init__(self, name):
            super().__init__(name=name)
            self.quant = M.QuantStub()
            self.linear = M.Linear(3, 3, bias=True, name="user-linear")
            self.dequant = M.DequantStub()

        def forward(self, x):
            out = self.quant(x)
            out = self.linear(out)
            out = self.dequant(out)
            return out

    m = Simple("simple")
    quantize_qat(m)
    quantize(m)
    m.eval()

    ops = _dump_and_load(m, symbolic)
    ops_name = (
        "x",
        "simple.quant.TypeCvt",
        "simple.user-linear.MatrixMul",
        "simple.user-linear.ADD",
        "simple.user-linear.TypeCvt",
        "simple.dequant.TypeCvt",
    )
    for op, name in zip(ops, ops_name):
        assert op.name == name


@pytest.mark.parametrize("symbolic", [False, True])
def test_quantized_module_user_naming_param(symbolic):
    class Simple(M.Module):
        def __init__(self, name):
            super().__init__(name=name)
            self.quant = M.QuantStub()
            self.linear = M.Linear(3, 3, bias=True)
            self.dequant = M.DequantStub()

            self.linear.weight.name = "user-weight"
            self.linear.bias.name = "user-bias"

        def forward(self, x):
            out = self.quant(x)
            out = self.linear(out)
            out = self.dequant(out)
            return out

    m = Simple("simple")
    quantize_qat(m)
    quantize(m)
    m.eval()

    ops = _dump_and_load(m, symbolic)

    (matrix_mul_op,) = [op for op in ops if op.name == "simple.linear.MatrixMul"]
    for var in matrix_mul_op.inputs:
        assert var.name in ("simple.quant.TypeCvt", "simple.linear.user-weight")
    # WONTFIX: bias' name does not meet expectations because of astype operator after quantization
