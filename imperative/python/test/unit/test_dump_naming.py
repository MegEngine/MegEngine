# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import io

import numpy as np
import pytest

import megengine.functional as F
import megengine.module as M
import megengine.utils.comp_graph_tools as cgtools
from megengine import Parameter, Tensor
from megengine.core.tensor import megbrain_graph as G
from megengine.jit.tracing import trace
from megengine.utils.naming import auto_naming


def _dump_and_load(func, symbolic, keep_opr_name=True):
    auto_naming.clear()
    func = trace(func, symbolic=symbolic, capture_as_const=True)
    x = Tensor(np.ones(shape=(2, 3)))
    func(x).numpy()
    file = io.BytesIO()
    func.dump(
        file,
        optimize_for_inference=False,
        arg_names="x",
        keep_opr_name=keep_opr_name,
        keep_var_name=2,
    )
    file.seek(0)
    *_, outputs = G.load_graph(file)
    op = cgtools.get_oprs_seq(outputs)[-1]
    return op


@pytest.mark.parametrize("symbolic", [False, True])
def test_auto_naming(symbolic):
    class Simple(M.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def forward(self, x):
            return x + x

    m = Simple("simple")
    op = _dump_and_load(m, symbolic)
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

    op = _dump_and_load(m, symbolic)
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

    op = _dump_and_load(m, symbolic)
    assert op.inputs[0].name == "x"
    assert op.inputs[1].name == "simple.k"


@pytest.mark.parametrize("symbolic", [False, True])
def test_without_module(symbolic):
    def f(x):
        return 2 * x

    op = _dump_and_load(f, symbolic)
    assert op.name == "MUL"


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

    op = _dump_and_load(m, symbolic)
    assert op.name == "simple.linear.ADD"
    assert op.inputs[0].owner.name == "simple.linear.MatrixMul"
    assert op.outputs[0].name == "simple.linear.ADD"


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

    op = _dump_and_load(m, symbolic)
    assert op.name == "simple.x.ADD"
    assert op.inputs[0].owner.name == "simple.x.MatrixMul"
    assert op.outputs[0].name == "simple.x.ADD"


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

    op = _dump_and_load(m, symbolic)
    assert op.name == "simple.RELU[1]"
    assert op.inputs[0].owner.name == "simple.RELU[0]"


def test_not_keep_opr_name():
    def f(x):
        return 2 * x

    op = _dump_and_load(f, True, False)
    assert op.name == "MUL(x,2[2])[4]"
