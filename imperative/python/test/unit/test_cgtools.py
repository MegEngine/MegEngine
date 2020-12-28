# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import io

import numpy as np
import pytest

import megengine
import megengine.functional as F
import megengine.module as M
import megengine.utils.comp_graph_tools as cgtools
from megengine.core.ops.builtin import Elemwise
from megengine.core.tensor import megbrain_graph as mgb_graph
from megengine.core.tensor.megbrain_graph import apply_normal_op
from megengine.core.tensor.utils import astensor1d
from megengine.jit import trace


def make_dev_tensor(value, dtype=None, device=None):
    return megengine.tensor(value, dtype=dtype, device=device)._dev_tensor()


def test_replace_vars():
    g = mgb_graph.Graph()
    g.options.async_exec_level = 0b100
    device = "xpux"
    dtype = np.float32
    a = mgb_graph.InputNode(device=device, dtype=dtype, graph=g)
    const = g.make_const(1.234, device=device)
    add_op = Elemwise(Elemwise.Mode.ADD)
    mul_op = Elemwise(Elemwise.Mode.MUL)
    a_plus_a = apply_normal_op(add_op, a.outputs[0], a.outputs[0])[0]
    a_plus_a_mul_const = apply_normal_op(mul_op, a_plus_a, const)[0]
    rst = apply_normal_op(add_op, a_plus_a_mul_const, a.outputs[0])[0]
    (new,) = cgtools.replace_vars([rst._node], {const._node: a_plus_a._node})
    out = mgb_graph.OutputNode(mgb_graph.VarNode(new))
    func = g.compile(out.outputs[0])
    func.execute()
    x = make_dev_tensor(5.0, device=device)
    a.set_value(x)
    res = out.get_value().numpy()
    np.testing.assert_equal(res, np.array([105.0]))


def test_replace_oprs():
    g = mgb_graph.Graph()
    g.options.async_exec_level = 0b100
    device = "xpux"
    dtype = np.float32
    a = mgb_graph.InputNode(device=device, dtype=dtype, graph=g)
    const = g.make_const(1.25, device=device)
    add_op = Elemwise(Elemwise.Mode.ADD)
    mul_op = Elemwise(Elemwise.Mode.MUL)
    a_plus_a = apply_normal_op(add_op, a.outputs[0], a.outputs[0])[0]
    old_opr = a_plus_a.op
    a_plus_a_mul_const = apply_normal_op(mul_op, a_plus_a, const)[0]
    a_mul_a = apply_normal_op(mul_op, a.outputs[0], a.outputs[0])[0]
    new_opr = a_mul_a.op
    (new,) = cgtools.replace_oprs(
        [a_plus_a_mul_const._node], {old_opr._node: new_opr._node}
    )
    out = mgb_graph.OutputNode(mgb_graph.VarNode(new))
    func = g.compile(out.outputs[0])
    func.execute()
    x = make_dev_tensor(5.0, device=device)
    a.set_value(x)
    res = out.get_value().numpy()
    np.testing.assert_equal(res, np.array([5.0 * 5.0 * 1.25]))


def test_graph_traversal():
    net = M.Conv2d(3, 32, 3)

    @trace(symbolic=True, capture_as_const=True)
    def fun(data):
        x = net(data)
        return x

    data = np.random.random([1, 3, 224, 224]).astype(np.float32)
    for _ in range(3):
        fun(megengine.tensor(data))

    file = io.BytesIO()
    fun.dump(file, optimize_for_inference=False)
    file.seek(0)
    cg, _, outputs = mgb_graph.load_graph(file)

    _, map_vars, var2oprs, *_ = cgtools.graph_traversal(outputs)
    input_var = map_vars[1]
    _, var_idx = var2oprs[input_var.id][0]

    assert var_idx == 0


def test_load_refcnt():
    graph = mgb_graph.Graph()
    varnode = graph.make_const(0)
    buf, _ = mgb_graph.dump_graph([varnode])
    graph, _, (varnode,) = mgb_graph.load_graph(io.BytesIO(buf))
    del graph
    varnode.owner


def test_get_opr_seq():
    class Net(M.Module):
        def __init__(self):
            super().__init__()
            self.data = megengine.tensor(
                np.random.random((1, 1, 4, 4)), dtype=np.float32
            )

        def forward(self, input):
            A = input.shape[0]
            shape = astensor1d((A, A), self.data, dtype="int32", device=input.device)
            x = F.reshape(self.data, shape)
            o = input + x
            return o

    net = Net()
    input = megengine.tensor(np.random.random((4, 4)), dtype=np.float32)

    @trace(symbolic=True, capture_as_const=True)
    def func(inp, *, net=None):
        return net(inp)

    func(input, net=net)
    file = io.BytesIO()
    func.dump(file, optimize_for_inference=False)
    file.seek(0)
    *_, outputs = mgb_graph.load_graph(file)

    seq_1 = cgtools.get_oprs_seq(outputs, True)
    assert len(seq_1) == 5

    seq_2 = cgtools.get_oprs_seq(outputs, False)
    assert len(seq_2) == 6
