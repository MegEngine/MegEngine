# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import pickle

import numpy as np

import megengine.functional as F
import megengine.module as M
from megengine.module.identity import Identity
from megengine.traced_module import trace_module
from megengine.traced_module.expr import CallFunction, Expr, GetAttr
from megengine.traced_module.node import Node


class IdentityMod(M.Module):
    def forward(self, x):
        return x


class MyBlock(M.Module):
    def __init__(self, in_channels=3, channels=3):
        super(MyBlock, self).__init__()
        self.conv1 = M.Conv2d(in_channels, channels, 3, 1, padding=1, bias=False)
        self.bn1 = M.BatchNorm2d(channels)
        self.nothing = IdentityMod()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x) + 1
        x = self.nothing(x)
        return x


class MyModule(M.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.block0 = MyBlock()
        self.block1 = MyBlock()
        self.nothing = IdentityMod()

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        x = self.nothing(x)
        return x


class NewModule(M.Module):
    def __init__(self, traced_module):
        super(NewModule, self).__init__()
        self.module = traced_module

    def forward(self, x):
        x = x - 1
        x = self.module(x)
        x = x + 1
        return x


def _init_cls(cls):
    module = cls()
    x = F.ones((1, 3, 3, 3))
    y = module(x)
    traced_module = trace_module(module, x)
    return traced_module, x, y


def _init_block():
    return _init_cls(MyBlock)


def _init_module():
    return _init_cls(MyModule)


def test_search():
    traced_module, *_ = _init_block()
    graph = traced_module.graph
    relu_expr = graph.get_function_by_type(F.relu).as_unique()
    assert isinstance(relu_expr, CallFunction) and relu_expr.func == F.relu


def test_insert():
    traced_module, x, expect = _init_block()
    graph = traced_module.graph
    relu_out = graph.get_function_by_type(F.relu).as_unique().outputs[0]
    with graph.insert_exprs():
        neg_out = F.neg(relu_out)
    graph.replace_node({relu_out: neg_out})
    graph.compile()
    np.testing.assert_allclose(expect - 1, 1 - traced_module(x), atol=1e-6)


def test_delete():
    traced_module, x, expect = _init_block()
    graph = traced_module.graph
    relu_expr = graph.get_function_by_type(F.relu).as_unique()
    node = relu_expr.outputs
    repl_node = relu_expr.inputs
    graph.replace_node({node[0]: repl_node[0]})
    graph.compile()
    np.testing.assert_allclose(expect - 1, F.relu(traced_module(x) - 1), atol=1e-6)

    # clear graph
    graph.replace_node({graph.outputs[0]: graph.inputs[1]})
    graph.compile()
    np.testing.assert_equal(len(list(graph._exprs)), 0)
    np.testing.assert_equal(traced_module(x).numpy(), x.numpy())


def test_flatten():
    traced_module, x, expect = _init_module()
    traced_module = traced_module.flatten()
    traced_module.graph.compile()
    assert all(not isinstance(i, GetAttr) for i in traced_module.graph._exprs)
    assert len(traced_module.graph._exprs) == 12
    np.testing.assert_equal(expect.numpy(), traced_module(x).numpy())


def test_id_and_name():
    def _check_id(traced_module):
        _total_ids = traced_module.graph._total_ids
        node_ids = [n._id for n in traced_module.graph.nodes().as_list()]
        assert len(set(node_ids)) == len(node_ids)
        assert max(node_ids) + 1 == len(node_ids)

        expr_ids = [n._id for n in traced_module.graph.exprs().as_list()]
        assert len(set(expr_ids)) == len(expr_ids)
        assert max(expr_ids) + 1 == _total_ids[1]

    def _check_name(flatened_module):
        node_names = [n._name for n in flatened_module.graph.nodes().as_list()]
        assert len(set(node_names)) == len(node_names)

    traced_module, x, expect = _init_module()
    _check_id(traced_module)

    flattened_module = traced_module.flatten()
    _check_id(flattened_module)
    _check_name(flattened_module)

    # pickle check
    obj = pickle.dumps(traced_module)
    traced_module = pickle.loads(obj)
    Node._set_next_id(159)
    Expr._set_next_id(1024)

    graph = traced_module.graph
    for expr in graph.get_function_by_type(F.relu).as_list():
        relu_out = expr.outputs[0]
        cur_graph = expr.top_graph
        with cur_graph.insert_exprs():
            neg_out = F.neg(relu_out)
        cur_graph.replace_node({relu_out: neg_out})
        cur_graph.compile()
    _check_id(traced_module)

    flattened_module = traced_module.flatten()
    _check_id(flattened_module)
    _check_name(flattened_module)

    # check trace TracedModule
    obj = pickle.dumps(traced_module)
    traced_module = pickle.loads(obj)
    module = NewModule(traced_module)
    traced_module = trace_module(module, x)
    _check_id(traced_module)

    flattened_module = traced_module.flatten()
    _check_id(flattened_module)
    _check_name(flattened_module)


def test_set_name():
    traced_module, x, expect = _init_module()
    graph = traced_module.graph
    output_node = graph.outputs[0]

    def rename(name):
        output_node.name = name

    np.testing.assert_raises(AssertionError, rename, "block1_out")
    rename("output")
    np.testing.assert_equal(str(graph.outputs[0]), "output")


def test_extra_block():
    class PostProcess(M.Module):
        def forward(self, x):
            return x * 2

    class Net(M.Module):
        def __init__(self, traced_module):
            super().__init__()
            self.post_process = PostProcess()
            self.traced_module = traced_module

        def forward(self, x):
            x = self.traced_module(x)
            x = self.post_process(x)
            return x

    traced_module, x, expect = _init_block()
    module = Net(traced_module)
    np.testing.assert_allclose(2 * expect, module(x), atol=1e-6)
    traced_module = trace_module(module, x)
    np.testing.assert_allclose(2 * expect, traced_module(x), atol=1e-6)
