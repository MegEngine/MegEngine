# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

import megengine.functional as F
import megengine.module as M
from megengine.experimental.traced_module import trace_module
from megengine.experimental.traced_module.expr import CallFunction, GetAttr


class MyBlock(M.Module):
    def __init__(self, in_channels=3, channels=3):
        super(MyBlock, self).__init__()
        self.conv1 = M.Conv2d(in_channels, channels, 3, 1, padding=1, bias=False)
        self.bn1 = M.BatchNorm2d(channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x) + 1
        return x


class MyModule(M.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.block0 = MyBlock()
        self.block1 = MyBlock()

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
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
    relu_expr = graph.get_call_function(F.relu).as_unique()
    assert isinstance(relu_expr, CallFunction) and relu_expr.func == F.relu


def test_insert():
    traced_module, x, expect = _init_block()
    graph = traced_module.graph
    relu_node = graph.get_call_function(F.relu).as_unique().outputs
    neg_node = graph.insert_call_function(F.neg, relu_node)
    graph.replace_node({relu_node[0]: neg_node[0]})
    graph.compile()
    np.testing.assert_allclose(expect - 1, 1 - traced_module(x), atol=1e-6)


def test_delete():
    traced_module, x, expect = _init_block()
    graph = traced_module.graph
    relu_expr = graph.get_call_function(F.relu).as_unique()
    node = relu_expr.outputs
    repl_node = relu_expr.inputs
    graph.replace_node({node[0]: repl_node[0]})
    graph.compile()
    np.testing.assert_allclose(expect - 1, F.relu(traced_module(x) - 1), atol=1e-6)


def test_flatten():
    traced_module, x, expect = _init_module()
    traced_module = traced_module.flatten()
    traced_module.graph.compile()
    assert all(not isinstance(i, GetAttr) for i in traced_module.graph._exprs)
    assert len(traced_module.graph._exprs) == 12


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
