# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.


import collections
from typing import List

from ...core._imperative_rt import OpDef
from ...core._imperative_rt.core2 import Tensor as RawTensor
from ...core._imperative_rt.core2 import apply, set_module_tracing, unset_module_tracing
from ...core.ops.special import Const
from ...tensor import Tensor
from .module_tracer import active_module_tracer
from .node import ModuleNode, Node, NodeMixin, TensorNode


class Expr:
    """
    ``Expr`` represents the operations(i.e. Call, Apply, GetAttr, Input, Constant) on ``Node``.
    """

    inputs = None  # type: List[Node]
    outputs = None  # type: List[Node]


# expr: None (i.e. fake expression which is used to mark input)
class Input(Expr):
    name = None

    def __init__(self, name=None, type=None):
        self.inputs = []
        node_cls = type if type else Node
        self.outputs = [
            node_cls(self, name=name),
        ]
        self.name = name

    @classmethod
    def make(cls, *args, **kwargs):
        expr = cls(*args, **kwargs)
        active_module_tracer().current_scope().add_input(expr.outputs[0])
        return expr.outputs[0]

    def __repr__(self):
        return "{} = Input({})".format(self.outputs[0], self.name)


# expr: outputs = getattr(inputs[0], self.name)
class GetAttr(Expr):
    name = None

    def __init__(self, module, name, type=None):
        assert isinstance(module, ModuleNode)
        self.inputs = [
            module,
        ]
        self.name = name
        node_cls = type if type else Node
        self.outputs = [
            node_cls(self),
        ]

    @classmethod
    def make(cls, *args, **kwargs):
        expr = cls(*args, **kwargs)
        active_module_tracer().current_scope().insert(expr)
        expr.outputs[0]._name = expr.name
        return expr.outputs[0]

    def interpret(self, *inputs):
        return (getattr(inputs[0], self.name),)

    def __repr__(self):
        return '{} = GetAttr({}, "{}")'.format(
            self.outputs[0], self.inputs[0], self.name
        )


# expr: outputs = inputs[0].__call__(*inputs[1:])
class Call(Expr):
    def __init__(self, module):
        assert isinstance(module, ModuleNode)
        self.inputs = [
            module,
        ]

    def add_input(self, node):
        self.inputs.append(node)

    def add_outputs(self, references):
        self.outputs = []
        if not isinstance(references, collections.Sequence):
            references = (references,)

        for i in references:
            self.outputs.append(NodeMixin.get_wrapped_type(i)(self))

    @classmethod
    def make(cls, *args, **kwargs):
        expr = cls(*args, **kwargs)
        active_module_tracer().current_scope().insert(expr)
        return expr

    def interpret(self, *inputs):
        mod = inputs[0]
        args = inputs[1:]
        outputs = mod(*args)
        if isinstance(outputs, RawTensor):
            outputs = (outputs,)
        return outputs

    def __repr__(self):
        return "{} = Call({})({})".format(
            ", ".join(str(i) for i in self.outputs),
            self.inputs[0],
            ", ".join(str(i) for i in self.inputs[1:]),
        )


# expr: outputs = apply(self.opdef, *inputs)
class Apply(Expr):
    opdef = None

    def __init__(self, opdef):
        assert isinstance(opdef, OpDef)
        self.opdef = opdef
        self.inputs = []

    def add_input(self, node):
        self.inputs.append(node)

    def add_outputs(self, references):
        self.outputs = []
        if not isinstance(references, collections.Sequence):
            references = (references,)

        for i in references:
            self.outputs.append(NodeMixin.get_wrapped_type(i)(self))

    @classmethod
    def make(cls, *args, **kwargs):
        expr = cls(*args, **kwargs)
        active_module_tracer().current_scope().insert(expr)
        return expr

    def interpret(self, *inputs):
        return apply(self.opdef, *inputs)

    def __repr__(self):
        return "{} = {}({})".format(
            ", ".join(str(i) for i in self.outputs),
            self.opdef,
            ", ".join(str(i) for i in self.inputs),
        )

    @classmethod
    def apply_module_trace_hook(cls, opdef, *inputs):
        for i in inputs:
            node = NodeMixin.get(i, None)
            if node is None:  # capture as constant
                NodeMixin.wrap_safe(i, Constant.make(i))
        apply_node = cls.make(opdef)
        for i in inputs:
            apply_node.add_input(NodeMixin.get(i))

        unset_module_tracing()
        outputs = apply(opdef, *inputs)
        set_module_tracing()

        apply_node.add_outputs(outputs)
        for n, v in zip(apply_node.outputs, outputs):
            NodeMixin.wrap_safe(v, n)
        return list(outputs)


# expr outputs = self.value
class Constant(Expr):
    value = None
    # TODO: constant cache to reduce the size of dumped model
    _constant_cache = {}

    def __init__(self, c):
        # TODO: type check, since not all types should be captured as constant
        self.value = c
        self.inputs = []
        node_cls = NodeMixin.get_wrapped_type(c)
        self.outputs = [
            node_cls(self),
        ]

    @classmethod
    def make(cls, *args, **kwargs):
        expr = cls(*args, **kwargs)
        active_module_tracer().current_scope().insert(expr)
        return expr.outputs[0]

    def interpret(self, *inputs):
        if isinstance(self.value, RawTensor):
            return Const(self.value.numpy())()
        return (self.value,)

    def __repr__(self):
        return "{} = Constant({})".format(self.outputs[0], self.value)

    def __getstate__(self):
        state = self.__dict__.copy()
        if isinstance(self.value, RawTensor):
            state["value"] = Tensor(self.value)
        return state
