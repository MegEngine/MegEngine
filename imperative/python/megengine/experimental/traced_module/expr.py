# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import builtins
import collections
from typing import Callable, List

from ...core._imperative_rt import OpDef
from ...core._imperative_rt.core2 import Tensor as RawTensor
from ...core._imperative_rt.core2 import apply, set_module_tracing, unset_module_tracing
from ...core.ops.special import Const
from ...module import Module
from ...tensor import Tensor
from .module_tracer import active_module_tracer, module_tracer
from .node import ModuleNode, Node, NodeMixin, TensorNode
from .pytree import TreeDef


class Expr:
    """
    ``Expr`` represents the operations(i.e. CallMethod, CallFunction, Apply, GetAttr, Input, Constant) on ``Node``.
    """

    inputs = None  # type: List[Node]
    outputs = None  # type: List[Node]
    const_val = None  # type: List[Any]
    arg_def = None  # type: TreeDef

    def add_inputs(self, vals):
        if not isinstance(vals, collections.abc.Sequence):
            vals = (vals,)
        for val in vals:
            node = NodeMixin.get(val, None)
            if isinstance(node, (TensorNode, ModuleNode)):
                if node not in self.inputs:
                    self.inputs.append(node)
            else:
                assert node is None
                assert type(val) in builtins.__dict__.values()
                idx = len(self.inputs) + len(self.const_val)
                self.const_val.append((idx, val))

    def add_outputs(self, outputs):
        self.outputs = []
        if not isinstance(outputs, collections.Sequence):
            outputs = (outputs,)

        for i in outputs:
            assert isinstance(i, RawTensor)
            self.outputs.append(NodeMixin.get_wrapped_type(i)(self))

        for i, node in zip(outputs, self.outputs,):
            NodeMixin.wrap_safe(i, node)

    def unflatten_args(self, inputs):
        if self.arg_def is not None:
            inputs = list(inputs)
            for idx, val in self.const_val:
                inputs.insert(idx, val)
            args, kwargs = self.arg_def.unflatten(inputs)
            return args, kwargs
        else:
            return inputs, {}

    @property
    def kwargs(self):
        _, kwargs = self.unflatten_args(self.inputs)
        return kwargs

    @property
    def args(self):
        args, _ = self.unflatten_args(self.inputs)
        return args


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
class CallMethod(Expr):
    def __init__(self, module, method="__call__"):
        assert isinstance(module, (TensorNode, ModuleNode))
        self.inputs = [
            module,
        ]
        self.const_val = []
        self.method = method

    @classmethod
    def make(cls, *args, **kwargs):
        expr = cls(*args, **kwargs)
        active_module_tracer().current_scope().insert(expr)
        return expr

    @property
    def graph(self):
        if isinstance(self.inputs[0], ModuleNode):
            m_node = self.inputs[0]
            if m_node.argdef_graph_map:
                assert self.arg_def in m_node.argdef_graph_map
                return m_node.argdef_graph_map[self.arg_def]
        return None

    def interpret(self, *inputs):
        args, kwargs = self.unflatten_args(inputs)
        obj = args[0]
        args = args[1:]
        outputs = getattr(obj, self.method)(*args, **kwargs)
        if isinstance(outputs, RawTensor):
            outputs = (outputs,)
        return outputs

    def __repr__(self):
        args = ", ".join(str(i) for i in self.args[1:])
        kwargs = ", ".join("{}={}".format(k, v) for k, v in self.kwargs.items())
        return "{} = {}.{}({})".format(
            ", ".join(str(i) for i in self.outputs),
            self.inputs[0],
            self.method,
            ", ".join([args, kwargs]),
        )


# expr: outputs = apply(self.opdef, *inputs)
class Apply(Expr):
    opdef = None

    def __init__(self, opdef):
        assert isinstance(opdef, OpDef)
        self.opdef = opdef
        self.inputs = []

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
            assert isinstance(i, RawTensor)
            apply_node.inputs.append(NodeMixin.get(i))

        unset_module_tracing()
        outputs = apply(opdef, *inputs)
        set_module_tracing()

        apply_node.add_outputs(outputs)
        for n, v in zip(apply_node.outputs, outputs):
            NodeMixin.wrap_safe(v, n)
        return list(outputs)


class CallFunction(Expr):
    def __init__(self, func):
        assert isinstance(func, Callable)
        self.func = func
        self.const_val = []
        self.inputs = []

    @classmethod
    def make(cls, *args, **kwargs):
        expr = cls(*args, **kwargs)
        active_module_tracer().current_scope().insert(expr)
        return expr

    def interpret(self, *inputs):
        args, kwargs = self.unflatten_args(inputs)
        outputs = self.func(*args, **kwargs)
        outputs = (
            outputs if isinstance(outputs, collections.abc.Sequence) else (outputs,)
        )
        return outputs

    def __repr__(self):
        args = ", ".join(str(i) for i in self.args)
        kwargs = ", ".join("{}={}".format(k, v) for k, v in self.kwargs.items())
        return "{} = {}({})".format(
            ", ".join(str(i) for i in self.outputs),
            self.func.__module__ + "." + self.func.__name__,
            ", ".join([args, kwargs]),
        )


# expr outputs = self.value
class Constant(Expr):
    value = None
    # TODO: constant cache to reduce the size of dumped model
    _constant_cache = {}

    def __init__(self, c):
        assert isinstance(c, (RawTensor, Module))
        if isinstance(c, Module):
            assert module_tracer.is_builtin(c)
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
