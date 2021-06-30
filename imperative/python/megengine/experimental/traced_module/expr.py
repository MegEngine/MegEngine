# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.


import collections
from typing import Callable, List

from ...core._imperative_rt import OpDef
from ...core._imperative_rt.core2 import Tensor as RawTensor
from ...core._imperative_rt.core2 import apply, set_module_tracing, unset_module_tracing
from ...core.ops.special import Const
from ...module import Module
from ...tensor import Tensor
from .module_tracer import active_module_tracer
from .node import ModuleNode, Node, NodeMixin, TensorNode


class Expr:
    """
    ``Expr`` represents the operations(i.e. CallMethod, CallFunction, Apply, GetAttr, Input, Constant) on ``Node``.
    """

    inputs = None  # type: List[Node]
    outputs = None  # type: List[Node]

    def add_input(self, node):
        self.inputs.append(node)

    def add_outputs(self, outputs):
        self.outputs = []
        if not isinstance(outputs, collections.Sequence):
            outputs = (outputs,)

        for i in outputs:
            self.outputs.append(NodeMixin.get_wrapped_type(i)(self))

        for i, node in zip(outputs, self.outputs,):
            NodeMixin.wrap_safe(i, node)

    @classmethod
    def get_args_node(cls, arg):
        """
        Create nodes by ``arg``, which may be a container.
        Return the same structure with arg.

        If ``arg`` was not Tensor or Module, it will be stored as const.

        :param arg: tensor, module or const.
        """
        if isinstance(arg, (RawTensor, Module)):
            if not NodeMixin.get(arg, None):
                NodeMixin.wrap_safe(arg, Constant.make(arg))
            return NodeMixin.get(arg)
        elif isinstance(arg, collections.abc.Sequence):
            seq_cls = type(arg)
            return seq_cls([Expr.get_args_node(a) for a in arg])
        else:
            # TODO: assert arg type
            return arg  # as const

    @classmethod
    def get_arg_value(cls, inp_node, node2value):
        """
        Get values from node2value by inp_node, which may be a container.
        Return the same structure with inp_node.

        If ``inp_node`` was not in node2value, it is a const.

        :param inp_node: nodes.
        :param node2value: dict from node to tensor and module.
        """
        if inp_node in node2value:
            return node2value[inp_node]
        elif isinstance(inp_node, collections.abc.Sequence):
            seq_cls = type(inp_node)
            return seq_cls([Expr.get_arg_value(i, node2value) for i in inp_node])
        else:
            return inp_node


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
        self.method = method
        self.arg_names = []
        self.kwargs = {}  # const kwargs

    def add_input(self, node, arg_name=None):
        if arg_name == "self":  # FIXME: <XP>
            return
        self.inputs.append(node)
        if arg_name is not None:
            self.arg_names.append(arg_name)

    @classmethod
    def make(cls, *args, **kwargs):
        expr = cls(*args, **kwargs)
        active_module_tracer().current_scope().insert(expr)
        return expr

    def interpret(self, *inputs):
        mod = inputs[0]
        args = inputs[1:]
        outputs = getattr(mod, self.method)(*args, **self.kwargs)
        if isinstance(outputs, RawTensor):
            outputs = (outputs,)
        return outputs

    def __repr__(self):
        return "{} = CallMethod({}, {})({})".format(
            ", ".join(str(i) for i in self.outputs),
            self.inputs[0],
            self.method,
            ", ".join(str(i) for i in self.inputs[1:]),
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
            apply_node.add_input(NodeMixin.get(i))

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
        self.inputs = []
        self.arg_names = []
        self.kwargs = {}  # const kwargs

    def add_input(self, node, arg_name):
        self.inputs.append(node)
        self.arg_names.append(arg_name)

    @classmethod
    def make(cls, *args, **kwargs):
        expr = cls(*args, **kwargs)
        active_module_tracer().current_scope().insert(expr)
        return expr

    def interpret(self, *inputs):
        inp_dict = dict([(name, node) for node, name in zip(inputs, self.arg_names)])
        outputs = self.func(**inp_dict, **self.kwargs)
        outputs = (
            outputs if isinstance(outputs, collections.abc.Sequence) else (outputs,)
        )
        return outputs

    def __repr__(self):
        return "{} = {}({})".format(
            ", ".join(str(i) for i in self.outputs),
            self.func.__module__ + "." + self.func.__name__,
            ", ".join(str(i) for i in self.inputs),
        )


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
