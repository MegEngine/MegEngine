# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import collections
import copy
import functools
from inspect import getmembers, isclass, ismethod
from typing import Dict, List, Type

from ... import module as M
from ...core._imperative_rt.core2 import Tensor as RawTensor
from ...core._imperative_rt.core2 import (
    is_tracing_module,
    set_module_tracing,
    unset_module_tracing,
)
from ...core.tensor.array_method import ArrayMethodMixin
from ...module import Module
from ...tensor import Tensor
from .expr import Apply, CallFunction, CallMethod, Constant, Expr, GetAttr, Input
from .module_tracer import (
    Patcher,
    active_module_tracer,
    module_tracer,
    set_active_module_tracer,
)
from .node import ModuleNode, Node, NodeMixin, TensorNode
from .pytree import tree_flatten


def _leaf_type(node):
    if isinstance(node, RawTensor):
        return (Tensor, TensorNode)
    elif isinstance(node, (NodeMixin, Module)):
        return (Module, ModuleNode, NodeMixin)
    else:
        return type(node)


class InternalGraph:
    """
    ``InternalGraph`` is a graph consist of ``Node`` and  ``Expr``, it is used to represent the execution procedure of Module's forward method.

    Attributes:
    _exprs: List of Exprs in order of execution
    _inputs: Input Nodes of InternalGraph
    _outputs: Output Nodes of InternalGraph
    """

    _exprs = None  # type: List[Expr]
    _inputs = None  # type: List[Node]
    _outputs = None  # type: List[Node]

    def __init__(self):
        self._exprs = []
        self._inputs = []
        self._outputs = []

    def insert(self, expr):
        self._exprs.append(expr)

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    def add_input(self, i):
        self._inputs.append(i)

    def add_output(self, o):
        self._outputs.append(o)

    def interpret(self, *inputs):
        # TODO: support kwargs ?
        # TODO: skip expressions which are independent and have no side effect
        node2value = {}
        for n, v in zip(self._inputs, inputs):
            node2value[n] = v
        for expr in self._exprs:
            values = expr.interpret(*list(node2value[i] for i in expr.inputs))
            for n, v in zip(expr.outputs, values):
                node2value[n] = v
        return list(node2value[i] for i in self._outputs)

    def __repr__(self):
        return "InternalGraph ({}) {{\n\t{}\n\treturn {}\n}}".format(
            ", ".join(str(i) for i in self._inputs),
            "\n\t".join(str(i) for i in self._exprs),
            ", ".join(str(i) for i in self._outputs),
        )


def _get_meth_name(obj, func):
    for cls in type(obj).mro():
        for k, v in cls.__dict__.items():
            if v == func:
                return k
    return None


def _wrapped_function(orig_func):
    @functools.wraps(orig_func)
    def wrapped_fn(*args, **kwargs):
        if is_tracing_module():
            unset_module_tracing()
            inputs, tree_def = tree_flatten((args, kwargs), leaf_type=_leaf_type)
            for i in inputs:
                if not NodeMixin.get(i, None):
                    if isinstance(i, (RawTensor, NodeMixin)):
                        NodeMixin.wrap_safe(i, Constant.make(i))
            meth_name = _get_meth_name(args[0], wrapped_fn)
            if meth_name:
                self = inputs[0]
                call_node = CallMethod.make(NodeMixin.get(self), meth_name)
            else:
                call_node = CallFunction.make(orig_func)

            call_node.add_inputs(inputs)

            call_node.arg_def = tree_def
            outputs = orig_func(*args, **kwargs)
            call_node.add_outputs(outputs)
            set_module_tracing()
            return outputs
        return orig_func(*args, **kwargs)

    return wrapped_fn


class TracedModuleBuilder(NodeMixin):

    _mod = None  # type: Module
    _body = None  # type: InternalGraph
    _is_builtin = None  # type: bool
    _arg_def = None  # type: TreeDef
    __builder_attributes__ = [
        "_mod",
        "_body",
        "_NodeMixin__node",
        "_is_builtin",
        "_is_traced",
        "_arg_def" "build",
    ]

    def __init__(self, mod):
        super(TracedModuleBuilder, self).__init__()
        self._mod = mod
        self._body = InternalGraph()
        self._is_traced = False
        self._is_builtin = module_tracer.is_builtin(mod)

    def build(self):
        if self._is_builtin:
            node = NodeMixin.get(self)
            node.module_type = type(self._mod)
            return self._mod
        else:
            node = NodeMixin.get(self)
            node.graph = self._body
            node.attr_type_map = {}
            node.arg_def = self._arg_def
            traced_module = TracedModule(node)
            for k, v in self.__dict__.items():
                if k not in TracedModuleBuilder.__builder_attributes__:
                    if isinstance(v, TracedModuleBuilder):
                        v = v.build()
                    setattr(traced_module, k, v)
                    traced_module.m_node.attr_type_map[k] = type(v)
            return traced_module

    def __call__(self, *args, **kwargs):
        assert isinstance(self._mod, Module)
        for arg in args:
            assert isinstance(arg, RawTensor)

        for k, v in kwargs.items():
            assert isinstance(v, RawTensor)
        # prepare args and kwargs for inner graph
        def mark_constant(x):
            node = NodeMixin.get(x, None)
            if node is None:  # capture as constant
                NodeMixin.wrap(x, lambda: Constant.make(x))

        inputs, tree_def = tree_flatten(((self, *args), kwargs), leaf_type=_leaf_type)
        if self._arg_def is None:
            self._arg_def = tree_def
        assert self._arg_def == tree_def
        for i in inputs:
            mark_constant(i)
        callnode = CallMethod.make(NodeMixin.get(self))

        callnode.add_inputs(inputs)

        callnode.arg_def = tree_def

        if self._is_builtin or self._is_traced:
            unset_module_tracing()
            outputs = self._mod(*args, **kwargs)
            set_module_tracing()
            if self._is_builtin:
                self._body = None
        else:
            active_module_tracer().push_scope(self._body)
            # rebind self to new input node
            orig_self = NodeMixin.get(self)
            NodeMixin.wrap_safe(
                self, Input.make("self", NodeMixin.get_wrapped_type(self))
            )
            # prepare args and kwargs for inner graph
            def wrap(x):
                wrapped = copy.copy(x)  # FIXME
                NodeMixin.wrap(
                    wrapped,
                    lambda: Input.make(type=NodeMixin.get_wrapped_type(wrapped)),
                )
                return wrapped

            args = [self]
            for i in inputs[1:]:
                args.append(wrap(i))
            args, kwargs = tree_def.unflatten(args)
            active_module_tracer().patcher.auto_patch(
                getattr(getattr(self._mod, "forward", self._mod), "__globals__", {})
            )
            outputs = type(self._mod).forward(*args, **kwargs)

            for i in (
                outputs if isinstance(outputs, collections.abc.Sequence) else (outputs,)
            ):
                active_module_tracer().current_scope().add_output(NodeMixin.get(i))

            NodeMixin.wrap_safe(self, orig_self)
            self._is_traced = True
            active_module_tracer().pop_scope()

        # rebind output to outer graph
        callnode.add_outputs(outputs)
        return outputs

    def __getattr__(self, name):
        if name not in self._mod.__dict__:
            attr = getattr(type(self._mod), name).__get__(self, type(self))
        else:
            attr = getattr(self._mod, name)
            if isinstance(attr, Module):
                attr = TracedModuleBuilder(attr)
            setattr(self, name, attr)
            NodeMixin.wrap(
                attr,
                lambda: GetAttr.make(
                    NodeMixin.get(self), name, type=NodeMixin.get_wrapped_type(attr)
                ),
            )
        return attr

    def __getattribute__(self, name):
        if name in TracedModuleBuilder.__builder_attributes__:
            return super().__getattribute__(name)
        else:
            wrapped = super().__getattribute__(name)
            if name in self._mod.__dict__ and not NodeMixin.get(wrapped, None):
                assert not self._is_builtin
                NodeMixin.wrap(
                    wrapped,
                    lambda: GetAttr.make(
                        NodeMixin.get(self),
                        name,
                        type=NodeMixin.get_wrapped_type(wrapped),
                    ),
                )
            return wrapped


class _expr_list:
    def __init__(self, module: "TracedModule"):
        self.module = module

    def __iter__(self):
        graph = self.module.m_node.graph
        for expr in graph._exprs:
            if isinstance(expr, CallMethod) and isinstance(expr.inputs[0], ModuleNode):
                yield expr
                assert isinstance(expr.inputs[0].expr, GetAttr)
                (obj,) = expr.inputs[0].expr.interpret(self.module)
                if isinstance(obj, TracedModule):
                    yield from obj.exprs
            yield expr


class TracedModule(Module):
    """
    `TracedModule` is the Module created by tracing normal module. It owns a ModuleNode(m_node), and will interpret the m_node.graph when it is called.
    """

    m_node = None  # type: ModuleNode

    def __init__(self, node):
        super(TracedModule, self).__init__()
        self.m_node = node

    def forward(self, *args, **kwargs):
        inputs, treedef = tree_flatten(((self, *args), kwargs), leaf_type=_leaf_type)
        assert treedef == self.m_node.arg_def
        rst = self.m_node.graph.interpret(*inputs)
        if len(rst) == 1:
            rst = rst[0]
        return rst

    @property
    def exprs(self):
        """
        Get all ``Expr`` s recursively.

        :return: Iterator[Expr]
        """
        return _expr_list(self)

    def flatten(self):
        """
        Get a new module, which eliminates ``GetAttr`` and has no hierarchy.

        :return: :class:`TracedModule`
        """
        new_module = copy.deepcopy(self)

        def _flatten_submodule(module, call=None):
            if not isinstance(module, TracedModule):
                call.inputs[0] = module
                return (call,)

            exprs = []

            graph = module.m_node.graph
            for expr in graph._exprs:

                # replace inputs for submodule's expr
                for idx, inp in enumerate(expr.inputs):
                    if call and inp in graph._inputs:
                        expr.inputs[idx] = call.inputs[idx]
                # replace outputs for submodule's expr
                for idx, outp in enumerate(expr.outputs):
                    if call and outp in graph._outputs:
                        expr.outputs[idx] = call.outputs[idx]

                if isinstance(expr, GetAttr):
                    # replace GetAttr with Constant
                    if isinstance(expr.outputs[0], TensorNode):
                        const = Constant(getattr(module, expr.name))
                        const.outputs = expr.outputs
                        exprs.append(const)
                elif isinstance(expr, CallMethod):
                    obj_node = expr.inputs[0]
                    if isinstance(obj_node, ModuleNode):
                        assert isinstance(expr.inputs[0].expr, GetAttr)
                        (obj,) = expr.inputs[0].expr.interpret(module)
                        exprs.extend(_flatten_submodule(obj, expr))
                    else:
                        exprs.append(expr)
                else:
                    exprs.append(expr)

            return exprs

        new_module.m_node.graph._exprs = _flatten_submodule(new_module)

        return new_module

    def __getstate__(self):
        d = self.__dict__
        for k in Module.__dict__:
            d.pop(k, None)
        return d


def cpp_apply_module_trace(opdef, *args):
    return Apply.apply_module_trace_hook(opdef, *args)


def register_as_builtin(mod_cls: Type[Module]) -> None:
    """
    Registers class ``mod_cls`` (subclass of megengine.module.Module) as builtin module.

    param mod_cls: the Module class which will be threated as builtin module in tracing
    """
    module_tracer.register_as_builtin(mod_cls)


def _register_all_builtin_module():

    for sub_mod in [M, M.qat, M.quantized]:
        for m in getmembers(sub_mod):
            if (
                isclass(m[1])
                and issubclass(m[1], M.Module)
                and m[1] is not M.Sequential
            ):
                module_tracer.register_as_builtin(m[1])


def trace_module(mod: Module, *args: Tensor, **kwargs: Tensor) -> TracedModule:
    """
    Traces module ``mod`` and returns corresponding TracedModule.

    param mod: the module will be converted to TracedModule
    param input: the positional arguments passed to forward method of ``mod``
    param kwargs: the keyword arguments passed to forward method of ``mod``
    """
    assert active_module_tracer() is None
    try:
        set_module_tracing()
        set_active_module_tracer(module_tracer(_wrapped_function))
        with active_module_tracer().patcher:
            global_scope = InternalGraph()
            active_module_tracer().push_scope(global_scope)

            builder = TracedModuleBuilder(mod)
            NodeMixin.wrap_safe(builder, Input.make("TopModule", ModuleNode))
            inputs, _ = tree_flatten((args, kwargs))
            for _, i in enumerate(inputs):
                NodeMixin.wrap_safe(
                    i, Input.make("arg_{}".format(_), NodeMixin.get_wrapped_type(i))
                )
            builder(*args, **kwargs)
            active_module_tracer().pop_scope()
            return builder.build()
    finally:
        set_active_module_tracer(None)
        unset_module_tracing()
