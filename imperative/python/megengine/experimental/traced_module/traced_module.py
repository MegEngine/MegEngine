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
from typing import Callable, Dict, Iterable, List, Sequence, Type

import numpy as np
from numpy.lib.arraysetops import isin

from ... import functional as F
from ... import get_logger
from ... import module as M
from ...core._imperative_rt.core2 import Tensor as RawTensor
from ...core._imperative_rt.core2 import (
    is_tracing_module,
    set_module_tracing,
    unset_module_tracing,
)
from ...core._trace_option import set_symbolic_shape
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

logger = get_logger(__name__)


def _leaf_type(node):
    if isinstance(node, RawTensor):
        return (Tensor, TensorNode)
    elif isinstance(node, (NodeMixin, Module)):
        return (Module, ModuleNode, NodeMixin)
    else:
        return type(node)


def _is_leaf(node):
    assert isinstance(node, RawTensor), type(node)
    return isinstance(node, RawTensor)


def _is_const_leaf(node):
    if isinstance(node, (RawTensor, NodeMixin, Module)):
        return False
    return True


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

    @property
    def exprs(self):
        return ExprFilter(_expr_iter(self))

    def get_call_function(self, func: Callable = None):
        return self.exprs.call_function(func)

    def get_call_method(self, method: str = None):
        return self.exprs.call_method(method)

    def add_input(self, i):
        self._inputs.append(i)

    def add_output(self, o):
        self._outputs.append(o)

    def get_dep_exprs(self, nodes: Sequence[Node]) -> List[Expr]:
        if not isinstance(nodes, Sequence):
            nodes = (nodes,)
        ret = list()
        queue = list(nodes)
        while queue:
            node = queue.pop()
            expr = node.expr
            if expr not in ret:
                ret.append(expr)
            for i in expr.inputs:
                if i not in queue:
                    queue.append(i)
        return ret

    def insert_call_function(self, func: Callable, nodes: Sequence[Node]):
        if not isinstance(nodes, Sequence):
            nodes = [nodes]
        assert isinstance(func, Callable)
        for i in nodes:
            assert isinstance(
                i, TensorNode
            ), "CallFunction only accept TensorNode as inputs"

        expr = CallFunction(func)
        expr.inputs = nodes

        for i in nodes:
            i.users.append(expr)

        idx = max(self._exprs.index(i.expr) for i in nodes) + 1
        self._exprs.insert(idx, expr)

        fake_inp_val = tuple(F.zeros(shape=i.shape, dtype=i.dtype) for i in nodes)
        fake_out_val = func(*fake_inp_val)

        def create_node(val: Tensor):
            node = TensorNode(expr)
            node.shape = val.shape
            node.dtype = val.dtype
            return node

        out_nodes = list(create_node(i) for i in fake_out_val)
        expr.outputs = out_nodes

        return out_nodes

    def insert_call_method(self, target, method, args):
        if not isinstance(args, Sequence):
            args = [args]
        assert isinstance(target, (TensorNode, ModuleNode))
        assert isinstance(method, str)
        for i in args:
            assert isinstance(i, TensorNode)

        expr = CallMethod(method)
        expr.inputs = [target, *args]

        if isinstance(target, TensorNode):
            fake_target_val = F.zeros(shape=target.shape, dtype=target.dtype)
            fake_inp_val = tuple(F.zeros(shape=i.shape, dtype=i.dtype) for i in args)
            fake_out_val = getattr(fake_target_val, method)(fake_inp_val)

            def create_node(val: Tensor):
                node = TensorNode(expr)
                node.shape = val.shape
                node.dtype = val.dtype
                return node

            out_nodes = list(create_node(i) for i in fake_out_val)
            expr.outputs = out_nodes
        else:
            raise NotImplementedError()

        return out_nodes

    def replace_node(self, repl_dict: Dict[Node, Node]):
        while repl_dict:
            node, repl_node = repl_dict.popitem()
            # check graph inputs and outputs
            assert node not in self.inputs, "Cannot replace inputs"
            for i, n in enumerate(self.outputs):
                if n is node:
                    self.outputs[i] = repl_node
            # update users of node and repl_node
            # update inputs of expr in node.users
            dep_exprs = self.get_dep_exprs(repl_node)
            i = 0
            while i < len(node.users):
                n = node.users[i]
                if n in dep_exprs:
                    logger.info("Find a loop: ignore this replacement once")
                    logger.info("node: %s" % node.__repr__())
                    logger.info("repl_node: %s" % repl_node.__repr__())
                    i += 1
                    continue
                repl_node.users.append(n)
                node.users.pop(i)
                idx = n.inputs.index(node)
                n.inputs[idx] = repl_node

    def compile(self):
        """
        Delete unused expr.
        """
        dep_exprs = self.get_dep_exprs(self.outputs)
        i = 0
        while i < len(self._exprs):
            expr = self._exprs[i]
            if expr in dep_exprs:
                i += 1
                continue
            for n in expr.inputs:
                n.users.remove(expr)
            self._exprs.remove(expr)

    def interpret(self, *inputs):
        node2value = {}
        for n, v in zip(self._inputs, inputs):
            node2value[n] = v
        for expr in self._exprs:
            values = expr.interpret(*list(node2value[i] for i in expr.inputs))
            if values is not None:
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
    tp = obj if isinstance(obj, type) else type(obj)
    for cls in tp.mro():
        for k, v in cls.__dict__.items():
            if v == func:
                return k
    return None


def _wrapped_function(orig_func):
    @functools.wraps(orig_func)
    def wrapped_fn(*args, **kwargs):
        if is_tracing_module():
            unset_module_tracing()
            inputs, tree_def = tree_flatten(
                (args, kwargs), leaf_type=_leaf_type, is_const_leaf=_is_const_leaf
            )
            for i in inputs:
                if not NodeMixin.get(i, None):
                    if isinstance(i, (RawTensor, NodeMixin)):
                        NodeMixin.wrap_safe(i, Constant.make(i))
            meth_name = _get_meth_name(args[0], wrapped_fn)
            if meth_name:
                self = inputs[0]
                if meth_name == "__new__":
                    if all([not isinstance(i, RawTensor) for i in inputs]):
                        # only trace Tensor.__new__() when there are tensors in args
                        set_module_tracing()
                        return orig_func(*args, **kwargs)
                    if isinstance(args[1], RawTensor):
                        node = NodeMixin.get(inputs[1])
                        inputs[1] = copy.copy(inputs[1])
                        # copy inputs[1] to avoid tensor and Tensor(tensor) share same m_tensor, which will cause they have same _NodeMixin__node in tracing.
                        NodeMixin.wrap_safe(inputs[1], node)
                        args, kwargs = tree_def.unflatten(inputs)
                    call_node = CallMethod.make(self, meth_name)
                else:
                    call_node = CallMethod.make(NodeMixin.get(self), meth_name)
                call_node.add_inputs(inputs[1:])
            else:
                call_node = CallFunction.make(orig_func)
                call_node.add_inputs(inputs)

            call_node.arg_def = tree_def
            outputs = orig_func(*args, **kwargs)
            if meth_name == "__new__":
                call_node.add_outputs(outputs, False)
            else:
                call_node.add_outputs(outputs)
            set_module_tracing()
            return outputs
        return orig_func(*args, **kwargs)

    return wrapped_fn


class TracedModuleBuilder(NodeMixin):

    _mod = None  # type: Module
    _body = None  # type: InternalGraph
    _is_builtin = None  # type: bool
    __builder_attributes__ = [
        "_mod",
        "_body",
        "_NodeMixin__node",
        "_is_builtin",
        "build",
    ]

    def __init__(self, mod, is_top_module=False):
        super(TracedModuleBuilder, self).__init__()
        self._mod = mod
        self._body = None
        self._is_builtin = module_tracer.is_builtin(mod)

    def build(self):
        if self._is_builtin:
            node = NodeMixin.get(self)
            node.module_type = type(self._mod)
            return self._mod
        else:
            node = NodeMixin.get(self)
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
        # prepare args and kwargs for inner graph
        def mark_constant(x):
            node = NodeMixin.get(x, None)
            if node is None:  # capture as constant
                NodeMixin.wrap(x, lambda: Constant.make(x))

        inputs, tree_def = tree_flatten(
            ((self, *args), kwargs), leaf_type=_leaf_type, is_const_leaf=_is_const_leaf
        )
        for i in inputs:
            mark_constant(i)
        callnode = CallMethod.make(NodeMixin.get(self))

        callnode.add_inputs(inputs[1:])

        callnode.arg_def = tree_def

        if self._is_builtin:
            unset_module_tracing()
            rst = self._mod(*args, **kwargs)
            outputs, out_def = tree_flatten(rst, leaf_type=_leaf_type, is_leaf=_is_leaf)
            set_module_tracing()
            if self._is_builtin:
                self._body = None
        else:
            self._body = InternalGraph()
            active_module_tracer().push_scope(self._body)
            # rebind self to new input node
            orig_self = NodeMixin.get(self)
            NodeMixin.wrap_safe(
                self, Input.make("self", NodeMixin.get_wrapped_type(self))
            )
            origin_inp_node = [NodeMixin.get(i, None) for i in inputs[1:]]
            # prepare args and kwargs for inner graph
            def wrap(x):
                NodeMixin.wrap(
                    x, lambda: Input.make(type=NodeMixin.get_wrapped_type(x)),
                )
                return x

            args = [self]
            for i in inputs[1:]:
                args.append(wrap(i))
            args, kwargs = tree_def.unflatten(args)
            active_module_tracer().patcher.auto_patch(
                getattr(getattr(self._mod, "forward", self._mod), "__globals__", {})
            )
            rst = type(self._mod).forward(*args, **kwargs)
            outputs, out_def = tree_flatten(rst, leaf_type=_leaf_type, is_leaf=_is_leaf)
            for i in (
                outputs if isinstance(outputs, collections.abc.Sequence) else (outputs,)
            ):
                active_module_tracer().current_scope().add_output(NodeMixin.get(i))

            NodeMixin.wrap_safe(self, orig_self)
            for arg, node in zip(inputs[1:], origin_inp_node):
                if node:
                    NodeMixin.wrap_safe(arg, node)
            active_module_tracer().pop_scope()

        # rebind output to outer graph
        callnode.add_outputs(outputs)
        self_node = NodeMixin.get(self)
        self_node.argdef_graph_map[callnode.arg_def] = self._body
        self_node.argdef_outdef_map[callnode.arg_def] = out_def
        return rst

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
            if name in self._mod.__dict__:
                if not NodeMixin.get(wrapped, None):
                    assert not self._is_builtin
                    NodeMixin.wrap(
                        wrapped,
                        lambda: GetAttr.make(
                            NodeMixin.get(self),
                            name,
                            type=NodeMixin.get_wrapped_type(wrapped),
                        ),
                    )
                else:
                    node = NodeMixin.get(wrapped)
                    expr = GetAttr.make(
                        NodeMixin.get(self),
                        name,
                        type=NodeMixin.get_wrapped_type(wrapped),
                    ).expr
                    expr.outputs[0] = node
            return wrapped


class _expr_iter:
    def __init__(self, graph: InternalGraph):
        self.graph = graph

    def __iter__(self):
        for expr in self.graph._exprs:
            if isinstance(expr, CallMethod) and isinstance(expr.inputs[0], ModuleNode):
                yield expr
                if expr.graph is not None:
                    yield from expr.graph.exprs
            else:
                yield expr


class ExprFilter:
    def __init__(self, expr_iter: Iterable):
        self._iter = expr_iter

    def __iter__(self):
        return iter(self._iter)

    def call_function(self, func):
        return ExprFilterCallFunction(self, func)

    def call_method(self, method):
        return ExprFilterCallMethod(self, method)

    def as_list(self):
        return list(self)

    def as_dict(self):
        raise NotImplementedError("need key")

    def as_unique(self):
        (expr,) = self
        return expr

    def as_count(self):
        return sum(1 for _ in self)


class ExprFilterCallFunction(ExprFilter):
    def __init__(self, expr_iter, func: Callable = None):
        super().__init__(expr_iter)
        self.func = func

    def __iter__(self):
        for i in self._iter:
            if not isinstance(i, CallFunction):
                continue
            if self.func is None or i.func == self.func:
                yield i


class ExprFilterCallMethod(ExprFilter):
    def __init__(self, expr_iter, method: str = None):
        super().__init__(expr_iter)
        self.method = method

    def __iter__(self):
        for i in self._iter:
            if not isinstance(i, CallMethod):
                continue
            if self.method is None or i.method == self.method:
                yield i


class TracedModule(Module):
    """
    `TracedModule` is the Module created by tracing normal module. It owns a ModuleNode(m_node). `TracedModule` can not be called directly. It can be 
    interpreted by CallMethod Expr.
    """

    m_node = None  # type: ModuleNode

    def __init__(self, node):
        super(TracedModule, self).__init__()
        self.m_node = node

    def forward(self, *args, **kwargs):
        inputs, treedef = tree_flatten(
            ((self, *args), kwargs), _leaf_type, is_const_leaf=_is_const_leaf
        )
        assert treedef in self.m_node.argdef_graph_map
        inputs = filter(
            lambda i: isinstance(i, (Module, TracedModuleBuilder, RawTensor)), inputs
        )  # allow TracedModuleBuilder for retrace.
        outputs = self.m_node.argdef_graph_map[treedef].interpret(*inputs)
        out_def = self.m_node.argdef_outdef_map[treedef]
        outputs = out_def.unflatten(outputs)
        return outputs

    @property
    def graph(self):
        assert len(self.m_node.argdef_graph_map) == 1
        return list(self.m_node.argdef_graph_map.values())[0]

    @property
    def exprs(self):
        return self.graph.exprs

    def flatten(self):
        """
        Get a new module, which eliminates ``GetAttr`` and has no hierarchy.

        :return: :class:`TracedModule`
        """
        new_module = copy.deepcopy(self)

        def _flatten_subgraph(graph, module, call=None):
            if graph is None:
                assert not isinstance(module, TracedModule)
                const = Constant(module)
                const.outputs[0] = call.inputs[0]
                const.outputs[0].expr = const
                return [const, call]
            exprs = []
            for expr in graph._exprs:
                # replace inputs for submodule's expr
                for idx, inp in enumerate(expr.inputs):
                    if call and inp in graph._inputs:
                        inp_idx = graph._inputs.index(inp)
                        expr.inputs[idx] = call.inputs[inp_idx]
                        call.inputs[inp_idx].users.append(expr)
                # replace outputs for submodule's expr
                for idx, outp in enumerate(expr.outputs):
                    if call and outp in graph._outputs:
                        oup_idx = graph._outputs.index(outp)
                        expr.outputs[idx] = call.outputs[oup_idx]
                        call.outputs[oup_idx].expr = expr

                if isinstance(expr, GetAttr):
                    # replace GetAttr with Constant
                    if isinstance(expr.outputs[0], TensorNode):
                        const = Constant(getattr(module, expr.name))
                        const.outputs = expr.outputs
                        const.outputs[0].expr = const
                        exprs.append(const)

                elif isinstance(expr, CallMethod):
                    obj_node = expr.inputs[0]
                    if isinstance(obj_node, ModuleNode):
                        pre_expr = expr.inputs[0].expr
                        if isinstance(pre_expr, GetAttr):
                            (obj,) = expr.inputs[0].expr.interpret(module)
                            exprs.extend(_flatten_subgraph(expr.graph, obj, expr))
                        else:
                            # module has been replaced.
                            assert isinstance(pre_expr, Constant)
                    else:
                        exprs.append(expr)
                else:
                    exprs.append(expr)

            if call is not None:
                for i in call.inputs:
                    i.users.remove(call)

            return exprs

        new_module.graph._exprs = _flatten_subgraph(new_module.graph, new_module)

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
        use_sym_shape = set_symbolic_shape(True)
        set_module_tracing()
        set_active_module_tracer(module_tracer(_wrapped_function))

        with active_module_tracer().patcher:
            global_scope = InternalGraph()
            active_module_tracer().push_scope(global_scope)

            builder = TracedModuleBuilder(mod, True)
            NodeMixin.wrap_safe(builder, Input.make("TopModule", ModuleNode))
            inputs, _ = tree_flatten((args, kwargs), is_const_leaf=_is_const_leaf)
            for _, i in enumerate(inputs):
                if isinstance(i, RawTensor):
                    NodeMixin.wrap_safe(
                        i, Input.make("arg_{}".format(_), NodeMixin.get_wrapped_type(i))
                    )
            builder(*args, **kwargs)
            active_module_tracer().pop_scope()
            return builder.build()
    finally:
        set_symbolic_shape(use_sym_shape)
        set_active_module_tracer(None)
        unset_module_tracing()
