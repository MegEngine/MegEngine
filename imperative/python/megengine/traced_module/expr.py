# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import builtins
import collections
import copy
import inspect
import re
from typing import Callable, Dict, List

from ..core._imperative_rt import OpDef
from ..core._imperative_rt.core2 import Tensor as RawTensor
from ..core._imperative_rt.core2 import apply, set_module_tracing, unset_module_tracing
from ..core.ops.builtin import FakeQuant
from ..core.ops.special import Const
from ..module import Module
from ..tensor import Parameter, Tensor
from .module_tracer import active_module_tracer, module_tracer
from .node import ModuleNode, Node, NodeMixin, TensorNode
from .pytree import ArgsIndex, TreeDef, _is_const_leaf, _is_leaf, tree_flatten
from .serialization import get_opdef_state, load_opdef_from_state


def rstrip(s: str, __chars: str):
    __chars = re.escape(__chars)
    s = re.sub(r"^(?P<left>.*?)(?:%s)+$" % __chars, "\g<left>", s)
    return s


class Expr:
    r"""``Expr`` represents the operations (i.e. ``CallMethod``, ``CallFunction``, ``Apply``, 
    ``GetAttr``, ``Input``, ``Constant``) on ``Node``.
    """

    inputs = None  # type: List[Node]
    r"""The input Nodes of this Expr."""
    outputs = None  # type: List[Node]
    r"""The output Nodes of this Expr."""
    const_val = None  # type: List[Any]
    r"""The non-tensor object in the input of the operation."""
    arg_def = None  # type: TreeDef
    r"""The :class:`TreeDef` used to reconstruct the input of the operation."""
    out_def = None  # type: TreeDef
    r"""The :class:`TreeDef` used to reconstruct the output of the operation."""
    _top_graph = None  # type: weakref.ReferenceType
    __total_id = 0

    def __init__(self) -> None:
        self._id = Expr.__total_id
        Expr.__total_id += 1
        self._disable_remove = False

    def enable_remove(self):
        self._disable_remove = False

    def disable_remove(self):
        self._disable_remove = True

    def add_inputs(self, vals):
        if not isinstance(vals, collections.abc.Sequence):
            vals = (vals,)
        for val in vals:
            node = NodeMixin.get(val, None)
            if isinstance(node, (TensorNode, ModuleNode)):
                self.inputs.append(node)
                node.users.append(self)
            else:
                assert node is None
                assert _is_leaf(val) and _is_const_leaf(val)
                idx = len(self.inputs) + len(self.const_val)
                self.const_val.append((idx, val))

    def add_outputs(self, outputs):
        self.outputs = []
        if outputs is not None:
            if not isinstance(outputs, collections.Sequence):
                outputs = (outputs,)

            name = None
            orig_name = None
            if isinstance(self, CallMethod):
                name = self.inputs[0]._name
                orig_name = self.inputs[0]._orig_name
                assert isinstance(name, str), "The name of ({}) must be a str".format(
                    self.inputs[0]
                )
                assert isinstance(
                    orig_name, str
                ), "The orig_name of ({}) must be a str".format(self.inputs[0])
                name = rstrip(name, "_out")
                if self.method == "__call__":
                    name += "_out"
                    orig_name += "_out"
                else:
                    strip_method = self.method.strip("_")
                    name = "%s_out" % strip_method
                    orig_name = name
            elif isinstance(self, CallFunction):
                name = self.func.__name__ + "_out"
            elif isinstance(self, Apply):
                name = str(self.opdef).lower() + "_out"

            for i in outputs:
                assert isinstance(i, RawTensor), "The output must be a Tensor"
                o_name = (
                    active_module_tracer().current_scope()._create_unique_name(name)
                )
                self.outputs.append(
                    NodeMixin.get_wrapped_type(i)(
                        expr=self,
                        name=o_name,
                        orig_name=orig_name if orig_name else o_name,
                    )
                )

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

    def replace_inputs(self, repl_dict: Dict[Node, Node]):
        r"""Replace the input Nodes of this Expr.
        
        Args:
            repl_dict: the map {old_Node: new_Node} that specifies how to replace the input Nodes.
        """
        while repl_dict:
            node, repl_node = repl_dict.popitem()
            assert type(node) == type(repl_node)
            assert node in self.inputs, "({}) is not in the ({})".format(node, self)
            assert (
                repl_node.top_graph == node.top_graph
            ), "({}) and ({}) are not in the same graph".format(node, repl_node)
            graph = self.top_graph
            repl_expr_idx = graph._exprs.index(repl_node.expr)
            self_idx = graph._exprs.index(self)
            assert (
                repl_expr_idx < self_idx
            ), "({}) must be generated before ({})".format(repl_node, self)
            idx = self.inputs.index(node)
            self.inputs[idx] = repl_node
            user_idx = node.users.index(self)
            assert user_idx >= 0
            node.users.pop(user_idx)
            repl_node.users.append(self)

    @property
    def kwargs(self):
        r"""Get the keyword arguments of the operation corresponding to this Expr."""
        _, kwargs = self.unflatten_args(self.inputs)
        return kwargs

    @property
    def args(self):
        r"""Get the positional arguments of the operation corresponding to this Expr."""
        args, _ = self.unflatten_args(self.inputs)
        return args

    @property
    def top_graph(self):
        r"""Get the parent graph of this Expr."""
        if self._top_graph:
            return self._top_graph()
        return None

    def __getstate__(self):
        state = self.__dict__.copy()
        if "_top_graph" in state:
            state.pop("_top_graph")
        return state

    @classmethod
    def _get_next_id(cls):
        return cls.__total_id

    @classmethod
    def _set_next_id(cls, id: int = 0):
        assert isinstance(id, int)
        cls.__total_id = id


# expr: None (i.e. fake expression which is used to mark input)
class Input(Expr):
    r"""A fake Expr which is used to mark the input of graph."""
    name = None

    def __init__(self, name=None, type=None, orig_name=None):
        super().__init__()
        self.inputs = []
        node_cls = type if type else Node
        if orig_name is None:
            orig_name = name
        self.outputs = [
            node_cls(self, name=name, orig_name=orig_name),
        ]
        self.name = name

    @classmethod
    def make(cls, *args, **kwargs):
        expr = cls(*args, **kwargs)
        oup_node = expr.outputs[0]
        name = (
            active_module_tracer().current_scope()._create_unique_name(oup_node._name)
        )
        oup_node._name = name
        active_module_tracer().current_scope()._add_input(oup_node)
        return expr.outputs[0]

    def __repr__(self):
        return "%{}:\t{} = Input()".format(self._id, self.outputs[0])


# expr: outputs = getattr(inputs[0], self.name)
class GetAttr(Expr):
    r"""``Getattr`` represents the fetch of an attribute from the ``Module`` hierarchy."""

    name = None
    r"""name: the qualified name of the attribute to be retrieved."""

    def __init__(self, module, name, type=None, orig_name=None):
        super().__init__()
        assert isinstance(module, ModuleNode)
        self.inputs = [
            module,
        ]
        module.users.append(self)
        self.name = name
        node_cls = type if type else Node
        self.outputs = [
            node_cls(self, name=name, orig_name=orig_name),
        ]

    @classmethod
    def make(cls, *args, **kwargs):
        expr = cls(*args, **kwargs)
        module = expr.inputs[0]
        oup_name = expr.name
        while module._name != "self":
            oup_name = module._name + "_" + oup_name
            module = module.expr.inputs[0]
        oup_name = active_module_tracer().current_scope()._create_unique_name(oup_name)
        expr.outputs[0]._name = oup_name
        active_module_tracer().current_scope()._insert(expr)
        return expr.outputs[0]

    def interpret(self, *inputs):
        return (getattr(inputs[0], self.name),)

    def __repr__(self):
        out_type = "Tensor"
        if isinstance(self.outputs[0], ModuleNode):
            out_type = self.outputs[0].module_type.__name__
        return '%{}:\t{} = getattr({}, "{}") -> ({})'.format(
            self._id, self.outputs[0], self.inputs[0], self.name, out_type
        )


# expr: outputs = inputs[0].__call__(*inputs[1:])
class CallMethod(Expr):
    r"""``CallMethod`` represents a call to the ``__call__`` method of ``Module`` or a method of ``Tensor``.

    Args:
        node: the Node to be called.
        method: the method name.
            Default: "__call__"
    """

    def __init__(self, node, method="__call__"):
        super().__init__()
        if isinstance(node, type):
            assert issubclass(node, Tensor)
            cls = Parameter if issubclass(node, Parameter) else Tensor

            self.inputs = []
            self.const_val = [(0, cls)]
        else:
            assert isinstance(node, (TensorNode, ModuleNode))
            node.users.append(self)
            self.inputs = [
                node,
            ]
            self.const_val = []
        self.method = method

    @classmethod
    def make(cls, *args, **kwargs):
        expr = cls(*args, **kwargs)
        active_module_tracer().current_scope()._insert(expr)
        return expr

    @property
    def graph(self):
        if isinstance(self.inputs[0], ModuleNode):
            m_node = self.inputs[0]
            if (
                hasattr(m_node.owner, "argdef_graph_map")
                and m_node.owner.argdef_graph_map
            ):
                assert self.arg_def in m_node.owner.argdef_graph_map
                return m_node.owner.argdef_graph_map[self.arg_def]
        return None

    def interpret(self, *inputs):
        args, kwargs = self.unflatten_args(inputs)
        obj = args[0]
        meth = getattr(obj, self.method)
        if inspect.ismethod(meth):
            args = args[1:]
        outputs = getattr(obj, self.method)(*args, **kwargs)
        if self.method == "__setitem__":
            outputs = obj
        if outputs is None:
            return outputs
        outputs, _ = tree_flatten(outputs, is_leaf=lambda x: isinstance(x, RawTensor))
        return outputs

    def __repr__(self):
        args = ", ".join(str(i) for i in self.args[1:])
        kwargs = ", ".join("{}={}".format(k, v) for k, v in self.kwargs.items())
        outputs = self.outputs
        if self.out_def:
            outputs = self.out_def.unflatten(outputs)
        method = ".%s" % self.method
        if method == ".__call__":
            method = ""
        return "%{}:\t{}{}{}({})".format(
            self._id,
            str(outputs) + " = " if outputs else "",
            self.args[0],
            method,
            ", ".join([args, kwargs]),
        )


# expr: outputs = apply(self.opdef, *inputs)
class Apply(Expr):
    r"""``Apply`` represents a call to :func:`apply`.

    Args:
        opdef: the applied :class:`OpDef`.
    """
    opdef = None

    def __init__(self, opdef):
        super().__init__()
        assert isinstance(opdef, OpDef)
        self.opdef = opdef
        self.inputs = []

    @classmethod
    def make(cls, *args, **kwargs):
        expr = cls(*args, **kwargs)
        active_module_tracer().current_scope()._insert(expr)
        return expr

    def interpret(self, *inputs):
        return apply(self.opdef, *inputs)

    def __repr__(self):
        return "%{}:\t{} = {}({})".format(
            self._id,
            ", ".join(str(i) for i in self.outputs),
            self.opdef,
            ", ".join(str(i) for i in self.inputs),
        )

    def __getstate__(self):
        state = super().__getstate__()
        state["opdef"] = get_opdef_state(state["opdef"])
        return state

    def __setstate__(self, state):
        state["opdef"] = load_opdef_from_state(state["opdef"])
        for k, v in state.items():
            setattr(self, k, v)

    @classmethod
    def apply_module_trace_hook(cls, opdef, *inputs):
        for i in inputs:
            node = NodeMixin.get(i, None)
            if node is None:  # capture as constant
                NodeMixin.wrap_safe(i, Constant.make(i))

        if isinstance(opdef, FakeQuant):
            inp_nodes = [NodeMixin.get(inputs[0])]
            for i in inputs[1:]:
                node = Constant.make(i)
                inp_nodes.append(node)
            apply_node = cls.make(opdef)
            for n in inp_nodes:
                n.users.append(apply_node)
            apply_node.inputs = inp_nodes
        else:
            apply_node = cls.make(opdef)
            apply_node.add_inputs(inputs)

        assert not apply_node.const_val

        unset_module_tracing()
        outputs = apply(opdef, *inputs)
        set_module_tracing()

        apply_node.add_outputs(outputs)
        for n, v in zip(apply_node.outputs, outputs):
            NodeMixin.wrap_safe(v, n)
        return list(outputs)


class CallFunction(Expr):
    r"""``CallFunction`` represents a call to a built-in function.
    
    Args:
        func: a built-in function.
    """

    def __init__(self, func):
        super().__init__()
        assert isinstance(func, Callable)
        self.func = func
        self.const_val = []
        self.inputs = []

    @classmethod
    def make(cls, *args, **kwargs):
        expr = cls(*args, **kwargs)
        active_module_tracer().current_scope()._insert(expr)
        return expr

    def interpret(self, *inputs):
        args, kwargs = self.unflatten_args(inputs)
        outputs = self.func(*args, **kwargs)
        if outputs is None:
            return outputs
        outputs, _ = tree_flatten(outputs, is_leaf=lambda x: isinstance(x, RawTensor))
        return outputs

    def __repr__(self):
        args = ", ".join(str(i) for i in self.args)
        kwargs = ", ".join("{}={}".format(k, v) for k, v in self.kwargs.items())
        outputs = self.outputs
        if self.out_def:
            outputs = self.out_def.unflatten(outputs)
        return "%{}:\t{}{}({})".format(
            self._id,
            str(outputs) + " = " if outputs else "",
            self.func.__module__.rsplit(".")[-1] + "." + self.func.__name__,
            ", ".join([args, kwargs]),
        )


# expr outputs = self.value
class Constant(Expr):
    r"""``Constant`` represents a ``Tensor`` or "Module" which is not the attribute of a Module.

    Args:
        c: a const Tensor or Module.
        name: the name of output Node.
    """
    value = None
    r"""The const Tensor or Module"""
    # TODO: constant cache to reduce the size of dumped model
    _constant_cache = {}

    def __init__(self, c, name=None):
        super().__init__()
        assert isinstance(c, (RawTensor, Module))
        if isinstance(c, Module):
            assert module_tracer.is_builtin(c) or c.is_qat
        self.value = c
        self.name = name
        self.inputs = []
        node_cls = NodeMixin.get_wrapped_type(c)
        self.outputs = [
            node_cls(self, name=name, orig_name=name),
        ]
        self.outputs[0]._name = name if name else "const_" + str(self._id)

    @classmethod
    def make(cls, *args, **kwargs):
        expr = cls(*args, **kwargs)
        name = "const_module" if isinstance(expr.value, Module) else "const_tensor"
        full_name = name
        if (
            isinstance(expr.value, RawTensor)
            and id(expr.value) in active_module_tracer().id2name
        ):
            full_name = active_module_tracer().id2name[id(expr.value)]
            scope_name = active_module_tracer().current_scope()._module_name
            if full_name and scope_name:
                full_name = ("self." + full_name)[len(scope_name) + 1 :]
            else:
                full_name = name
        else:
            full_name = name
        name = active_module_tracer().current_scope()._create_unique_name(full_name)
        expr.outputs[0]._name = name
        expr.outputs[0]._orig_name = full_name
        active_module_tracer().current_scope()._insert(expr)
        return expr.outputs[0]

    def interpret(self, *inputs):
        if isinstance(self.value, RawTensor):
            return Const(self.value.numpy())()
        return (self.value,)

    def __repr__(self):
        name = self.name
        if name is None:
            name = type(self.value)
        node_type = "Module"
        if isinstance(self.outputs[0], TensorNode):
            node_type = "Tensor"
        return "%{}:\t{} = Constant({}) -> ({})".format(
            self._id, self.outputs[0], name, node_type
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        if "_top_graph" in state:
            state.pop("_top_graph")
        if isinstance(self.value, RawTensor):
            state["value"] = Tensor(self.value)
        return state
