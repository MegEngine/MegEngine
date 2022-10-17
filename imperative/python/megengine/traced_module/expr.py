import builtins
import collections
import copy
import inspect
import re
import weakref
from importlib import import_module
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

from ..core._imperative_rt import OpDef
from ..core._imperative_rt.core2 import Const
from ..core._imperative_rt.core2 import Tensor as RawTensor
from ..core._imperative_rt.core2 import (
    apply,
    is_tracing_module,
    set_module_trace_hook,
    set_module_tracing,
    unset_module_tracing,
)
from ..core.ops.builtin import FakeQuant
from ..module import Module
from ..tensor import Parameter, Tensor
from ..version import __version__
from .module_tracer import active_module_tracer, module_tracer
from .node import ModuleNode, Node, NodeMixin, TensorNode
from .pytree import ArgsIndex, TreeDef, _is_const_leaf, _is_leaf, tree_flatten
from .serialization import _ModuleState
from .tm_config import _exclude_from_trace, _get_expr_checker
from .utils import _check_builtin_module_attr, _check_obj_attr, _convert_kwargs_to_args


def rstrip(s: str, __chars: str):
    __chars = re.escape(__chars)
    s = re.sub(r"^(?P<left>.*?)(?:%s)+$" % __chars, "\g<left>", s)
    return s


def get_suffix_name(prefix: str, name: str):
    if prefix == name:
        return ""
    matchd = re.compile("^%s\.(.*)" % prefix).match(name)
    if matchd is None:
        return None
    return matchd.group(1)


def is_call_module(expr, module_cls: Module = None):
    return (
        isinstance(expr, CallMethod)
        and isinstance(expr.inputs[0], ModuleNode)
        and expr.method == "__call__"
    ) and (module_cls is None or isinstance(expr.inputs[0].owner, module_cls))


def is_call_tensor_method(expr, method: Iterable[str] = None):
    if method and isinstance(method, str):
        method = (method,)
    return (
        isinstance(expr, CallMethod)
        and not is_call_module(expr)
        and (method is None or any(expr.method == f for f in method))
    )


def is_call_function(expr, func: Iterable[Callable] = None):
    if func and not isinstance(func, Iterable):
        func = (func,)
    return isinstance(expr, CallFunction) and (
        func is None or any(expr.func == f for f in func)
    )


def is_constant(expr):
    return isinstance(expr, Constant)


def is_getattr(expr):
    return isinstance(expr, GetAttr)


def is_apply_def(expr, opdef=None):
    return isinstance(expr, Apply) and (opdef is None or isinstance(expr.opdef, opdef))


def is_input(expr):
    return isinstance(expr, Input)


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
                assert not isinstance(val, (Module, RawTensor))
                assert _is_leaf(val) and _is_const_leaf(val)
                idx = len(self.inputs) + len(self.const_val)
                self.const_val.append((idx, val))

    def add_outputs(self, outputs):
        assert active_module_tracer() is not None
        self.outputs = []
        if outputs is None:
            return
        current_graph = active_module_tracer().current_scope()
        if not isinstance(outputs, collections.abc.Sequence):
            outputs = (outputs,)
        for i in outputs:
            assert isinstance(i, RawTensor), "The output must be a Tensor"
            node = NodeMixin.get_wrapped_type(i)(expr=self, name="", qualname="",)
            NodeMixin.wrap_safe(i, node)
            self.outputs.append(node)
        current_graph._namespace.auto_naming_for_outputs(self)

    def unflatten_args(self, inputs):
        assert self.arg_def is not None, "{} expr doesn't have args/kwargs".format(
            type(self).__name__
        )
        inputs = list(inputs)
        for idx, val in self.const_val:
            inputs.insert(idx, val)
        args, kwargs = self.arg_def.unflatten(inputs)
        return args, kwargs

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
            node.users.remove(self)
            repl_node.users.append(self)

    @property
    def _support_set_args_kwargs(self):
        return False

    def set_args_kwargs(self, *args, **kwargs):
        r""" Set args and kwargs for Expr.
        """
        assert (
            self._support_set_args_kwargs
        ), "Doesn't support set args/kwargs for {} expr".format(type(self).__name__)
        args, kwargs = _convert_kwargs_to_args(self._get_func(), args, kwargs)
        inputs, arg_def = tree_flatten((args, kwargs))
        orig_inputs = self.inputs
        self.inputs = []
        self.const_val = []
        for val in inputs:
            if isinstance(val, (TensorNode, ModuleNode)):
                self.inputs.append(val)
            else:
                assert _is_leaf(val) and _is_const_leaf(val)
                idx = len(self.inputs) + len(self.const_val)
                self.const_val.append((idx, val))

        for n in orig_inputs:
            if n not in self.inputs:
                n.users.remove(self)

        for n in self.inputs:
            if n not in orig_inputs:
                n.users.append(self)

        self.arg_def = arg_def

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

    def _get_func(self):
        # get called function when the expr is interpreted
        raise NotImplementedError

    @property
    def named_args(self):
        func = self._get_func()
        return inspect.getcallargs(func, *self.args, **self.kwargs)

    def set_arg(self, name, val):
        func = self._get_func()
        if name in self.kwargs:
            new_kwargs = self.kwargs
            new_kwargs[name] = val
            self.set_args_kwargs(*self.args, **new_kwargs)
        else:
            arg_spec = inspect.getfullargspec(func)
            if name in arg_spec.args:
                ind = arg_spec.args.index(name)
                new_args = list(self.args)
                new_args[ind] = val
                self.set_args_kwargs(*new_args)
            elif name == arg_spec.varargs:
                assert arg_spec.varargs is not None
                assert len(self.args) >= len(arg_spec.args)
                val = (val,) if not isinstance(val, Sequence) else val
                self.set_args_kwargs(*self.args[0 : len(arg_spec.args)], *val)
            else:
                assert (
                    arg_spec.varkw is not None
                ), "func {} does't have argument named {}".format(func, name)
                new_kwargs = self.kwargs
                new_kwargs[name] = val
                self.set_args_kwargs(*self.args, **new_kwargs)

    @property
    def return_val(self):
        return self.out_def.unflatten(self.outputs)

    @return_val.setter
    def return_val(self, new_outputs):
        outputs, out_def = tree_flatten(
            new_outputs, is_leaf=lambda x: isinstance(x, Node)
        )
        assert all(
            isinstance(o, Node) for o in outputs
        ), "Return values of expr must be ModuleNode or TensorNode or Container with them"
        assert all(
            o.expr in (None, self) for o in outputs
        ), "Some nodes are produced by other expr, can not be output of expr {}".format(
            self
        )
        self.outputs = outputs
        self.out_def = out_def

    @property
    def top_graph(self):
        r"""Get the parent graph of this Expr."""
        if self._top_graph:
            return self._top_graph()
        return None

    @classmethod
    def _get_next_id(cls):
        return cls.__total_id

    @classmethod
    def _set_next_id(cls, id: int = 0):
        assert isinstance(id, int)
        cls.__total_id = id

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        state = {}
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if not isinstance(v, weakref.ReferenceType):
                state[k] = copy.deepcopy(v, memo)
        result.__dict__.update(state)
        return result


# expr: None (i.e. fake expression which is used to mark input)
class Input(Expr):
    r"""A fake Expr which is used to mark the input of graph."""
    name = None

    def __init__(self, type: List[Node], name: str = "args", qualname: str = ""):
        super().__init__()
        assert type in [ModuleNode, TensorNode]
        assert name and qualname
        self.inputs = []
        node_cls = type if type else Node
        self.outputs = [
            node_cls(self, name=name, qualname=qualname),
        ]
        self.name = name

    @classmethod
    def make(cls, *args, **kwargs):
        assert active_module_tracer() is not None
        current_graph = active_module_tracer().current_scope()
        expr = cls(*args, **kwargs)
        out_node = expr.outputs[0]
        current_graph._namespace.auto_naming_for_outputs(expr)
        current_graph._add_input(out_node)
        return expr.outputs[0]

    def __repr__(self):
        return "%{}:\t{} = Input()".format(self._id, self.outputs[0])

    def __getstate__(self):
        state = {
            "_id": self._id,
            "_disable_remove": self._disable_remove,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "name": self.name,
        }
        _check_obj_attr(state)
        return state


# expr: outputs = getattr(inputs[0], self.name)
class GetAttr(Expr):
    r"""``Getattr`` represents the fetch of an attribute from the ``Module`` hierarchy."""

    name = None
    r"""name: the qualified name of the attribute to be retrieved."""

    def __init__(
        self, module: ModuleNode, type: Union[Node], attr_name: str, name: str = "",
    ):
        super().__init__()
        assert isinstance(module, ModuleNode)
        assert type in [TensorNode, ModuleNode]
        self.inputs = [
            module,
        ]
        module.users.append(self)
        self.name = attr_name
        self.outputs = [
            type(self, name=name, qualname="{}.{}".format(module.qualname, attr_name)),
        ]

    @classmethod
    def make(cls, *args, **kwargs):
        assert active_module_tracer() is not None
        current_graph = active_module_tracer().current_scope()
        expr = cls(*args, **kwargs)
        current_graph._namespace.auto_naming_for_outputs(expr)
        current_graph._insert(expr)
        return expr.outputs[0]

    def interpret(self, *inputs):
        mod = inputs[0]
        module_path, _, name = self.name.rpartition(".")
        if module_path == "":
            return (getattr(mod, name),)
        module_names = module_path.split(".")
        for item in module_names:
            mod = getattr(mod, item)
            if not isinstance(mod, Module):
                raise AttributeError("`{}` is not an Module".format(item))
        return (getattr(mod, name),)

    def __repr__(self):
        out_type = "Tensor"
        if isinstance(self.outputs[0], ModuleNode):
            m_type = self.outputs[0].module_type
            out_type = m_type.__name__ if isinstance(m_type, type) else m_type[1]
        return '%{}:\t{} = getattr({}, "{}") -> ({})'.format(
            self._id, self.outputs[0], self.inputs[0], self.name, out_type
        )

    def __getstate__(self):
        state = {
            "_id": self._id,
            "_disable_remove": self._disable_remove,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "name": self.name,
        }
        _check_obj_attr(state)
        return state


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
        self.arg_def = tree_flatten(((node,), {}))[1]
        self.method = method

    @classmethod
    def make(cls, *args, **kwargs):
        assert active_module_tracer() is not None
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

    def _get_func(self):
        if isinstance(self.args[0], type):
            obj_type = self.args[0]
        elif isinstance(self.args[0], ModuleNode):
            obj_type = self.args[0].module_type
        else:
            assert isinstance(self.args[0], TensorNode)
            obj_type = Tensor
        meth = getattr(
            obj_type, "forward" if issubclass(obj_type, Module) else self.method
        )
        return meth

    @property
    def _support_set_args_kwargs(self):
        # only expr call tensor method or builtin module support modify args/kwargs
        return (
            isinstance(self.args[0], (TensorNode, type))
            or self.args[0].module_type is not Module
        )

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

    def __getstate__(self):
        state = {
            "_id": self._id,
            "_disable_remove": self._disable_remove,
            "inputs": self.inputs,
            "const_val": self.const_val,
            "method": self.method,
            "arg_def": self.arg_def,
            "out_def": self.out_def,
            "outputs": self.outputs,
            "version": __version__,
        }
        _check_obj_attr(state)
        return state


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
        assert active_module_tracer() is not None
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
        opdef_state = self.opdef.__getstate__()
        opdef_state["opdef_type"] = type(self.opdef)
        state = {
            "_id": self._id,
            "_disable_remove": self._disable_remove,
            "opdef_state": opdef_state,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "version": __version__,
        }
        _check_obj_attr(state)
        return state

    def __setstate__(self, state):
        # compat with mge 1.6
        if "opdef" in state and "opdef_state" not in state:
            opdef_state = state.pop("opdef")
            opdef_state["opdef_type"] = opdef_state.pop("type")
            state["opdef_state"] = opdef_state
        self.__dict__.update(state)
        assert isinstance(state["opdef_state"], dict)
        opdef_state = state["opdef_state"].copy()
        opdef_type = opdef_state.pop("opdef_type")
        opdef_obj = opdef_type()
        opdef_obj.__setstate__(opdef_state)
        setattr(self, "opdef", opdef_obj)

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
                if _get_expr_checker():
                    active_module_tracer().checker.record_node2value(node, Tensor(i))
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

        if _get_expr_checker():
            with _exclude_from_trace():
                active_module_tracer().checker.check_apply(apply_node, outputs, opdef)

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
        assert active_module_tracer() is not None
        expr = cls(*args, **kwargs)
        active_module_tracer().current_scope()._insert(expr)
        return expr

    def interpret(self, *inputs):
        args, kwargs = self.unflatten_args(inputs)
        func = (
            self.func
            if not is_tracing_module()
            else active_module_tracer().patcher.wrap_fn(self.func)
        )
        outputs = func(*args, **kwargs)
        if outputs is None:
            return outputs
        outputs, _ = tree_flatten(outputs, is_leaf=lambda x: isinstance(x, RawTensor))
        return outputs

    def _get_func(self):
        return self.func

    @property
    def _support_set_args_kwargs(self):
        return True

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

    def __getstate__(self):
        state = {
            "_id": self._id,
            "_disable_remove": self._disable_remove,
            "func": (self.func.__module__, self.func.__qualname__),
            "const_val": self.const_val,
            "inputs": self.inputs,
            "arg_def": self.arg_def,
            "out_def": self.out_def,
            "outputs": self.outputs,
            "version": __version__,
        }
        _check_obj_attr(state)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        try:
            if isinstance(self.func, tuple):
                mname, fname = self.func
                f = import_module(mname)
                for i in fname.split("."):
                    f = getattr(f, i)
                self.func = f
        except Exception:
            pass


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

    def __init__(self, c, name: str = "", qualname: str = ""):
        super().__init__()
        assert isinstance(c, (RawTensor, Module))
        if isinstance(c, Module):
            assert module_tracer.is_builtin(c) or c.is_qat
        if type(c) is RawTensor:
            with _exclude_from_trace():
                c = Tensor(c)
        self.value = c
        self.name = name
        self.inputs = []
        node_cls = NodeMixin.get_wrapped_type(c)
        self.outputs = [
            node_cls(self, name=name, qualname=qualname),
        ]

    @classmethod
    def make(cls, *args, **kwargs):
        assert active_module_tracer() is not None
        expr = cls(*args, **kwargs)
        current_graph = active_module_tracer().current_scope()
        current_graph._namespace.auto_naming_for_outputs(expr)
        current_graph._insert(expr)
        active_module_tracer().current_constant_cache().append(expr.value)
        return expr.outputs[0]

    def interpret(self, *inputs):
        if isinstance(self.value, RawTensor):
            return (Const(self.value.numpy(), None, None),)
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
        state = {
            "_id": self._id,
            "_disable_remove": self._disable_remove,
            "value": self.value,
            "name": self.name,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }
        _check_obj_attr(state)
        if isinstance(self.value, RawTensor):
            state["value"] = Tensor(self.value)
        if isinstance(self.value, Module) and module_tracer.is_builtin(self.value):
            _check_builtin_module_attr(self.value)
            state["value"] = _ModuleState.get_module_state(self.value)

        return state

    def __setstate__(self, state):
        for k, v in state.items():
            if isinstance(v, _ModuleState):
                state[k] = v.to_module()
        self.__dict__.update(state)


def _module_trace_capture(value):
    node = Constant.make(value)
    NodeMixin.wrap_safe(value, node)
    return node


set_module_trace_hook(Apply.apply_module_trace_hook)
