import builtins
import collections
import copy
import fnmatch
import functools
import inspect
import keyword
import re
import weakref
from importlib import import_module
from inspect import getcallargs, getmembers, isclass, ismethod
from itertools import chain
from types import FunctionType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from .. import functional as F
from .. import get_logger
from .. import module as M
from ..core._imperative_rt.core2 import Tensor as RawTensor
from ..core._imperative_rt.core2 import (
    apply,
    is_tracing_module,
    set_module_tracing,
    unset_module_tracing,
)
from ..core._trace_option import set_symbolic_shape, use_symbolic_shape
from ..core.ops.builtin import Copy
from ..module import Module
from ..module import external as MExternal
from ..module.qat import QATModule
from ..quantization.fake_quant import LSQ, TQT, FakeQuantize, _FakeQuantize
from ..quantization.observer import (
    ExponentialMovingAverageObserver,
    HistogramObserver,
    MinMaxObserver,
    Observer,
    PassiveObserver,
    SyncExponentialMovingAverageObserver,
    SyncMinMaxObserver,
)
from ..tensor import Tensor
from ..utils.max_recursion_limit import max_recursion_limit
from ..version import __version__
from .expr import (
    Apply,
    CallFunction,
    CallMethod,
    Constant,
    Expr,
    GetAttr,
    Input,
    get_suffix_name,
    is_apply_def,
    is_call_function,
    is_call_module,
    is_call_tensor_method,
    is_constant,
    is_getattr,
    is_input,
)
from .fake_quant import FakeQuantize as TM_FakeQuant
from .module_tracer import (
    PatchedFn,
    Patcher,
    active_module_tracer,
    get_tensor_wrapable_method,
    module_tracer,
    set_active_module_tracer,
)
from .node import ModuleNode, Node, NodeMixin, TensorNode
from .pytree import (
    USER_REGISTERED_CONTAINER_TYPE,
    USER_REGISTERED_LEAF_TYPE,
    ArgsIndex,
    TreeDef,
    _register_supported_type,
    tree_flatten,
)
from .serialization import (
    _ModuleState,
    load_apply_expr,
    load_call_module_expr,
    load_call_tensor_method_expr,
    load_functional,
)
from .tm_config import (
    _exclude_from_trace,
    _get_default_checker,
    _get_expr_checker,
    _graph_surgery_mode,
    _set_graph_surgery_mode,
)
from .utils import (
    _check_builtin_module_attr,
    _check_obj_attr,
    _convert_kwargs_to_args,
    replace_container_with_module_container,
)

logger = get_logger(__name__)


def _is_builtin_name(name: str) -> bool:
    return (
        name in builtins.__dict__
        or name in keyword.kwlist
        or name in {"inf", "nan", "NoneType"}
    )


def _is_leaf(node):
    assert isinstance(
        node, RawTensor
    ), 'doesn\'t support {} in return values, MUST use Tensor or use "register_supported_type" method to register self-defined type'.format(
        type(node)
    )
    return isinstance(node, RawTensor)


def _node_to_tensor(*args, **kwargs):
    tensors = []
    nodes, tree_def = tree_flatten((args, kwargs))
    for n in nodes:
        if isinstance(n, TensorNode):
            if n.top_graph is not None:
                active_module_tracer().current_scope()._add_input(n)
            value = n.value
            if value is None:
                flag = _set_graph_surgery_mode(False)
                with _exclude_from_trace():
                    value = F.zeros(shape=n._shape, dtype=n._dtype)
                _set_graph_surgery_mode(flag)
            orig_n = NodeMixin.get(value, None)
            if orig_n is None or "setitem" not in orig_n._name:
                NodeMixin.wrap_safe(value, n)
            tensors.append(value)
        else:
            tensors.append(n)
    tensors = tree_def.unflatten(tensors)
    return tensors


def _tensor_to_node(tensors):
    if tensors is None:
        return None
    nodes = []
    tensors, out_def = tree_flatten(tensors)
    for t in tensors:
        if isinstance(t, Tensor):
            n = NodeMixin.get(t, None)
            if isinstance(n, TensorNode):
                n.value = t
                nodes.append(n)
            else:
                nodes.append(t)
        else:
            nodes.append(t)
    nodes = out_def.unflatten(nodes)
    return nodes


def _name_setter(node: Node, new_name: str):
    surgery_mode = _set_graph_surgery_mode(False)
    graph = active_module_tracer().current_scope()

    if node.top_graph is not None:
        top_graph = active_module_tracer().top_scope()
        if node is top_graph._namespace.used_names.get(node._name, None):
            graph = top_graph
        else:
            graph = node.top_graph

    assert (
        graph._namespace.used_names.get(new_name, None) is None
    ), "The name(%s) is already in use. Please try a different one again." % (new_name)
    graph._namespace.unassociate_name_with_obj(node)
    node._name = graph._namespace.create_unique_name(new_name, node)
    _set_graph_surgery_mode(surgery_mode)


def _wrap_method_to_tensor_node():
    def _any_method(name, func):
        def _any(*args, **kwargs):
            if is_tracing_module() and _graph_surgery_mode():
                args, kwargs = _node_to_tensor(*args, **kwargs)
                attr = getattr(args[0], name)
                outs = attr
                if callable(attr):
                    outs = attr(*(args[1:]), **kwargs)
                if name == "__setitem__":
                    _node_to_tensor(outs)
                    return None
                outs = _tensor_to_node(outs)
                return outs
            else:
                outs = func
                if callable(func):
                    outs = func(*args, **kwargs)
                if isinstance(func, property):
                    outs = func.__get__(*args, **kwargs)
            return outs

        return _any

    tensor_method_patch = []
    for method in get_tensor_wrapable_method():
        patch = PatchedFn(TensorNode, method)
        if type(getattr(Tensor, method)) == property:
            # Only support property.getter
            patch.set_func(property(_any_method(method, patch.origin_fn)))
        else:
            patch.set_func(_any_method(method, patch.origin_fn))
        tensor_method_patch.append(patch)

    patch = PatchedFn(Node, "name")
    patch.set_func(property(patch.origin_fn.fget, _name_setter))
    tensor_method_patch.append(patch)
    return tensor_method_patch


def _convert_node_and_tensor(orig_func):
    @functools.wraps(orig_func)
    def _convert(*args, **kwargs):
        if is_tracing_module() and _graph_surgery_mode():
            args, kwargs = _node_to_tensor(*args, **kwargs)
            rst = orig_func(*args, **kwargs, method_func=_convert)
            rst = _tensor_to_node(rst)
            return rst
        else:
            rst = orig_func(*args, **kwargs)
        return rst

    return _convert


def _wrap_mnode_getattr(orig_getattr):
    @functools.wraps(orig_getattr)
    def wraped_fn(self, name):
        if is_tracing_module() and _graph_surgery_mode():
            obj = self.owner
            current_graph = active_module_tracer().current_scope()
            if self.top_graph is not None:
                current_graph._add_input(self)
            attr = getattr(obj, name)
            node = attr
            if not isinstance(attr, TracedModuleBuilder):
                if isinstance(attr, Module):
                    attr = TracedModuleBuilder(attr)
                    setattr(obj, name, attr)

                if isinstance(attr, (NodeMixin, RawTensor)):
                    NodeMixin.wrap(
                        attr,
                        lambda: GetAttr.make(
                            self,
                            type=NodeMixin.get_wrapped_type(attr),
                            attr_name=name,
                            name="",
                        ),
                    )
            if isinstance(attr, (NodeMixin, RawTensor)):
                node = NodeMixin.get(attr)
            if isinstance(node, ModuleNode) and isinstance(attr, (NodeMixin, Module)):
                node._owner = weakref.ref(attr)
            return node
        else:
            node = object.__getattribute__(self, name)
        return node

    return wraped_fn


def _wrap_mnode_call(orig_call):
    @functools.wraps(orig_call)
    def wraped_fn(self, *args, **kwargs):
        if is_tracing_module() and _graph_surgery_mode():
            obj = self.owner
            if self.top_graph is not None:
                active_module_tracer().current_scope()._add_input(self)
            rst = obj(*args, **kwargs)
        else:
            raise TypeError("'ModuleNode' object is not callable")
        return rst

    return wraped_fn


class _InsertExprs:
    def __init__(self, graph, expr: Optional[Expr] = None):
        self.graph = graph
        while graph.top_graph is not None:
            graph = graph.top_graph
        assert graph.inputs[0].owner._is_top
        self.root_graph = graph
        self.global_scope = InternalGraph(self.graph._name, self.graph._qualname)
        self.global_scope._namespace.merge(self.graph._namespace)
        self.expr = expr
        self._tensor_method_patch = None

    def __enter__(self):
        self.use_sym_shape = set_symbolic_shape(True)
        node_id, expr_id = self.root_graph._total_ids
        Node._set_next_id(node_id)
        Expr._set_next_id(expr_id)
        set_module_tracing()
        _set_graph_surgery_mode(True)
        assert active_module_tracer() is None
        set_active_module_tracer(
            module_tracer(lambda x: _convert_node_and_tensor(_wrapped_function(x)))
        )
        active_module_tracer().patcher.__enter__()
        for cls, name, func in [
            [ModuleNode, "__getattr__", _wrap_mnode_getattr],
            [ModuleNode, "__call__", _wrap_mnode_call],
            [TracedModuleBuilder, "__call__", _convert_node_and_tensor],
        ]:
            active_module_tracer().patcher.patch_function(cls, name, func)
        self._tensor_method_patch = _wrap_method_to_tensor_node()
        active_module_tracer().push_scope(self.global_scope)

    def __exit__(self, ty, va, tr):
        if va is not None:
            return False
        active_module_tracer().patcher.__exit__(ty, va, tr)

        while self._tensor_method_patch:
            pf = self._tensor_method_patch.pop()
            pf.set_func(pf.origin_fn)

        # delete ModuleNode.__call__ to avoid entering the
        # ModuleNode.__init__ method when call a ModuleNode object.
        delattr(ModuleNode, "__call__")

        module = self.graph.inputs[0].owner

        def build_traced_module(
            module: TracedModuleBuilder, target_module: TracedModule
        ):
            for k, v in module.__dict__.items():
                if isinstance(v, TracedModuleBuilder):
                    traced_v = v.build()
                    build_traced_module(v, traced_v)
                    setattr(target_module, k, traced_v)

        build_traced_module(module, module)

        set_symbolic_shape(self.use_sym_shape)
        _set_graph_surgery_mode(False)
        set_active_module_tracer(None)
        unset_module_tracing()

        extra_inp_nodes = set(self.global_scope.inputs)
        max_inp_expr_idx = -1
        for node in extra_inp_nodes:
            assert (
                node.top_graph == self.graph
            ), "The input node ({}) is not in the graph ({})".format(node, self.graph)
            if node.expr in self.graph._exprs:
                max_inp_expr_idx = max(
                    max_inp_expr_idx, self.graph._exprs.index(node.expr)
                )
        max_inp_expr_idx += 1

        insert_index = -1
        if self.expr in self.graph._exprs:
            insert_index = self.graph._exprs.index(self.expr)
        insert_index += 1

        if insert_index < max_inp_expr_idx:
            insert_index = max_inp_expr_idx

        for expr in self.global_scope._exprs:
            self.graph._exprs.insert(insert_index, expr)
            insert_index += 1

        self.graph._namespace.merge(self.global_scope._namespace)
        self.root_graph._total_ids = (Node._get_next_id(), Expr._get_next_id())
        self.root_graph.inputs[0].owner._update_ref()
        for node in self.root_graph.nodes():
            if isinstance(node, TensorNode):
                node.value = None
        return True


class NameSpace:
    def __init__(self, name, qualname):
        self.name = name
        self.qualname = qualname
        self._used_names = {}

    def create_unique_name(self, name: str, node: Any = None) -> str:
        assert isinstance(name, str), "The name must be a string"

        if name in self._used_names and (self._used_names[name] is node):
            return name

        name = re.sub("[^0-9a-zA-Z_]+", "_", name)
        if name[0].isdigit():
            name = "_{}".format(name)

        while (
            name in self._used_names and self._used_names[name] is not None
        ) or _is_builtin_name(name):
            match = re.match(r"(.*)_(\d+)$", name)
            if match is None:
                name = name + "_1"
            else:
                base, num = match.group(1, 2)
                name = "{}_{}".format(base, int(num) + 1)

        self._used_names.setdefault(name)

        if node is not None:
            self.associate_name_with_obj(name, node)

        return name

    def auto_naming_for_outputs(self, expr: Expr):
        _add_suffix = lambda x: x + "_out"
        if is_call_module(expr):
            call_node = expr.inputs[0]
            qualname = "%s.[out]" % (call_node.qualname)
            name = call_node.name
        elif is_call_tensor_method(expr):
            name = expr.method.strip("_")
            qualname = "{}.[{}]".format(
                self.qualname, self.create_unique_name("method_%s" % (name)),
            )
        elif is_call_function(expr):
            name = expr.func.__name__
            qualname = "{}.[{}]".format(
                self.qualname, self.create_unique_name("func_%s" % name),
            )
        elif is_apply_def(expr):
            name = str(expr.opdef).lower()
            qualname = "{}.[{}]".format(
                self.qualname, self.create_unique_name("def_%s" % name),
            )
        elif is_getattr(expr):
            qualname = "{}.{}".format(expr.inputs[0].qualname, expr.name)
            name = get_suffix_name(self.qualname, qualname)
            _add_suffix = lambda x: x
        elif is_constant(expr) or is_input(expr):
            name = (
                expr.name if expr.name else "const_" + type(expr.value).__name__.lower()
            )
            qualname = "{}.[{}]".format(self.qualname, name)
            _add_suffix = lambda x: x

        for node in expr.outputs:
            cur_name = node._name if node._name else _add_suffix(name)
            node._name = self.create_unique_name(cur_name, node)
            if node._qualname == "":
                node._qualname = qualname
            assert get_suffix_name(self.qualname, qualname) is not None

    def merge(self, other: "NameSpace"):
        self._used_names.update(other.used_names)

    def associate_name_with_obj(self, name: str, node: Node):
        assert name in self.used_names
        assert self.used_names[name] is None, "The name(%s) is already in use" % (name)
        self._used_names[name] = node

    def unassociate_name_with_obj(self, node: Node):
        assert node.name in self.used_names
        # assert self.used_names[node.name] is node
        self._used_names[node.name] = None

    @property
    def used_names(self):
        return self._used_names


class InternalGraph:
    r"""``InternalGraph`` is the main data structure used in the TracedModule.
    It is used to represent the execution procedure of Module's forward method.

    For example, the following code

    .. code-block::

        import megengine.random as rand
        import megengine.functional as F
        import megengine.module as M

        import megengine.traced_module as tm

        class MyModule(M.Module):
            def __init__(self):
                super().__init__()
                self.param = rand.normal(size=(3, 4))
                self.linear = M.Linear(4, 5)

            def forward(self, x):
                return F.relu(self.linear(x + self.param))

        net = MyModule()

        inp = F.zeros(shape = (3, 4))
        traced_module = tm.trace_module(net, inp)

    Will produce the following ``InternalGraph``::

        print(traced_module.graph)

    .. code-block:: text

        MyModule.Graph (self, x) {
                %2:     linear = getattr(self, "linear") -> (Linear)
                %3:     param = getattr(self, "param") -> (Tensor)
                %4:     add_out = x.__add__(param, )
                %5:     linear_out = linear(add_out, )
                %6:     relu_out = nn.relu(linear_out, )
                return relu_out
        }
    """

    _exprs = None  # type: List[Expr]
    _inputs = None  # type: List[Node]
    _outputs = None  # type: List[Node]
    _top_graph = None  # type: InternalGraph
    _total_ids = None  # type: List[int]

    def __init__(self, name: str, qualname: str):
        self._exprs = []
        self._inputs = []
        self._outputs = []
        self._watch_point = []
        self._end_point = []
        self._namespace = NameSpace(name, qualname)
        self._rst = collections.defaultdict(list)
        self._name = name
        self._qualname = qualname

    def _insert(self, expr):
        self._exprs.append(expr)

    @property
    def name(self) -> str:
        r"""Get the name of this graph."""
        return self._name

    @name.setter
    def name(self, new_name: str):
        r"""Set a new name to this graph."""
        mod = self.inputs[0].owner
        graph = self.top_graph
        assert graph is not None or mod._is_top, "The parent graph cannot be None."
        if graph is not None:
            assert graph._namespace.used_names.get(new_name, None) is None, (
                "The name(%s) is already in use. Please try a different one again."
                % (new_name)
            )
            new_name = graph._namespace.create_unique_name(new_name, self)
        self._name = new_name

    @property
    def qualname(self) -> str:
        r"""Get the `qualname` of this graph. The `qualname` can be used to get the
        submodule from the traced Module or Module.

        Example:
            .. code-block::

                import megengine.module as M
                import megengine.traced_module as tm
                import megengine as mge

                class block(M.Module):
                    def __init__(self):
                        super().__init__()
                        self.relu = M.ReLU()

                    def forward(self, x):
                        return self.relu(x)

                class module(M.Module):
                    def __init__(self):
                        super().__init__()
                        self.block = block()

                    def forward(self, x):
                        x = self.block(x)
                        return x

                net = module()
                traced_net = tm.trace_module(net, mge.Tensor([0.]))

                qualname = traced_net.block.graph.qualname  # qualname = "module.block"
                qualname = qualname.split(".", 1)[-1]  # qualname = "block"

                assert qualname in list(map(lambda x: x[0], net.named_modules()))
                assert qualname in list(map(lambda x: x[0], traced_net.named_modules()))
        """
        return self._qualname

    @property
    def inputs(self) -> List[Node]:
        r"""Get the list of input Nodes of this graph.

        Returns:
            A list of ``Node``.
        """
        return self._inputs

    @property
    def outputs(self) -> List[Node]:
        r"""Get the list of output Nodes of this graph.

        Returns:
            A list of ``Node``.
        """
        return self._outputs

    @property
    def top_graph(self):
        r"""Get the parent graph of this graph.

        Returns:
            An ``InternalGraph``.
        """
        if self._top_graph:
            return self._top_graph()
        return None

    def exprs(self, recursive=True):
        r"""Get the Exprs that constitute this graph.

        Args:
            recursive: whether to get the Exprs in the subgraph.
                Default: True
        Returns:
            A ``ExprFilter`` containing all Exprs of this graph.
        """
        return ExprFilter(_expr_iter(self, recursive))

    def nodes(self, recursive=True):
        r"""Get the Nodes that constitute this graph.

        Args:
            recursive: whether to get the Nodes in the subgraph.
                Default: True
        Returns:
            A ``NodeFilter`` containing all Nodes of this graph.
        """
        return NodeFilter(_node_iter(self, recursive))

    def get_function_by_type(self, func: Callable = None, recursive=True):
        r"""Filter Exprs by the type of ``CallFunction``.

        Args:
            func: a built-in function, such as ``F.relu``.
            recursive: whether to get the Exprs in the subgraph.
                Default: True
        Returns:
            A :class:`~.TracedModule.ExprFilterCallFunction`.
        """
        return self.exprs(recursive).call_function(func)

    def get_method_by_type(self, method: str = None, recursive=True):
        r"""Filter Exprs by the type of ``CallMethod``.

        Args:
            method: a method string, such as "__add__".
            recursive: whether to get the Exprs in the subgraph.
                Default: True
        Returns:
            A :class:`~.TracedModule.ExprFilterCallMethod`.
        """
        return self.exprs(recursive).call_method(method)

    def get_expr_by_id(self, expr_id: List[int] = None, recursive=True):
        r"""Filter Exprs by their ``id``.

        Args:
            expr_id: a list of :class:`int`.
            recursive: whether to get the Exprs in the subgraph.
                Default: True
        Returns:
            A :class:`~.TracedModule.ExprFilterExprId`.
        """
        return self.exprs(recursive).expr_id(expr_id)

    def get_module_by_type(self, module_cls: Module, recursive=True):
        r"""Filter Nodes by the ``module_type`` of ``ModuleNode``.

        Args:
            module_cls: a subclass of :class:`~.Module`.
            recursive: whether to get the Nodes in the subgraph.
                Default: True
        Returns:
            A :class:`~.TracedModule.NodeFilterType`.
        """
        return self.nodes(recursive).type(module_cls)

    def get_node_by_id(self, node_id: List[int] = None, recursive=True):
        r"""Filter Nodes by their ``id``.

        The ``id`` of the ``Node`` can be obtained by the following code

        .. code-block::

            # node : Node
            print("{:i}".format(node))
            print(node.__format__("i"))
            # graph : InternalGraph
            print("{:i}".format(graph))
            print(graph.__format__("i"))

        Args:
            node_id: a list of :class:`int`.
            recursive: whether to get the Nodes in the subgraph.
                Default: True
        Returns:
            A :class:`~.TracedModule.NodeFilterNodeId`.
        """
        return self.nodes(recursive).node_id(node_id)

    def get_node_by_name(
        self, name: str = None, ignorecase: bool = True, recursive=True
    ):
        r"""Filter Nodes by their full name.

        The full name of the ``Node`` can be obtained by the following code

        .. code-block::

            # node : Node
            print("{:p}".format(node))
            print(node.__format__("p"))
            # graph : InternalGraph
            print("{:p}".format(graph))
            print(graph.__format__("p"))

        Args:
            name: a string in glob syntax that can contain ``?`` and
             ``*`` to match a single or arbitrary characters.
            ignorecase: whether to ignroe case.
                Default: True
            recursive: whether to get the Nodes in the subgraph.
                Default: True
        Returns:
            A :class:`~.TracedModule.NodeFilterName`.
        """
        return self.nodes(recursive).name(name, ignorecase)

    def _add_input(self, i):
        self._inputs.append(i)

    def _add_output(self, o):
        self._outputs.append(o)

    def get_dep_exprs(self, nodes: Sequence[Node]) -> List[Expr]:
        r"""Get the dependent Exprs of the ``nodes``.

        Args:
            nodes: a list of :class:`Node`.
        Returns:
            A list of dependent :class:`Expr`.
        """
        if not isinstance(nodes, Sequence):
            nodes = (nodes,)
        ret = list()
        queue = list(nodes)
        visited_queue = list()
        while queue:
            node = queue.pop()
            visited_queue.append(node)

            expr = node.expr

            if expr not in ret:
                ret.append(expr)

            for i in expr.inputs:
                if i not in queue and i not in visited_queue:
                    queue.append(i)
        return ret

    def reset_inputs(self, *args, **kwargs):
        forma_mnode = self.inputs[0]
        moudle = forma_mnode.owner
        assert moudle._is_top, "reset_inputs only supports top graph"

        inputs, tree_def = tree_flatten(((moudle, *args), kwargs))

        def create_node(val: Tensor):
            name = self._namespace.create_unique_name("args")
            node = Input(
                type=TensorNode, name=name, qualname="%s.[%s]" % (self._qualname, name)
            ).outputs[0]
            self._namespace.associate_name_with_obj(node.name, node)
            node.shape = val.shape
            node.dtype = val.dtype
            return node

        formal_node_inputs = [
            forma_mnode,
        ]

        org_argdef = list(moudle.argdef_graph_map.keys())[0]

        for v in inputs[1:]:
            assert isinstance(v, RawTensor)
            formal_node_inputs.append(create_node(v))

        self._inputs[:] = formal_node_inputs
        moudle.argdef_graph_map[tree_def] = moudle.argdef_graph_map.pop(org_argdef)
        moudle.argdef_outdef_map[tree_def] = moudle.argdef_outdef_map.pop(org_argdef)
        return formal_node_inputs[1:]

    def add_input_node(
        self, shape: Tuple[int], dtype: str = "float32", name: str = "args"
    ):
        r"""Add an input node to the graph.

        The new Node will be the last of the positional arguments.

        Args:
            shape: the shape of the new input Node.
            dtype: the dtype of the new input Node.
                Default: float32
            name: the name of the new input Node. When the name is used in the graph,
             a suffix will be added to it.
        """
        forma_mnode = self.inputs[0]
        moudle = forma_mnode.owner
        assert moudle._is_top, "add_input_node only supports top graph"

        def create_node(name=None):
            name = self._namespace.create_unique_name(name)
            node = Input(
                type=TensorNode, name=name, qualname="%s.[%s]" % (self._qualname, name)
            ).outputs[0]
            self._namespace.associate_name_with_obj(node.name, node)
            node.shape = shape
            node.dtype = dtype
            return node

        org_argdef = list(moudle.argdef_graph_map.keys())[0]

        args, kwargs = org_argdef.unflatten(self._inputs)
        formal_inp_node = create_node(name)
        inputs, tree_def = tree_flatten(
            ((*args, formal_inp_node), kwargs),
            is_const_leaf=lambda x: not isinstance(x, (TensorNode, ModuleNode)),
        )
        self._inputs[:] = inputs[:]

        moudle.argdef_graph_map[tree_def] = moudle.argdef_graph_map.pop(org_argdef)
        moudle.argdef_outdef_map[tree_def] = moudle.argdef_outdef_map.pop(org_argdef)
        return formal_inp_node

    def reset_outputs(self, outputs):
        r"""Reset the output Nodes of the graph.

        .. note::

            This method only supports resetting the output of graphs
            that do not have a parent graph.

        Args:
            outputs: an object which inner element is Node. Support tuple, list
             dict, etc.

        For example, the following code

        .. code-block::

            import megengine.functional as F
            import megengine.module as M
            import megengine.traced_module as tm

            class MyModule(M.Module):
                def forward(self, x):
                    x = x + 1
                    return x

            net = MyModule()

            inp = F.zeros(shape = (1, ))
            traced_module = tm.trace_module(net, inp)
            graph = traced_module.graph
            inp_node = graph.inputs[1]
            out_node = graph.outputs[0]
            graph.reset_outputs((out_node, {"input": inp_node}))
            out = traced_module(inp)

        Will produce the following ``InternalGraph`` and ``out``::

            print(graph)
            print(out)

        .. code-block:: text

            MyModule.Graph (self, x) {
                    %2:     add_out = x.__add__(1, )
                    return add_out, x
            }
            (Tensor([1.], device=xpux:0), {'input': Tensor([0.], device=xpux:0)})
        """
        outputs, out_def = tree_flatten(
            outputs, is_leaf=lambda x: isinstance(x, TensorNode),
        )
        forma_mnode = self.inputs[0]
        moudle = forma_mnode.owner
        assert moudle._is_top, "reset_outputs only supports top graph"

        tree_def = list(moudle.argdef_graph_map.keys())[0]

        self._outputs[:] = outputs
        moudle.argdef_outdef_map[tree_def] = out_def

    def add_output_node(self, node: TensorNode):
        r"""Add an output node to the Graph.

        The Graph output will become a ``tuple`` after calling ``add_output_node``.
        The first element of the ``tuple`` is the original output, and the second
        is the ``node``.

        For example, the following code

        .. code-block::

            import megengine.functional as F
            import megengine.module as M
            import megengine.traced_module as tm

            class MyModule(M.Module):
                def forward(self, x):
                    x = x + 1
                    return x

            net = MyModule()

            inp = F.zeros(shape = (1, ))
            traced_module = tm.trace_module(net, inp)
            graph = traced_module.graph
            inp_node = graph.inputs[1]
            out_node = graph.outputs[0]
            graph.add_output_node(inp_node)
            graph.add_output_node(out_node)
            out = traced_module(inp)

        Will produce the following ``InternalGraph`` and ``out``::

            print(graph)
            print(out)

        .. code-block:: text

            MyModule.Graph (self, x) {
                    %2:     add_out = x.__add__(1, )
                    return add_out, x, add_out
            }
            ((Tensor([1.], device=xpux:0), Tensor([0.], device=xpux:0)), Tensor([1.], device=xpux:0))
        """
        forma_mnode = self.inputs[0]
        moudle = forma_mnode.owner
        assert moudle._is_top, "add_output_node only supports top graph"

        tree_def = list(moudle.argdef_graph_map.keys())[0]

        org_out_def = moudle.argdef_outdef_map[tree_def]
        org_outs = org_out_def.unflatten(self._outputs)
        outputs, out_def = tree_flatten(
            (org_outs, node), is_leaf=lambda x: isinstance(x, TensorNode),
        )
        self._outputs[:] = outputs
        moudle.argdef_outdef_map[tree_def] = out_def

    def insert_exprs(self, expr: Optional[Expr] = None):
        r"""Initialize the trace mode and insertion position.

        When used within a 'with' statement, this will temporary set the trace mode and
        then restore normal mode when the with statement exits::

            with graph.insert_exprs(e): # set the trace mode
                ... # trace function or module
            ... # inert exprs into graph and resotre normal mode

        Args:
            expr: the ``expr`` after which to insert. If None, the insertion position will be
                automatically set based on the input node.

        Returns:
            A resource manager that will initialize trace mode on ``__enter__`` and
            restore normal mode on ``__exit__``.
        """
        if expr is not None:
            assert expr.top_graph == self, "Expr to insert after is not in graph."
        return _InsertExprs(self, expr)

    def replace_node(self, repl_dict: Dict[Node, Node]):
        r"""Replace the Nodes in the graph.

        Args:
            repl_dict: the map {old_Node: new_Node} that specifies how to replace the Nodes.
        """
        while repl_dict:
            node, repl_node = repl_dict.popitem()
            assert type(node) == type(
                repl_node
            ), "The type of {}({}) and {}({}) are not the same".format(
                node, type(node).__name__, repl_node, type(repl_node).__name__
            )
            # check graph inputs and outputs
            for i, n in enumerate(self.outputs):
                if n is node:
                    self.outputs[i] = repl_node
            # update users of node and repl_node
            # update inputs of expr in node.users
            graph = repl_node.top_graph
            assert graph is not None
            assert graph is self
            index = -1
            if not isinstance(repl_node.expr, Input):
                index = graph._exprs.index(repl_node.expr)
            dep_exprs = self.get_dep_exprs(repl_node)
            i = 0
            while i < len(node.users):
                n = node.users[i]
                if n in graph._exprs and index >= graph._exprs.index(n):
                    i += 1
                    continue
                if n in dep_exprs:
                    logger.info("Find a loop: ignore this replacement once")
                    logger.info("node: %s" % node.__repr__())
                    logger.info("expr: %s" % n.__repr__())
                    i += 1
                    continue
                repl_node.users.append(n)
                node.users.pop(i)
                idx = n.inputs.index(node)
                n.inputs[idx] = repl_node

    def _merge_getattr_expr(self):
        getattr_nodes_map = dict()  # Dcit[(Node, str), Node]
        node_to_attrname = dict()  # Dict[Node, (Node, Str)]
        for expr in filter(lambda x: isinstance(x, GetAttr), self._exprs):
            base_node, attr_name = expr.inputs[0], expr.name
            if expr.inputs[0] in node_to_attrname:
                base_node, base_name = node_to_attrname[expr.inputs[0]]
                attr_name = "{}.{}".format(base_name, expr.name)

            if get_suffix_name(self.qualname, expr.outputs[0].qualname) != attr_name:
                expected_qualname = base_node.qualname + "." + attr_name
                logger.warning(
                    "{}.qualname expects {}, got {} actually. You can re-trace this "
                    "TracedModel to make the name correct.".format(
                        expr.outputs[0], expected_qualname, expr.outputs[0].qualname
                    )
                )
                expr.outputs[0]._qualname = expected_qualname

            key = (base_node, attr_name)
            node_to_attrname[expr.outputs[0]] = key
            if key in getattr_nodes_map:
                existed_node = getattr_nodes_map[key]
                repl_node = expr.outputs[0]
                for expr in repl_node.users:
                    existed_node.users.append(expr)
                    idx = expr.inputs.index(repl_node)
                    expr.inputs[idx] = existed_node
                repl_node.users = []
            else:
                if attr_name != expr.name:
                    expr.name = attr_name
                    expr.inputs[0].users.remove(expr)
                    self.inputs[0].users.append(expr)
                    expr.inputs[0] = self.inputs[0]
                getattr_nodes_map[key] = expr.outputs[0]

    def compile(self):
        r"""Delete unused expr."""
        self._merge_getattr_expr()
        dep_exprs = self.get_dep_exprs(self.outputs)
        i = 0
        while i < len(self._exprs):
            expr = self._exprs[i]
            if expr in dep_exprs or expr._disable_remove:
                i += 1
                continue
            for n in expr.inputs:
                n.users.remove(expr)
            self._exprs.remove(expr)
            for n in expr.outputs:
                self._namespace.unassociate_name_with_obj(n)

    def _reset_ids(self):
        for total_expr_id, expr in enumerate(self.exprs()):
            expr._id = total_expr_id
        for total_node_id, node in enumerate(self.nodes()):
            node._id = total_node_id
        self._total_ids = (total_node_id + 1, total_expr_id + 1)

    def _re_associate_name(self):
        self._namespace.used_names.clear()
        for node in self.nodes(False):
            node._name = self._namespace.create_unique_name(node.name, node)

    def interpret(self, *inputs):
        node2value = {}
        end_nodes_set = set(self._end_point)
        endnode2value = {}

        def get_all_endnode_val(n, v):
            if n in end_nodes_set:
                endnode2value[n] = v
                end_nodes_set.remove(n)
                return not end_nodes_set
            return False

        ref_count = lambda n: len(n.users) + (1 if n in self._outputs else 0)

        for n, v in zip(self._inputs, inputs):
            if ref_count(n) > 0:
                node2value[n] = [v, ref_count(n)]
            if n in self._watch_point:
                self._rst[n].append(v)
            if n in self._end_point and get_all_endnode_val(n, v):
                return list(endnode2value[i] for i in self._end_point)

        for expr in self._exprs:
            values = expr.interpret(*list(node2value[i][0] for i in expr.inputs))
            for n in expr.inputs:
                node2value[n][1] -= 1
                if node2value[n][1] == 0:
                    node2value.pop(n)
            if values is not None:
                assert len(values) == len(expr.outputs)
                for n, v in zip(expr.outputs, values):
                    if ref_count(n) > 0:
                        node2value[n] = [v, ref_count(n)]
                    if n in self._watch_point:
                        self._rst[n] = v
                    if self._end_point and get_all_endnode_val(n, v):
                        return list(endnode2value[i] for i in self._end_point)

        return list(node2value[i][0] for i in self._outputs)

    def eval(self, *inputs: Tuple[Tensor]):
        r"""Call this method to execute the graph.

        Args:
            inputs: the tensors corresponding to the ``graph.inputs[1:]``.
        """
        assert len(inputs) == len(self._inputs) - 1
        inp = [self._inputs[0].owner] + list(inputs)
        return self.interpret(*inp)

    def __repr__(self):
        return self.__format__()

    def __format__(self, format_spec: str = "") -> str:
        saved_format_spec = Node._set_format_spec(format_spec)
        name = ""
        if self._name:
            name = "%s.Graph" % self._name
        res = "{} ({}) {{\n\t{}\n\treturn {}\n}}".format(
            name,
            ", ".join(str(i) for i in self._inputs),
            "\n\t".join("{}".format(str(i)) for i in self._exprs),
            ", ".join(str(i) for i in self._outputs),
        )
        Node._set_format_spec(saved_format_spec)
        return res

    def __getstate__(self):
        state = {
            "_exprs": self._exprs,
            "_inputs": self._inputs,
            "_outputs": self._outputs,
            "_watch_point": [],
            "_end_point": [],
            "_namespace": self._namespace,
            "_rst": collections.defaultdict(list),
            "_name": self._name,
            "_qualname": self._qualname,
        }
        if self._total_ids:
            state["_total_ids"] = self._total_ids

        _check_obj_attr(state)

        return state

    def __setstate__(self, state):
        old_version = False

        if "_module_name" in state:
            old_version = True
            state["_qualname"] = state.pop("_module_name")
            prefix_name = state.pop("_prefix_name")
            if prefix_name:
                state["_name"] = "{}_{}".format(prefix_name, state["_name"])

        self.__dict__.update(state)

        if old_version:
            self.inputs[0]._qualname = self._qualname
            for e in self.exprs(False):
                if isinstance(e, GetAttr):
                    e.outputs[0]._qualname = "{}.{}".format(
                        e.inputs[0]._qualname, e.name
                    )

            for n in self.nodes(False):
                if isinstance(n.expr, CallMethod) and isinstance(
                    n.expr.inputs[0], ModuleNode
                ):
                    n._qualname = n.expr.inputs[0]._qualname + ".[out]"
                    continue
                if (
                    not isinstance(n.expr, GetAttr)
                    and isinstance(n, TensorNode)
                    and n._qualname
                ):
                    n._qualname = "{}.{}".format(self._qualname, n._qualname)
            self._namespace = NameSpace(self._name, self._qualname)
            self._re_associate_name()

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        with max_recursion_limit():
            if id(self) in memo:
                return memo[id(self)]
            cls = self.__class__
            result = cls.__new__(cls)
            state = {}
            memo[id(self)] = result
            for k, v in self.__dict__.items():
                if not isinstance(v, weakref.ReferenceType):
                    state[k] = copy.deepcopy(v, memo)
            result.__dict__.update(state)
            return result


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
        method_func = kwargs.pop("method_func", wrapped_fn)
        if not is_tracing_module():
            return orig_func(*args, **kwargs)

        with _exclude_from_trace():
            inputs, tree_def = tree_flatten((args, kwargs))
            for i in inputs:
                if not NodeMixin.get(i, None):
                    if isinstance(i, (RawTensor, NodeMixin)):
                        NodeMixin.wrap_safe(i, Constant.make(i))
            args, kwargs = _convert_kwargs_to_args(orig_func, args, kwargs)
            meth_name = _get_meth_name(args[0], method_func)
            arg_type = args[0] if isinstance(args[0], type) else type(args[0])
            if meth_name and arg_type and issubclass(arg_type, RawTensor):
                inputs, tree_def = tree_flatten((args, kwargs))
                self = inputs[0]
                if meth_name == "__new__":
                    if all([not isinstance(i, RawTensor) for i in inputs]):
                        # only trace Tensor.__new__() when there are tensors in args
                        return orig_func(*args, **kwargs)
                    if isinstance(args[1], RawTensor):
                        node = NodeMixin.get(inputs[1])
                        inputs[1] = apply(
                            Copy(comp_node=inputs[1].device), Tensor(inputs[1])
                        )[0]
                        # copy inputs[1] to avoid tensor and Tensor(tensor) share same m_tensor,
                        # which will cause they have same _NodeMixin__node in tracing.
                        NodeMixin.wrap_safe(inputs[1], node)
                        args, kwargs = tree_def.unflatten(inputs)
                    call_node = CallMethod.make(self, meth_name)
                else:
                    call_node = CallMethod.make(NodeMixin.get(self), meth_name)
                call_node.add_inputs(inputs[1:])
            else:
                inputs, tree_def = tree_flatten((args, kwargs))
                call_node = CallFunction.make(orig_func)
                call_node.add_inputs(inputs)

            call_node.arg_def = tree_def
            rst = orig_func(*args, **kwargs)
            if meth_name == "__setitem__":
                rst = self
            if rst is not None:
                outputs, out_def = tree_flatten(rst, is_leaf=_is_leaf)
                call_node.out_def = out_def
            else:
                outputs = None
            call_node.add_outputs(outputs)

            if _get_expr_checker():
                with _exclude_from_trace():
                    active_module_tracer().checker.check_expr_interpret(
                        call_node, outputs
                    )

            return rst

    return wrapped_fn


class TracedModuleBuilder(NodeMixin):

    _mod = None  # type: Module
    _body = None  # type: InternalGraph
    _is_builtin = None  # type: bool
    _argdef_graph_map = None  # type: Dict[TreeDef, "InternalGraph"]
    _argdef_outdef_map = None  # type: Dict[TreeDef, TreeDef]
    nodes = None

    __builder_attributes__ = [
        "_mod",
        "_body",
        "_NodeMixin__node",
        "_is_builtin",
        "build",
        "_record_wrapped_nodes",
        "_argdef_graph_map",
        "_argdef_outdef_map",
        "_check_qat_module",
        "nodes",
        "__class__",
        "__dict__",
        "_is_top",
    ]

    def __init__(self, mod, is_top_module=False):
        super(TracedModuleBuilder, self).__init__()
        assert isinstance(mod, Module)
        self._mod = mod
        self._body = None
        self._is_top = is_top_module
        self._is_builtin = (
            True
            if isinstance(mod, (Observer, _FakeQuantize))
            else module_tracer.is_builtin(mod)
        )
        if isinstance(self._mod, QATModule):
            with _exclude_from_trace():
                self._check_qat_module(self._mod)
        self._argdef_graph_map = {}
        self._argdef_outdef_map = {}

        self.nodes = set()
        # The builder will be passed to self._mod.forward as 'self' argument. If the 'forward' uses super().xxx to call method of its base classes, the trace procedure will throw exceprion, because the builder doesn't inherit from self._mod.__bases__.
        # modify self.__class__ and let the builder inherit from TracedModuleBuilder and mod.__class__.
        self.__class__ = type(
            "TracedModuleBuilder",
            (TracedModuleBuilder, mod.__class__),
            dict(TracedModuleBuilder.__dict__),
        )

    def _check_qat_module(self, qat_module):
        def isbuiltin(m):
            return m is None or module_tracer.is_builtin(m)

        if qat_module.with_act:
            act_observer = qat_module.act_observer
            act_fakequant = qat_module.act_fake_quant
            if not isbuiltin(act_observer) or not isbuiltin(act_fakequant):
                qparams = (
                    act_observer.get_qparams()
                    if hasattr(act_observer, "get_qparams")
                    else act_fakequant.get_qparams()
                )
                dtype = (
                    act_observer.dtype
                    if hasattr(act_observer, "dtype")
                    else act_fakequant.dtype
                )
                qat_module.act_observer = None
                qat_module.act_fake_quant = TM_FakeQuant(dtype)
                qat_module.act_fake_quant.set_qparams(qparams)

        if qat_module.with_weight:
            weight_observer = qat_module.weight_observer
            weight_fakequant = qat_module.weight_fake_quant
            if not isbuiltin(weight_observer) or not isbuiltin(weight_fakequant):
                qparams = (
                    weight_observer.get_qparams()
                    if hasattr(weight_observer, "get_qparams")
                    else weight_fakequant.get_qparams()
                )
                dtype = (
                    weight_observer.dtype
                    if hasattr(weight_observer, "dtype")
                    else weight_fakequant.dtype
                )
                qat_module.weight_observer = None
                qat_module.weight_fake_quant = TM_FakeQuant(dtype)
                qat_module.weight_fake_quant.set_qparams(qparams)

    def build(self):
        if self._is_builtin:
            assert module_tracer.is_builtin(self._mod)
            mod_type = type(self._mod)

            for node in self.nodes:
                node.module_type = mod_type

            return self._mod
        elif isinstance(self._mod, TracedModule) and _graph_surgery_mode():
            return self._mod
        else:
            is_qat = isinstance(self._mod, QATModule) or (
                isinstance(self._mod, TracedModule) and self._mod.is_qat
            )
            traced_module = TracedModule(
                self._is_top, self._argdef_graph_map, self._argdef_outdef_map, is_qat
            )
            for _, g in self._argdef_graph_map.items():
                g.compile()
                if self._is_top:
                    g._total_ids = (Node._get_next_id(), Expr._get_next_id())

            for k, v in self.__dict__.items():
                if k not in TracedModuleBuilder.__builder_attributes__:
                    if isinstance(v, TracedModuleBuilder):
                        v = v.build()
                        setattr(traced_module, k, v)
                    elif isinstance(v, RawTensor):
                        setattr(traced_module, k, v)

            if isinstance(self._mod, QATModule):
                with _exclude_from_trace():
                    traced_module.with_act = self._mod.with_act
                    traced_module.with_weight = self._mod.with_weight
                    if not hasattr(traced_module, "act_fake_quant"):
                        traced_module.act_fake_quant = None
                    if not hasattr(traced_module, "act_observer"):
                        traced_module.act_observer = None
                    if not hasattr(traced_module, "weight_fake_quant"):
                        traced_module.weight_fake_quant = None
                    if not hasattr(traced_module, "weight_observer"):
                        traced_module.weight_observer = None

            if self._is_top:
                traced_module._update_ref()

            return traced_module

    def _record_wrapped_nodes(self, node):
        self.nodes.add(node)

    def __call__(self, *args, **kwargs):
        assert isinstance(self._mod, Module)
        is_graph_surgery_mode = _graph_surgery_mode()
        if isinstance(self._mod, TracedModule) and is_graph_surgery_mode:
            _set_graph_surgery_mode(False)

        # prepare args and kwargs for inner graph
        if "method_func" in kwargs:
            kwargs.pop("method_func")
        args, kwargs = _convert_kwargs_to_args(self._mod.forward, args, kwargs, True)

        def mark_constant(x):
            node = NodeMixin.get(x, None)
            if node is None:  # capture as constant
                NodeMixin.wrap(x, lambda: Constant.make(x))

        inputs, tree_def = tree_flatten(((self, *args), kwargs))
        for i in inputs:
            mark_constant(i)
        callnode = CallMethod.make(NodeMixin.get(self))

        callnode.add_inputs(inputs[1:])

        callnode.arg_def = tree_def

        if self._is_builtin or tree_def in self._argdef_graph_map:
            with _exclude_from_trace():
                rst = self._mod(*args, **kwargs)
                outputs, out_def = tree_flatten(rst, is_leaf=_is_leaf)
                if _get_expr_checker():
                    tmp = self.build()
                    active_module_tracer().checker.check_builtin_module(
                        tmp, callnode, outputs
                    )
            if self._is_builtin:
                self._body = None
            elif tree_def in self._argdef_graph_map:
                self._body = self._argdef_graph_map[tree_def]
        else:
            orig_self = NodeMixin.get(self)
            parent_graph = active_module_tracer().current_scope()
            module_qualname = orig_self._qualname
            self._body = InternalGraph(
                name=parent_graph._namespace.create_unique_name(module_qualname),
                qualname=module_qualname,
            )
            parent_graph._namespace.associate_name_with_obj(self._body.name, self._body)
            active_module_tracer().push_scope(self._body)
            # rebind self to new input node

            NodeMixin.wrap_safe(
                self,
                Input.make(
                    name="self",
                    qualname=module_qualname,
                    type=NodeMixin.get_wrapped_type(self),
                ),
            )

            origin_inp_node = [NodeMixin.get(i, None) for i in inputs[1:]]
            # prepare args and kwargs for inner graph
            index_args, index_kwargs = tree_def.unflatten(
                [
                    ArgsIndex(0),
                    *list(ArgsIndex(i + 1) for i in range(len(origin_inp_node))),
                ]
            )
            key2idx = getcallargs(type(self._mod).forward, *index_args, **index_kwargs)
            idx2key = {}
            for k, v in key2idx.items():
                if isinstance(v, ArgsIndex):
                    idx2key[v.index] = k
                else:
                    flatten_argidx, _ = tree_flatten(v)
                    for _i, v in enumerate(flatten_argidx):
                        if isinstance(v, ArgsIndex):
                            idx2key[v.index] = k + "_%d" % _i

            def wrap(x, name):
                if isinstance(x, (RawTensor, NodeMixin)):
                    NodeMixin.wrap(
                        x,
                        lambda: Input.make(
                            type=NodeMixin.get_wrapped_type(x),
                            name=name,
                            qualname="%s.[%s]" % (module_qualname, name),
                        ),
                    )
                return x

            args = [self]
            orig_traced_inputs = (
                None
                if not isinstance(self._mod, TracedModule)
                else self._mod.argdef_graph_map[tree_def].inputs
            )
            ind = 1
            for v in inputs[1:]:
                if isinstance(v, (RawTensor, NodeMixin)):
                    args_name = (
                        orig_traced_inputs[ind]._name
                        if orig_traced_inputs
                        else idx2key[ind]
                    )
                    ind += 1
                    args.append(wrap(v, args_name))
                else:
                    args.append(v)

            args, kwargs = tree_def.unflatten(args)
            active_module_tracer().patcher.auto_patch(
                getattr(getattr(self._mod, "forward", self._mod), "__globals__", {})
            )
            rst = type(self._mod).forward(*args, **kwargs)

            if _graph_surgery_mode():
                rst = _node_to_tensor(rst)[0][0]

            outputs, out_def = tree_flatten(rst, is_leaf=_is_leaf)

            for i in (
                outputs if isinstance(outputs, collections.abc.Sequence) else (outputs,)
            ):
                mark_constant(i)
                active_module_tracer().current_scope()._add_output(NodeMixin.get(i))

            NodeMixin.wrap_safe(self, orig_self)
            for arg, node in zip(inputs[1:], origin_inp_node):
                if node:
                    NodeMixin.wrap_safe(arg, node)
            active_module_tracer().pop_scope()

        # rebind output to outer graph
        callnode.out_def = out_def
        callnode.add_outputs(outputs)
        self._argdef_graph_map[callnode.arg_def] = self._body
        self._argdef_outdef_map[callnode.arg_def] = out_def
        _set_graph_surgery_mode(is_graph_surgery_mode)
        return rst

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

    def __repr__(self):
        return repr(self._mod)

    def __getattr__(self, name):
        if name not in self._mod.__dict__:
            attr = getattr(type(self._mod), name).__get__(self, type(self))
        else:
            attr = getattr(self._mod, name)
            if (
                isinstance(attr, FunctionType)
                and id(attr) in active_module_tracer().patcher.patched_fn_ids
            ):
                return active_module_tracer().patcher.wrap_fn(attr)

            if isinstance(attr, (List, Dict)):
                flag = _set_graph_surgery_mode(False)
                with _exclude_from_trace():
                    has_module, m_container = replace_container_with_module_container(
                        attr
                    )
                    if m_container:
                        attr = m_container
                    if has_module and not m_container:
                        raise ValueError(
                            "Can not trace the module that uses the same container to store"
                            " Module and Non-Module objects."
                        )
                _set_graph_surgery_mode(flag)

            if isinstance(attr, Module):
                attr = TracedModuleBuilder(attr)

            if isinstance(attr, (Module, RawTensor)):
                setattr(self, name, attr)

            NodeMixin.wrap(
                attr,
                lambda: GetAttr.make(
                    NodeMixin.get(self),
                    type=NodeMixin.get_wrapped_type(attr),
                    attr_name=name,
                    name="",
                ),
            )
        return attr

    def __getattribute__(self, name):
        if name in TracedModuleBuilder.__builder_attributes__:
            return object.__getattribute__(self, name)
        else:
            wrapped = object.__getattribute__(self, name)
            class_members = dict(inspect.getmembers(self.__class__))
            if name in self._mod.__dict__:
                mod_attr = getattr(self._mod, name)
                if name in class_members:
                    if (
                        not isinstance(wrapped, TracedModuleBuilder)
                        and wrapped is not mod_attr
                    ):
                        wrapped = self.__getattr__(name)

                if isinstance(wrapped, TracedModuleBuilder):
                    if not isinstance(mod_attr, (List, Dict, QATModule)):
                        assert (
                            mod_attr is wrapped._mod
                        ), "TracedModule do not support modify module attributes, please check your code."
                if isinstance(wrapped, RawTensor):
                    assert (
                        mod_attr is wrapped
                    ), "TracedModule do not support modify tensor attributes, please check your code."

                if isinstance(wrapped, (NodeMixin, RawTensor)):
                    NodeMixin.wrap(
                        wrapped,
                        lambda: GetAttr.make(
                            NodeMixin.get(self),
                            type=NodeMixin.get_wrapped_type(wrapped),
                            attr_name=name,
                            name="",
                        ),
                    )

            return wrapped


class _expr_iter:
    def __init__(self, graph: InternalGraph, recursive: bool = True):
        self.graph = graph
        self.recursive = recursive
        self._visited_graph = set()

    def __iter__(self):
        yield from self._gen_expr(self.graph)

    def _gen_expr(self, graph: InternalGraph):
        visit_inp = set()
        for inp_node in graph.inputs:
            if inp_node not in visit_inp:
                yield inp_node.expr
            visit_inp.add(inp_node)

        for expr in graph._exprs:
            yield expr
            if (
                self.recursive
                and hasattr(expr, "graph")
                and expr.graph is not None
                and id(expr.graph) not in self._visited_graph
            ):
                self._visited_graph.add(id(expr.graph))
                yield from self._gen_expr(expr.graph)


class _node_iter:
    def __init__(self, graph: InternalGraph, recursive: bool = True) -> None:
        nodes = []
        node_ids = set()
        for expr in graph.exprs(recursive):
            for n in expr.outputs:
                assert id(n) not in node_ids
                nodes.append(n)
                node_ids.add(id(n))
        self.nodes = nodes

    def __iter__(self):
        for node in self.nodes:
            yield node


class BaseFilter:
    r"""``BaseFilter`` exposes some methods for converting ``_node_iter/_expr_iter`` to ``list``, ``dict``, etc."""

    def __init__(self, iter: Iterable):
        self._iter = iter

    def __iter__(self):
        return iter(self._iter)

    def as_list(self):
        r"""Consume this iterator and return its content as a list.

        Returns:
            A list of ``Node`` or ``Expr``.
        """
        return list(self)

    def as_dict(self):
        r"""Construct an ordered dict to map from ``id`` to objects in this iterator.

        Returns:
            An :class:`OrderedDict`.
        """
        return collections.OrderedDict((i._id, i) for i in self)

    def as_unique(self):
        """Assert that this iterator yields only one ``Node`` or ``Expr`` and return it.

        Rerurns:
            A ``Node`` or ``Expr``.
        """
        rst = self.as_list()
        assert len(rst) == 1, "{} elements found".format(len(rst))
        (elem,) = self
        return elem

    def as_count(self):
        r"""Consume this iterator and get the number of elements."""
        return sum(1 for _ in self)


class ExprFilter(BaseFilter):
    """Filter on Expr iterator.
    This class is an iterator of :class:`.Expr` objects and multiple
    filtering conditions and mappers can be chained.
    """

    def call_function(self, func):
        r"""Filter by specific ``CallFunction.func``.
        See :meth:`~.InternalGraph.get_function_by_type` for details.
        """
        return ExprFilterCallFunction(self, func)

    def call_method(self, method):
        r"""Filter by specific ``CallMethod.method``.
        See :meth:`~.InternalGraph.get_function_by_type` for details.
        """
        return ExprFilterCallMethod(self, method)

    def expr_id(self, expr_id: List[int]):
        r"""Filter Exprs by their ``id``.
        See :meth:`~.InternalGraph.get_function_by_type` for details.
        """
        return ExprFilterExprId(self, expr_id)


class NodeFilter(BaseFilter):
    """Filter on Node iterator.
    This class is an iterator of :class:`~.traced_module.Node` objects and multiple
    filtering conditions and mappers can be chained.
    """

    def type(self, owner_type):
        r"""Filter by specific Module type.
        See :meth:`~.InternalGraph.get_module_by_type` for details.
        """
        return NodeFilterType(self, owner_type)

    def node_id(self, node_id: List[int]):
        r"""Filter Nodes by their ``id``.
        See :meth:`~.InternalGraph.get_node_by_id` for details.
        """
        return NodeFilterNodeId(self, node_id)

    def name(self, name: str, ignorecase: bool = True):
        r"""Filter Nodes by their full name.
        See :meth:`~.InternalGraph.get_node_by_name` for details.
        """
        return NodeFilterName(self, name, ignorecase)


class NodeFilterType(NodeFilter):
    """See :meth:`~.InternalGraph.get_module_by_type`"""

    def __init__(self, expr_iter, owner_type):
        super().__init__(expr_iter)
        self.owner_type = owner_type

    def __iter__(self):
        for node in self._iter:
            if not isinstance(node, ModuleNode):
                continue
            if not hasattr(node, "owner"):
                continue
            if isinstance(node.owner, self.owner_type):
                yield node


class NodeFilterNodeId(NodeFilter):
    """See :meth:`~.InternalGraph.get_node_by_id`"""

    def __init__(self, expr_iter, node_id: List[int]):
        super().__init__(expr_iter)
        if not isinstance(node_id, Sequence):
            node_id = [node_id]
        self.node_id = node_id

    def __iter__(self):
        for node in self._iter:
            if node._id in self.node_id:
                yield node


class NodeFilterName(NodeFilter):
    """See :meth:`~.InternalGraph.get_node_by_name`"""

    _re = None

    def __init__(self, node_iter, pattern, ignorecase):
        super().__init__(node_iter)
        self.pattern = pattern
        self._re = self.make_re(pattern, ignorecase)

    @classmethod
    def make_re(cls, pattern, ignorecase=True):
        assert isinstance(pattern, str), "bad pattern: {!r}".format(pattern)
        assert isinstance(ignorecase, bool)
        flags = 0
        if ignorecase:
            flags |= re.IGNORECASE
        return re.compile(fnmatch.translate(pattern), flags=flags)

    def __iter__(self):
        for i in self._iter:
            graph = i.top_graph
            name = "{}_{}".format(graph._name, i._name)
            if self.pattern == name or self._re.match(name):
                yield i


class ExprFilterCallFunction(ExprFilter):
    """See :meth:`~.InternalGraph.get_function_by_type`"""

    def __init__(self, expr_iter, func: Callable = None):
        super().__init__(expr_iter)
        self.func = func

    def __iter__(self):
        for expr in self._iter:
            if not isinstance(expr, CallFunction):
                continue
            if self.func is None or expr.func == self.func:
                yield expr


class ExprFilterCallMethod(ExprFilter):
    """See :meth:`~.InternalGraph.get_method_by_type`"""

    def __init__(self, expr_iter, method: str = None):
        super().__init__(expr_iter)
        self.method = method

    def __iter__(self):
        for expr in self._iter:
            if not isinstance(expr, CallMethod):
                continue
            if self.method is None or expr.method == self.method:
                yield expr


class ExprFilterExprId(ExprFilter):
    """See :meth:`~.InternalGraph.get_expr_by_id`"""

    def __init__(self, expr_iter, expr_id: List[int]):
        super().__init__(expr_iter)
        if not isinstance(expr_id, Sequence):
            expr_id = [expr_id]
        self.expr_id = expr_id

    def __iter__(self):
        for expr in self._iter:
            if expr._id in self.expr_id:
                yield expr


class TracedModule(Module):
    r"""``TracedModule`` is the Module created by tracing normal module.

    It owns an argdef to graph(InternalGraph) map. The forward method of ``TracedModule``
    will get a graph from ``argdef_graph_map`` according to the argdef of input ``args/kwargs``
    and interpret it.

    .. note::
        ``TracedModule`` can only be created by :func:`~.trace_module`. See :func:`~.trace_module`
        for more details.
    """
    # m_node = None  # type: ModuleNode
    argdef_graph_map = None
    argdef_outdef_map = None

    def __init__(self, is_top, argdef_graph_map, argdef_outdef_map, is_qat=False):
        super(TracedModule, self).__init__()
        self.argdef_graph_map = argdef_graph_map
        self.argdef_outdef_map = argdef_outdef_map
        self._is_top = is_top
        self.watch_points = []
        self.watch_node_value = {}
        self.end_points = []
        self.is_qat = is_qat
        self.argspec = None

    def forward(self, *args, **kwargs):
        if hasattr(self, "argspec") and self.argspec is not None:
            args, kwargs = _convert_kwargs_to_args(self.argspec, args, kwargs, True)
        inputs, treedef = tree_flatten(((self, *args), kwargs))
        assert (
            treedef in self.argdef_graph_map
        ), "support input args kwargs format: \n{}, but get: \n{}".format(
            "\n ".join(
                "forward({})".format(i._args_kwargs_repr())
                for i in self.argdef_graph_map.keys()
            ),
            treedef._args_kwargs_repr(),
        )
        inputs = filter(
            lambda i: isinstance(i, (Module, TracedModuleBuilder, RawTensor)), inputs
        )  # allow TracedModuleBuilder for retrace.
        outputs = self.argdef_graph_map[treedef].interpret(*inputs)
        if self.watch_points:
            self.watch_node_value = {}
            for n in self.watch_points:
                self.watch_node_value[n] = n.top_graph._rst.pop(n)

        if self.end_points:
            return outputs

        out_def = self.argdef_outdef_map[treedef]
        outputs = out_def.unflatten(outputs)

        return outputs

    def set_watch_points(self, nodes):
        r"""Initialize the :attr:`~.TracedModule.watch_points`.

        You can call this function to get the ``Tensor/Module`` corresponding to a ``Node`` at runtime.

        Args:
            nodes: a list of ``Node``.

        For example, the following code

        .. code-block::

            import megengine.module as M
            import megengine as mge
            import megengine.traced_module as tm

            class MyModule(M.Module):
                def forward(self, x):
                    x = x + 1 + 2
                    return x

            net = MyModule()

            inp = mge.Tensor([0])
            traced_module = tm.trace_module(net, inp)
            add_1_node = traced_module.graph.get_node_by_id(2).as_unique()
            traced_module.set_watch_points(add_1_node)

            out = traced_module(inp)

        Will get the following ``watch_node_value``::

            print(traced_module.watch_node_value)

        .. code-block:: text

            {add_out: Tensor([1.], device=xpux:0)}
        """
        if not isinstance(nodes, Sequence):
            nodes = [nodes]
        self.watch_points = nodes
        if nodes:
            nodes[0].top_graph._watch_point = []
        for n in nodes:
            n.top_graph._watch_point.append(n)

    def clear_watch_points(self):
        r"""Clear the :attr:`~.TracedModule.watch_points` and :attr:`~.TracedModule.watch_node_value`.
        """
        for n in self.watch_points:
            n.top_graph._watch_point = []
        self.watch_points = []
        self.watch_node_value = {}

    def set_end_points(self, nodes: Sequence[Node]):
        r"""Initialize the :attr:`~.TracedModule.end_points`.

        When all the ``nodes`` are generated, the Module will stop execution and return directly.

        Args:
            nodes: a list of ``Node``.

        For example, the following code

        .. code-block::

            import megengine.module as M
            import megengine as mge
            import megengine.traced_module as tm

            class MyModule(M.Module):
                def forward(self, x):
                    x = x + 1 + 2
                    return x

            net = MyModule()

            inp = mge.Tensor([0])
            traced_module = tm.trace_module(net, inp)
            add_1_node = traced_module.graph.get_node_by_id(2).as_unique()
            traced_module.set_end_points(add_1_node)

            out = traced_module(inp)

        Will get the following ``out``::

            print(out)

        .. code-block:: text

            [Tensor([1.], device=xpux:0)]
        """
        if not isinstance(nodes, Sequence):
            nodes = [nodes]
        self.end_points = nodes
        graphs = list(self.argdef_graph_map.values())
        for n in nodes:
            assert n.top_graph in graphs
            n.top_graph._end_point.append(n)

    def clear_end_points(self):
        r"""Clear the :attr:`~.TracedModule.end_points`.
        """
        for n in self.end_points:
            n.top_graph._end_point = []
        self.end_points = []

    @property
    def graph(self) -> InternalGraph:
        """Return the ``InternalGraph`` of this ``TracedModule``.
        """
        assert len(self.argdef_graph_map) == 1
        return list(self.argdef_graph_map.values())[0]

    def _update_ref(self, actual_node_map: Union[Dict] = None, top_graph=None):
        for inp_def, graph in self.argdef_graph_map.items():
            if top_graph is not None:
                graph._top_graph = weakref.ref(top_graph)
            for n in graph._inputs + graph._outputs:
                n.expr._top_graph = weakref.ref(graph)
                n._top_graph = weakref.ref(graph)
            graph._inputs[0]._owner = weakref.ref(self)
            for i, n in enumerate(graph._inputs):
                n.actual_node = []
                if actual_node_map is not None and inp_def in actual_node_map.keys():
                    n.actual_node = list(list(zip(*(actual_node_map[inp_def])))[i])
            node2obj = {}
            next_actual_node_map = collections.defaultdict(
                lambda: collections.defaultdict(list)
            )
            node2obj[graph._inputs[0]] = self
            for expr in graph._exprs:
                for n in expr.inputs + expr.outputs:
                    n._top_graph = weakref.ref(graph)
                expr._top_graph = weakref.ref(graph)
                if isinstance(expr, GetAttr) and isinstance(
                    expr.outputs[0], ModuleNode
                ):
                    obj = expr.interpret(node2obj[expr.inputs[0]])[0]
                    expr.outputs[0]._owner = weakref.ref(obj)
                    node2obj[expr.outputs[0]] = obj
                if isinstance(expr, Constant) and isinstance(
                    expr.outputs[0], ModuleNode
                ):
                    obj = expr.value
                    expr.outputs[0]._owner = weakref.ref(obj)
                    node2obj[expr.outputs[0]] = obj
                if (
                    isinstance(expr, CallMethod)
                    and expr.method == "__call__"
                    and isinstance(expr.inputs[0], ModuleNode)
                ):
                    obj = node2obj[expr.inputs[0]]
                    if expr.arg_def is not None:
                        next_actual_node_map[obj][expr.arg_def].append(expr.inputs)

            for obj in node2obj.values():
                if obj is self:
                    continue
                mnode_map = None
                if obj in next_actual_node_map.keys():
                    mnode_map = next_actual_node_map[obj]
                if isinstance(obj, TracedModule):
                    obj._update_ref(mnode_map, graph)

    def flatten(self):
        r"""Get a new TracedModule, which eliminates ``GetAttr`` and has no hierarchy.

        Retruns:
            A new :class:`TracedModule`.
        """
        new_module = copy.deepcopy(self)

        def _replace_inputs_and_outputs(expr: Expr, repl_dict: Dict[Node, Node]):
            inputs, outputs = expr.inputs, expr.outputs
            for i, node in enumerate(inputs):
                if node in repl_dict:
                    inputs[i] = repl_dict[node]
            for i, node in enumerate(outputs):
                if node in repl_dict:
                    outputs[i] = repl_dict[node]
                    outputs[i].expr = expr

        def _flatten_subgraph(
            parent_graph: InternalGraph,
            graph: InternalGraph,
            call: CallMethod,
            module: Module,
        ):
            repl_dict, node2obj, rename_blacklist = {}, {}, []

            if call is not None:
                graph = copy.deepcopy(graph)
                node2obj[call.inputs[0]] = module

                repl_dict = dict(zip(graph._inputs, call.inputs))
                for ind, out in enumerate(graph.outputs):
                    if isinstance(out.expr, Input):
                        assert out in repl_dict
                        call_out = call.outputs[ind]
                        for expr in call.outputs[ind].users:
                            for index, inp in enumerate(expr.inputs):
                                if inp is call_out:
                                    expr.inputs[index] = repl_dict[out]
                                    repl_dict[out].users.append(expr)
                        if parent_graph is not None:
                            for index, parent_out in enumerate(parent_graph._outputs):
                                if parent_out is call_out:
                                    parent_graph._outputs[index] = repl_dict[out]
                        continue
                    repl_dict[out] = call.outputs[ind]
                    if isinstance(out, TensorNode):
                        call.outputs[ind]._qualname = out._qualname

                for node, repl_node in repl_dict.items():
                    assert node in graph._inputs or node in graph._outputs
                    repl_node.users.extend(node.users)

                rename_blacklist = list(chain(call.inputs, call.outputs))

            node2obj[graph._inputs[0]] = module
            prefix_name = call.inputs[0]._name if call else ""
            flattened_exprs = []

            for expr in graph._exprs:
                exprs = [expr]

                if call is not None:
                    _replace_inputs_and_outputs(expr, repl_dict)

                if isinstance(expr, GetAttr):
                    mnode = expr.inputs[0]
                    node2obj[expr.outputs[0]] = expr.interpret(node2obj[mnode])[0]

                if isinstance(expr, CallMethod):
                    obj_node = expr.inputs[0]
                    if isinstance(obj_node, ModuleNode) and isinstance(
                        obj_node.expr, GetAttr
                    ):
                        obj = node2obj[obj_node]
                        expr_graph = (
                            obj.argdef_graph_map[expr.arg_def]
                            if hasattr(obj, "argdef_graph_map")
                            else None
                        )
                        if expr_graph is not None and not obj.is_qat:
                            exprs = _flatten_subgraph(graph, expr_graph, expr, obj)

                if parent_graph is not None:
                    for node in expr.outputs:
                        name = node._name
                        if node not in rename_blacklist:
                            name = "{}_{}".format(prefix_name, name)
                        node._name = parent_graph._namespace.create_unique_name(
                            name, node
                        )

                flattened_exprs.extend(exprs)

            if call is not None:
                for i in call.inputs:
                    i.users.remove(call)

            return flattened_exprs

        new_module.graph._exprs = _flatten_subgraph(
            None, new_module.graph, None, new_module
        )
        new_module.graph._re_associate_name()
        new_module.graph.compile()
        new_module._update_ref()
        new_module.graph._reset_ids()
        return new_module

    def __getstate__(self):
        d = self.__dict__.copy()
        for k in Module.__dict__:
            d.pop(k, None)
        _check_obj_attr(d)
        for k in d:
            if module_tracer.is_builtin(d[k]):
                assert _check_builtin_module_attr(
                    d[k]
                ), "Module {} can not be serialized. ".format(type(d[k]))
                d[k] = _ModuleState.get_module_state(d[k])
        dump_info = {
            "version": __version__,
            "register_type": USER_REGISTERED_LEAF_TYPE,
            "register_container_type": USER_REGISTERED_CONTAINER_TYPE,
            "register_mdule": USER_REGISTERED_MODULE,
            "register_function": USER_REGISTERED_FUNCTION,
        }
        d["dump_info"] = dump_info
        return d

    def __setstate__(self, state):

        for k, v in state.items():
            if isinstance(v, _ModuleState):
                state[k] = v.to_module()
        super().__setstate__(state)
        self._update_ref()

        for _, graph in self.argdef_graph_map.items():
            for expr in graph._exprs:
                if isinstance(expr, CallFunction):
                    load_functional(expr)
                if isinstance(expr, CallMethod):
                    if expr.method == "__call__":
                        load_call_module_expr(expr)
                    else:
                        load_call_tensor_method_expr(expr)
                if isinstance(expr, Apply):
                    load_apply_expr(expr)

        for _, graph in self.argdef_graph_map.items():
            ind = 0
            while ind < len(graph._exprs):
                cur_expr = graph._exprs[ind]
                has_new_expr = False
                for i in cur_expr.inputs:
                    if i.expr not in graph._exprs and not isinstance(i.expr, Input):
                        graph._exprs.insert(ind, i.expr)
                        has_new_expr = True
                if not has_new_expr:
                    ind += 1
            for expr in graph._exprs:
                for i in expr.inputs:
                    if expr.inputs.count(i) != i.users.count(expr):
                        add_or_del_count = expr.inputs.count(i) - i.users.count(expr)
                        if add_or_del_count > 0:
                            i.users.extend([expr] * add_or_del_count)
                        else:
                            [i.users.remove(expr) for i in range(-add_or_del_count)]

                for o in expr.outputs:
                    if o.expr is not expr:
                        assert o not in o.expr.outputs
                        o.expr = expr
            for node in graph.nodes(False):
                # remove users of node which doesn't use node as input
                node.users = [e for e in node.users if node in e.inputs]

            for expr in graph._exprs:
                graph._namespace.auto_naming_for_outputs(expr)
        self._update_ref()
        for _, graph in self.argdef_graph_map.items():
            graph._reset_ids()

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        with max_recursion_limit():
            cls = self.__class__
            result = cls.__new__(cls)
            state = {}
            memo[id(self)] = result
            for k, v in self.__dict__.items():
                if not isinstance(v, weakref.ReferenceType):
                    state[k] = copy.deepcopy(v, memo)
            result.__dict__.update(state)
            result._update_ref()
            return result


def cpp_apply_module_trace(opdef, *args):
    return Apply.apply_module_trace_hook(opdef, *args)


USER_REGISTERED_MODULE = []
USER_REGISTERED_FUNCTION = []


def register_as_builtin(mod_cls: Type[Module]) -> None:
    r"""Registers class ``mod_cls`` (subclass of :class:`~.Module`) as builtin module.

    Args:
        mod_cls: the module class which will be treated as builtin module in tracing.
    """
    USER_REGISTERED_MODULE.append((mod_cls.__module__, mod_cls.__qualname__))
    module_tracer.register_as_builtin(mod_cls)


def wrap(func: Callable):
    r"""Call this function to register ``func`` as a builtin function.

    This function can be called at module-level scope to register ``func`` as a builtin function.
    A builtin function will be converted to a :class:`CallFunction` Expr in tracing::

        def my_func(x, y):
            return x + y

        import megengine.traced_module as tm
        tm.wrap(my_func)

    This function can also equivalently be used as a decorator::

        @tm.wrap
        def my_func(x, y):
            return x + y

    Args:
        func: the function of the global function to insert into the graph when it's called.
    """
    USER_REGISTERED_FUNCTION.append((func.__module__, func.__qualname__))
    assert callable(func), "func must be a callable"
    assert hasattr(func, "__code__")
    fn_name = func.__code__.co_name
    currentframe = inspect.currentframe()
    assert currentframe is not None
    f = currentframe.f_back
    assert f is not None
    assert (
        f.f_code.co_name == "<module>"
    ), "wrap must be called at the top level of a module"
    Patcher._builtin_functions.append((f.f_globals, fn_name))
    return func


def _register_all_builtin_module():

    for sub_mod in [M, M.qat, M.quantized, MExternal]:
        for m in getmembers(sub_mod):
            if (
                isclass(m[1])
                and issubclass(m[1], M.Module)
                and m[1] is not M.Sequential
            ):
                module_tracer.register_as_builtin(m[1])

    module_tracer.register_as_builtin(Observer)
    module_tracer.register_as_builtin(MinMaxObserver)
    module_tracer.register_as_builtin(SyncMinMaxObserver)
    module_tracer.register_as_builtin(ExponentialMovingAverageObserver)
    module_tracer.register_as_builtin(SyncExponentialMovingAverageObserver)
    module_tracer.register_as_builtin(HistogramObserver)
    module_tracer.register_as_builtin(PassiveObserver)

    module_tracer.register_as_builtin(LSQ)
    module_tracer.register_as_builtin(TQT)
    module_tracer.register_as_builtin(FakeQuantize)
    module_tracer.register_as_builtin(TM_FakeQuant)


def trace_module(
    mod: Module, *args: Tuple[Any], **kwargs: Dict[str, Any]
) -> TracedModule:
    r"""Traces module ``mod`` and returns corresponding :class:`TracedModule`.

    Args:
        mod: the module will be converted to :class:`TracedModule`.
        args: the positional arguments passed to forward method of ``mod``.
        kwargs: the keyword arguments passed to forward method of ``mod``.
    """
    assert active_module_tracer() is None
    assert isinstance(mod, Module)
    use_sym_shape = use_symbolic_shape()
    inputs = []
    try:
        net_name = mod._name if mod._name else mod.__class__.__name__
        use_sym_shape = set_symbolic_shape(True)
        set_active_module_tracer(module_tracer(_wrapped_function))
        set_module_tracing()
        for cls in [Expr, Node]:
            cls._set_next_id(0)
        with active_module_tracer().patcher:
            global_scope = InternalGraph(name="top", qualname=net_name)
            active_module_tracer().push_scope(global_scope)
            builder = TracedModuleBuilder(mod, True)

            NodeMixin.wrap_safe(
                builder, Input.make(name="top", type=ModuleNode, qualname=net_name)
            )
            forward_argspec = (
                mod.argspec
                if hasattr(mod, "argspec")
                else inspect.getfullargspec(mod.forward)
            )
            args, kwargs = _convert_kwargs_to_args(forward_argspec, args, kwargs, True)
            inputs, _ = tree_flatten((args, kwargs))
            for _, i in enumerate(inputs):
                # assert isinstance(i, Tensor), "not support "
                if isinstance(i, RawTensor):
                    NodeMixin.wrap_safe(
                        i,
                        Input.make(
                            name="arg_{}".format(_),
                            type=NodeMixin.get_wrapped_type(i),
                            qualname="{}.[{}]".format(net_name, "arg_{}".format(_)),
                        ),
                    )
            rst = builder(*copy.deepcopy(args), **copy.deepcopy(kwargs))
            active_module_tracer().pop_scope()
            traced_mod = builder.build()
            traced_mod.argspec = forward_argspec
            traced_mod.graph._reset_ids()

            has_expr_not_check = False
            if _get_expr_checker():
                has_expr_not_check = (
                    active_module_tracer().checker.check_node_not_in_scope()
                )
            if _get_default_checker() or has_expr_not_check:
                with _exclude_from_trace():
                    tm_res = traced_mod(*args, **kwargs)
                    tm_res, _ = tree_flatten(tm_res, is_leaf=_is_leaf)
                    rst, _ = tree_flatten(rst, is_leaf=_is_leaf)
                    active_module_tracer().checker.check_net_outputs(tm_res, rst)
            return traced_mod
    finally:
        set_symbolic_shape(use_sym_shape)
        unset_module_tracing()
        for t in mod.tensors(recursive=True):
            NodeMixin.clear_node(t)
        for t in inputs:
            NodeMixin.clear_node(t)
        set_active_module_tracer(None)
