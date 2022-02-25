import abc
import copy
import weakref
from importlib import import_module
from typing import Any, Dict, List, Tuple, Type

import numpy

from .. import get_logger
from ..core._imperative_rt.core2 import Tensor as RawTensor
from ..module import Module
from ..quantization.utils import QParams
from ..tensor import Tensor
from .module_tracer import active_module_tracer
from .tm_config import _get_expr_checker
from .utils import _check_obj_attr

logger = get_logger(__name__)


class Node:
    r"""``Node`` represents the variables (``Tensor``, ``Module``) used in Module's forward method.
    They are inputs/outputs of Expr (the operations on variables).
    """

    expr = None  # type: Expr
    r"""The Expr which produces the Node."""
    __total_id = 0  # type: int
    _id = None  # type: int
    _top_graph = None  # type: weakref.ReferenceType
    _format_spec = ""  # type: str

    def __init__(self, expr, name: str, qualname: str):
        self.expr = expr
        self.users = []  # List[Expr]
        self._id = Node.__total_id
        Node.__total_id += 1
        self._name = name
        self._qualname = qualname
        self.actual_node = []  # type: List[Node]

    def __repr__(self):
        format_spec = Node._format_spec
        return self.__format__(format_spec)

    def __format__(self, format_spec: str) -> str:
        if not format_spec:
            format_spec = Node._format_spec
        name = self._name
        if name is None:
            name = ""
        if format_spec in ["i", "p", "ip", "pi"]:
            if "p" in format_spec:
                prefix_name = self.top_graph._name
                name = "{}_{}".format(prefix_name, name)
            if "i" in format_spec:
                name = "%{}_{}".format(self._id, name)
            return name
        else:
            return name if name else ("%d" % self._id)

    @property
    def name(self):
        r"""Return the name of this Node."""
        return self._name

    @name.setter
    def name(self, new_name: str):
        r"""Set a new name to this Node."""
        graph = self.top_graph
        assert graph is not None, "The parent graph of this Node cannot be None."
        assert graph._namespace.used_names.get(new_name, None) is None, (
            "The name(%s) is already in use. Please try a different one again."
            % (new_name)
        )
        graph._namespace.unassociate_name_with_obj(self)
        self._name = graph._namespace.create_unique_name(new_name, self)

    @property
    def qualname(self):
        r"""Get the `qualname` of this Node. The `qualname` can be used to get the
        submodule from the traced Module or Module.

        Example:
            .. code-block::

                import megengine.module as M
                import megengine.functional as F
                import megengine.traced_module as tm
                import megengine as mge

                class block(M.Module):
                    def __init__(self):
                        super().__init__()
                        self.param = mge.Tensor([1.])
                        self.relu = M.ReLU()

                    def forward(self, x):
                        x = x + self.param
                        return self.relu(F.relu(x))

                class module(M.Module):
                    def __init__(self):
                        super().__init__()
                        self.block = block()

                    def forward(self, x):
                        x = self.block(x)
                        return x

                net = module()
                traced_net = tm.trace_module(net, mge.Tensor([0.]))
                traced_net = traced_net.flatten()
                out_node = traced_net.graph.outputs[0]

                # qualname : "module.block.relu.[out]"
                qualname = out_node.qualname
                # qualname : "block.relu"
                qualname = qualname.split(".", 1)[-1].rsplit(".", 1)[0]

                assert qualname in list(map(lambda x: x[0], net.named_modules()))
                assert qualname in list(map(lambda x: x[0], traced_net.named_modules()))
        """
        return self._qualname

    @property
    def top_graph(self):
        r"""Get the parent graph of this Node."""
        if self._top_graph:
            return self._top_graph()
        return None

    @classmethod
    def _set_format_spec(cls, str):
        old_format_spec = cls._format_spec
        cls._format_spec = str
        return old_format_spec

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
            if not isinstance(v, weakref.ReferenceType) and k != "actual_node":
                state[k] = copy.deepcopy(v, memo)
        result.__dict__.update(state)
        return result


class ModuleNode(Node):
    r"""``ModuleNode`` represents the Module objects."""

    module_type = Module  # type: Type[Module]
    r"""The type of the Module correspending to the ModuleNode."""
    _owner = None  # type: weakref.ReferenceType

    def __init__(self, expr, name: str = None, qualname: str = None):
        super().__init__(expr, name, qualname)

    def __getstate__(self):
        state = {
            "expr": self.expr,
            "users": self.users,
            "_id": self._id,
            "_name": self._name,
            "_qualname": self._qualname,
            "module_type": (self.module_type.__module__, self.module_type.__qualname__),
        }
        _check_obj_attr(state)
        return state

    def __setstate__(self, state):
        if "_orig_name" in state:
            state["_qualname"] = state.pop("_orig_name")
        self.__dict__.update(state)
        try:
            if isinstance(self.module_type, tuple):
                mname, classname = self.module_type
                mtype = getattr(import_module(mname), classname)
                self.module_type = mtype
        except Exception:
            pass

    @property
    def owner(self):
        r"""Get the ``Module`` corresponding to this ``ModuleNode``.
        """
        if self._owner:
            return self._owner()
        return None


class TensorNode(Node):
    r"""``TensorNode`` represents the Tensor objects."""

    _shape = None  # type: Tuple[int]
    _dtype = None  # type: numpy.dtype
    _qparams = None  # type: QParams
    _device = None
    _value = None  # type: Tensor

    def __init__(
        self,
        expr,
        name: str = None,
        qualname: str = None,
        shape: Tuple[int] = None,
        dtype: numpy.dtype = None,
        qparams: QParams = None,
    ):
        super().__init__(expr, name, qualname)
        self._shape = shape
        self._dtype = dtype
        self._qparams = qparams

    def __getstate__(self):
        state = {
            "expr": self.expr,
            "users": self.users,
            "_id": self._id,
            "_qparams": self._qparams,
            "_shape": self._shape,
            "_dtype": self._dtype,
            "_device": self._device,
            "_name": self._name,
            "_qualname": self._qualname,
        }
        _check_obj_attr(state)
        return state

    def __setstate__(self, state):
        if "_orig_name" in state:
            qualname = state.pop("_orig_name")
            modulepath, comma, qualname = qualname.rpartition(".")
            expr_name = state["expr"].__class__.__name__
            if expr_name not in ["GetAttr"]:
                qualname = "[{}]".format(qualname)
            if comma:
                qualname = "{}.{}".format(modulepath, qualname)
            state["_qualname"] = qualname
        self.__dict__.update(state)

    @property
    def shape(self):
        r"""Get the shape of this Node."""
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape

    @property
    def dtype(self):
        r"""Get the dtype of this Node."""
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype

    @property
    def device(self):
        r"""Get the device of this Node pointed Tensor."""
        return self._device

    @device.setter
    def device(self, device):
        self._device = device

    @property
    def qparams(self):
        r"""Get the :class:`QParams` of this Node."""
        return self._qparams

    @qparams.setter
    def qparams(self, qparams):
        self._qparams = qparams

    @property
    def value(self):
        r"""Get the bound Tensor of this Node."""
        return self._value

    @value.setter
    def value(self, value):
        r"""Bind a :class:`Tensor` to this Node."""
        if isinstance(value, RawTensor) and NodeMixin.get(value, None) is not None:
            setattr(value, "_NodeMixin__node", None)
        self._value = value


class NodeMixin(abc.ABC):
    __node = None

    @abc.abstractmethod
    def _record_wrapped_nodes(self, node):
        # record the nodes which had been bound to this NodeMixin
        pass

    @classmethod
    def _record_tensornode_property(cls, node, value):
        assert isinstance(node, TensorNode)
        assert isinstance(value, RawTensor)
        if isinstance(value, RawTensor):
            try:
                node._dtype = value.dtype
            except RuntimeError:
                node._dtype = None
            node._shape = (
                value._tuple_shape if isinstance(value, Tensor) else value.shape
            )
            node._device = value.device
            if hasattr(value, "_qparams") and value._qparams is not None:
                node._qparams = value.qparams

    @classmethod
    def wrap(cls, value, node):
        if isinstance(value, (NodeMixin, RawTensor)):
            if isinstance(node, Node):
                if isinstance(value, RawTensor):
                    cls._record_tensornode_property(node, value)
                if isinstance(value, NodeMixin):
                    value._record_wrapped_nodes(node)
                setattr(value, "_NodeMixin__node", node)
                if _get_expr_checker():
                    if isinstance(value, RawTensor):
                        active_module_tracer().checker.record_node2value(node, value)
                    if isinstance(value, NodeMixin):
                        active_module_tracer().checker.record_nodemixin(node, value)
            else:
                assert callable(node)
                n = node()
                assert isinstance(n, Node)
                if isinstance(value, RawTensor):
                    cls._record_tensornode_property(n, value)
                if isinstance(value, NodeMixin):
                    value._record_wrapped_nodes(n)
                setattr(value, "_NodeMixin__node", n)
                if _get_expr_checker():
                    if isinstance(value, RawTensor):
                        active_module_tracer().checker.record_node2value(n, value)
                    if isinstance(value, NodeMixin):
                        active_module_tracer().checker.record_nodemixin(n, value)

    @classmethod
    def wrap_safe(cls, value, node):
        assert isinstance(value, (NodeMixin, RawTensor))
        if isinstance(value, RawTensor):
            cls._record_tensornode_property(node, value)
        setattr(value, "_NodeMixin__node", node)
        if _get_expr_checker():
            if isinstance(value, RawTensor):
                active_module_tracer().checker.record_node2value(node, value)
            if isinstance(value, NodeMixin):
                active_module_tracer().checker.record_nodemixin(node, value)
        if isinstance(value, NodeMixin):
            value._record_wrapped_nodes(node)

    @classmethod
    def clear_node(cls, value):
        if hasattr(value, "_NodeMixin__node"):
            delattr(value, "_NodeMixin__node")

    @classmethod
    def get(cls, value, *default):
        return getattr(value, "_NodeMixin__node", *default)

    @classmethod
    def get_wrapped_type(cls, value):
        if isinstance(value, RawTensor):
            return TensorNode
        if isinstance(value, (Module, NodeMixin)):
            return ModuleNode
        return Node
