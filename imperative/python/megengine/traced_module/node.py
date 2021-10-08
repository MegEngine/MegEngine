# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import abc
import weakref
from typing import Any, Dict, List, Tuple, Type

import numpy

from .. import get_logger
from ..core._imperative_rt.core2 import Tensor as RawTensor
from ..module import Module
from ..tensor import Tensor

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
    _name = None  # type: str
    _orig_name = None  # type: str
    _format_spec = ""  # type: str

    def __init__(self, expr, name: str, orig_name: str):
        self.expr = expr
        self.users = []  # List[Expr]
        self._id = Node.__total_id
        Node.__total_id += 1
        self._name = name
        self._orig_name = orig_name
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
                graph = self.top_graph
                prefix_name = ""
                if graph is not None:
                    prefix_name = graph._name
                    if graph._prefix_name:
                        prefix_name = "{}_{}".format(
                            graph._prefix_name, prefix_name.lstrip("_")
                        )
                if name:
                    name = "_" + name.lstrip("_")
                name = "{}{}".format(prefix_name, name)
            if "i" in format_spec:
                if name:
                    name = "_" + name.lstrip("_")
                name = "%{}{}".format(self._id, name)
            return name
        else:
            return name if name else ("%d" % self._id)

    @property
    def name(self):
        r"""Return the name of this Node."""
        return self._name

    @name.setter
    def name(self, new_name: str):
        graph = self.top_graph
        assert graph is not None, "The parent graph of this Node cannot be None."
        assert new_name not in graph._used_names, (
            "The name(%s) is already in use. Please try a different one again."
            % (new_name)
        )
        new_name = graph._create_unique_name(new_name)
        self._name = new_name
        self._orig_name = new_name

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


class ModuleNode(Node):
    r"""``ModuleNode`` represents the Module objects."""

    module_type = Module  # type: Type[Module]
    r"""The type of the Module correspending to the ModuleNode."""
    _owner = None  # type: weakref.ReferenceType

    def __init__(self, expr, name: str = None, orig_name: str = None):
        super().__init__(expr, name, orig_name)

    def __getstate__(self):
        return {
            "expr": self.expr,
            "users": self.users,
            "_id": self._id,
            "_name": self._name,
            "_orig_name": self._orig_name,
            "module_type": self.module_type,
        }

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
    _qparams = None
    _device = None
    _value = None  # type: Tensor

    def __getstate__(self):
        return {
            "expr": self.expr,
            "users": self.users,
            "_id": self._id,
            "_qparams": self._qparams,
            "_shape": self._shape,
            "_dtype": self._dtype,
            "_device": self._device,
            "_name": self._name,
            "_orig_name": self._orig_name,
        }

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
            node._dtype = value.dtype
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
            else:
                assert callable(node)
                n = node()
                assert isinstance(n, Node)
                if isinstance(value, RawTensor):
                    cls._record_tensornode_property(n, value)
                if isinstance(value, NodeMixin):
                    value._record_wrapped_nodes(n)
                setattr(value, "_NodeMixin__node", n)

    @classmethod
    def wrap_safe(cls, value, node):
        assert isinstance(value, (NodeMixin, RawTensor))
        if isinstance(value, RawTensor):
            cls._record_tensornode_property(node, value)
        setattr(value, "_NodeMixin__node", node)
        if isinstance(value, NodeMixin):
            value._record_wrapped_nodes(node)

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
