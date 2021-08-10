# -*- coding: utf-8 -*-
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

from ...core._imperative_rt.core2 import Tensor as RawTensor
from ...module import Module
from ...tensor import Tensor


class Node:
    """
    ``Node`` represents the variables （Tensor/Module/other python object) used in Module's forward method. They are inputs/outputs of Expr(the operations on variables).

    param expr: the Expr which produces the node
    param name: the name of the node
    """

    expr = None
    __total_id = 0
    _id = None
    _top_graph = None  # type: weakref.ReferenceType
    _name = None
    _orig_name = None
    _format_spec = ""

    def __init__(self, expr: "Expr", name: str = None, orig_name: str = None):
        self.expr = expr
        self.users = []  # List[Expr]
        self._id = Node.__total_id
        Node.__total_id += 1
        self._name = name
        self._orig_name = orig_name
        self.actual_node = []  # type: List[Node]

    def __setstate__(self, d):
        self.__dict__ = d
        Node.__total_id = max(Node.__total_id, self._id) + 1

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
    def top_graph(self):
        if self._top_graph:
            return self._top_graph()
        return None

    @classmethod
    def set_format_spec(cls, str):
        old_format_spec = cls._format_spec
        cls._format_spec = str
        return old_format_spec


class ModuleNode(Node):
    """
    ``ModuleNode`` represents the Module objects.

    Attributes:
        module_type: type of the Module correspending to the ModuleNode
        graph: the InternalGraph which will be interpreted when call Module's forward method
        attr_type_map: record the type of Module's attributes
    """

    module_type = Module  # type: Type[Module]
    _owner = None  # type: weakref.ReferenceType

    def __init__(self, expr: "Expr", name: str = None, orig_name: str = None):
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
        if self._owner:
            return self._owner()
        return None


class TensorNode(Node):
    """
    ``TensorNode`` represents the Tensor objects.
    """

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
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device

    @property
    def qparams(self):
        return self._qparams

    @qparams.setter
    def qparams(self, qparams):
        self._qparams = qparams

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
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
