# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import collections
from collections import OrderedDict
from typing import Callable, NamedTuple

import numpy as np

from ..core._imperative_rt.common import CompNode
from ..core._imperative_rt.core2 import Tensor as RawTensor
from ..core._wrap import Device
from ..core.tensor.dtype import QuantDtypeMeta
from ..module import Module
from ..quantization.utils import LSQParams, QParams, QuantMode
from ..tensor import Parameter, Tensor
from .node import ModuleNode, Node, NodeMixin, TensorNode


class ArgsIndex:
    def __init__(self, index=0, name="") -> None:
        self.index = index
        self.name = name

    def __repr__(self) -> str:
        return self.name


SUPPORTED_TYPE = {}

# if type(object) or obj in SUPPORTED_LEAF_TYPE, the object could be treated as leaf node of pytree
SUPPORTED_LEAF_TYPE = {
    RawTensor,
    Tensor,
    Parameter,
    str,
    int,
    float,
    bool,
    QuantDtypeMeta,
    CompNode,
    Device,
    type(None),
    type(Ellipsis),
    QuantMode,
    ArgsIndex,
}

# if isinstance(object, SUPPORTED_LEAF_CLS) or issubclass(obj, SUPPORTED_LEAF_CLS) is True, the object could be threated as leaf node of pytree
SUPPORTED_LEAF_CLS = [Module, Node, NodeMixin, np.dtype, np.ndarray, np.number]

NodeType = NamedTuple("NodeType", [("flatten", Callable), ("unflatten", Callable)])


def register_supported_type(type, flatten=None, unflatten=None):
    if flatten and unflatten:
        SUPPORTED_TYPE[type] = NodeType(flatten, unflatten)
    else:
        SUPPORTED_LEAF_CLS.append(type)


def _dict_flatten(inp):
    aux_data = []
    results = []
    for key, value in sorted(inp.items()):
        results.append(value)
        aux_data.append(key)
    return results, tuple(aux_data)


def _dict_unflatten(inps, aux_data):
    return dict(zip(aux_data, inps))


def _ordereddict_flatten(inp):
    aux_data = []
    results = []
    for key, value in inp.items():
        results.append(value)
        aux_data.append(key)
    return results, tuple(aux_data)


def _ordereddict_unflatten(inps, aux_data):
    return OrderedDict(zip(aux_data, inps))


def qparams_flatten(inp):
    aux_data = []
    results = []
    for key in inp.__slots__:
        aux_data.append(key)
        results.append(getattr(inp, key, None))
    return results, tuple(aux_data)


def qparams_unflatten(inp, aux_data):
    obj = QParams.__new__(QParams)
    for k, v in zip(aux_data, inp):
        setattr(obj, k, v)
    return obj


register_supported_type(list, lambda x: (x, None), lambda x, aux_data: list(x))
register_supported_type(tuple, lambda x: (x, None), lambda x, aux_data: tuple(x))
register_supported_type(dict, _dict_flatten, _dict_unflatten)
register_supported_type(
    collections.OrderedDict, _ordereddict_flatten, _ordereddict_unflatten
)
register_supported_type(
    slice,
    lambda x: ([x.start, x.stop, x.step], None),
    lambda x, aux_data: slice(x[0], x[1], x[2]),
)

register_supported_type(QParams, qparams_flatten, qparams_unflatten)


def _is_leaf(obj):
    if isinstance(obj, type):
        return issubclass(obj, tuple(SUPPORTED_LEAF_CLS)) or obj in SUPPORTED_LEAF_TYPE
    return (
        isinstance(obj, tuple(SUPPORTED_LEAF_CLS)) or type(obj) in SUPPORTED_LEAF_TYPE
    )


def _leaf_type(node):
    if isinstance(node, (RawTensor, TensorNode)):
        return (Tensor, TensorNode, ArgsIndex)
    elif isinstance(node, (NodeMixin, Module, ModuleNode)):
        return (Module, ModuleNode, NodeMixin, ArgsIndex)
    else:
        return (type(node), ArgsIndex)


def _is_const_leaf(node):
    if isinstance(node, (RawTensor, NodeMixin, Module)):
        return False
    return True


def tree_flatten(
    values,
    leaf_type: Callable = _leaf_type,
    is_leaf: Callable = _is_leaf,
    is_const_leaf: Callable = _is_const_leaf,
):
    r"""Flattens a pytree into a list of values and a :class:`TreeDef` that can be used
    to reconstruct the pytree.
    """
    if type(values) not in SUPPORTED_TYPE:
        assert is_leaf(values), values
        node = LeafDef(leaf_type(values))
        if is_const_leaf(values):
            node.const_val = values
        return [values,], node

    rst = []
    children_defs = []
    children_values, aux_data = SUPPORTED_TYPE[type(values)].flatten(values)
    for v in children_values:
        v_list, treedef = tree_flatten(v, leaf_type, is_leaf, is_const_leaf)
        rst.extend(v_list)
        children_defs.append(treedef)

    return rst, TreeDef(type(values), aux_data, children_defs)


class TreeDef:
    r"""A ``TreeDef`` represents the structure of a pytree.

    Args:
        type: the type of root Node of the pytree.
        aux_data: some const data that is useful in unflattening the pytree.
        children_defs: ``TreeDef`` for each child of the root Node.
        num_leaves: the number of leaves.
    """

    def __init__(self, type, aux_data, children_defs):
        self.type = type
        self.aux_data = aux_data
        self.children_defs = children_defs
        self.num_leaves = sum(ch.num_leaves for ch in children_defs)

    def unflatten(self, leaves):
        r"""Given a list of values and a ``TreeDef``, builds a pytree.
        This is the inverse operation of ``tree_flatten``.
        """
        assert len(leaves) == self.num_leaves
        start = 0
        children = []
        for ch in self.children_defs:
            children.append(ch.unflatten(leaves[start : start + ch.num_leaves]))
            start += ch.num_leaves
        return SUPPORTED_TYPE[self.type].unflatten(children, self.aux_data)

    def __hash__(self):
        return hash(
            tuple(
                [
                    self.type,
                    self.aux_data,
                    self.num_leaves,
                    tuple([hash(x) for x in self.children_defs]),
                ]
            )
        )

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __eq__(self, other) -> bool:
        return (
            self.type == other.type
            and self.aux_data == other.aux_data
            and self.num_leaves == other.num_leaves
            and self.children_defs == other.children_defs
        )

    def __repr__(self):
        return "{}[{}]".format(self.type.__name__, self.children_defs)


class LeafDef(TreeDef):
    def __init__(self, type):
        if not isinstance(type, collections.abc.Sequence):
            type = (type,)
        super().__init__(type, None, [])
        self.num_leaves = 1
        self.const_val = None

    def unflatten(self, leaves):
        assert len(leaves) == 1
        assert isinstance(leaves[0], self.type), self.type
        return leaves[0]

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __eq__(self, other):
        if isinstance(self.const_val, np.ndarray):
            return self.type == other.type and (self.const_val == other.const_val).all()
        return self.type == other.type and self.const_val == other.const_val

    def __hash__(self):
        if isinstance(self.const_val, np.ndarray):
            return hash(tuple([self.type, str(self.const_val)]))
        return hash(tuple([self.type, self.const_val]))

    def __repr__(self):
        return "Leaf({}[{}])".format(
            ", ".join(t.__name__ for t in self.type), self.const_val
        )
