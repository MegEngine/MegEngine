import collections
from collections import OrderedDict, defaultdict
from functools import partial
from inspect import FullArgSpec
from typing import Any, Callable, Dict, List, NamedTuple, Tuple

import numpy as np

from ..core._imperative_rt import OpDef
from ..core._imperative_rt.common import CompNode
from ..core._imperative_rt.core2 import Tensor as RawTensor
from ..core._wrap import Device
from ..core.tensor.dtype import QuantDtypeMeta
from ..distributed import Group
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
    bytes,
    bytearray,
    QuantDtypeMeta,
    CompNode,
    Device,
    type(None),
    type(Ellipsis),
    QuantMode,
    ArgsIndex,
    Group,
    FullArgSpec,
}

USER_REGISTERED_LEAF_TYPE = []
USER_REGISTERED_CONTAINER_TYPE = []
# if isinstance(object, SUPPORTED_LEAF_CLS) or issubclass(obj, SUPPORTED_LEAF_CLS) is True, the object could be threated as leaf node of pytree
SUPPORTED_LEAF_CLS = [
    Module,
    Node,
    NodeMixin,
    np.dtype,
    np.ndarray,
    np.number,
    np.bool_,
    OpDef,
]

NodeType = NamedTuple("NodeType", [("flatten", Callable), ("unflatten", Callable)])


def register_supported_type(
    type,
    flatten_fn: Callable[[Any], Tuple[List, Any]] = None,
    unflatten_fn: Callable[[List, Any], Any] = None,
):
    r"""Call this function to register the ``type`` as a built-in type. The registered ``type`` 
    can be used and serialized correctly in :py:class:`TracedModule`.

    Examples:
        .. code-block::

            def dict_flatten(obj: Dict):
                context, values = [], []
                # obj.keys() needs to be sortable
                keys = sorted(obj.keys())
                for key in keys:
                    values.append(obj[key])
                    context.append(key)
                return values, tuple(context)
            
            def dict_unflatten(values: List, context: Any):
                return dict(zip(context, values))
            
            register_supported_type(dict, dict_flatten, dict_unflatten)

    Args:
        type: the type that needs to be registered.
        flatten_fn: a function that should take an object created from ``type`` and return a
            flat list of values. It can also return some context that is used in reconstructing
            the object. Default: None
        unflatten_fn: a function that should take a flat list of values and some context
            (returned by flatten_fn). It returns the object by reconstructing
            it from the list and the context. Default: None
    """
    tp_info = (type.__module__, type.__qualname__)
    if flatten_fn and unflatten_fn:
        USER_REGISTERED_CONTAINER_TYPE.append(tp_info)
    else:
        USER_REGISTERED_LEAF_TYPE.append(tp_info)
    _register_supported_type(type, flatten_fn, unflatten_fn)


def _register_supported_type(type, flatten_fn=None, unflatten_fn=None):
    if flatten_fn and unflatten_fn:
        SUPPORTED_TYPE[type] = NodeType(flatten_fn, unflatten_fn)
    else:
        SUPPORTED_LEAF_CLS.append(type)


def _dict_flatten(ordered, inp):
    aux_data = []
    results = []
    dict_items = inp.items() if ordered else sorted(inp.items())
    for key, value in dict_items:
        results.append(value)
        aux_data.append(key)
    return results, tuple(aux_data)


def _dict_unflatten(dict_type, inps, aux_data):
    return dict_type(zip(aux_data, inps))


def qparams_flatten(inp):
    aux_data = []
    results = []
    for key in inp.__slots__:
        aux_data.append(key)
        results.append(getattr(inp, key, None))
    return results, tuple(aux_data)


def qparams_unflatten(qparam_type, inp, aux_data):
    obj = qparam_type.__new__(qparam_type)
    for k, v in zip(aux_data, inp):
        setattr(obj, k, v)
    return obj


_register_supported_type(list, lambda x: (x, None), lambda x, aux_data: list(x))
_register_supported_type(tuple, lambda x: (x, None), lambda x, aux_data: tuple(x))
_register_supported_type(
    dict, partial(_dict_flatten, False), partial(_dict_unflatten, dict)
)
_register_supported_type(
    defaultdict, partial(_dict_flatten, False), partial(_dict_unflatten, defaultdict)
)
_register_supported_type(
    OrderedDict, partial(_dict_flatten, True), partial(_dict_unflatten, OrderedDict)
)

_register_supported_type(
    slice,
    lambda x: ([x.start, x.stop, x.step], None),
    lambda x, aux_data: slice(x[0], x[1], x[2]),
)

_register_supported_type(QParams, qparams_flatten, partial(qparams_unflatten, QParams))
_register_supported_type(
    LSQParams, qparams_flatten, partial(qparams_unflatten, LSQParams)
)


def _is_leaf(obj):
    obj_type = obj if isinstance(obj, type) else type(obj)
    return (
        issubclass(obj_type, tuple(SUPPORTED_LEAF_CLS))
        or obj_type in SUPPORTED_LEAF_TYPE
    )


def _leaf_type(node):
    if isinstance(node, (RawTensor, TensorNode)):
        return (Tensor, TensorNode, ArgsIndex)
    elif isinstance(node, (NodeMixin, Module, ModuleNode)):
        return (Module, ModuleNode, NodeMixin, ArgsIndex)
    else:
        return (type(node), ArgsIndex)


def _is_const_leaf(node):
    if isinstance(node, (RawTensor, Node, NodeMixin, Module)):
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
        assert is_leaf(
            values
        ), 'doesn\'t support {} type, MUST use "register_supported_type" method to register self-defined type'.format(
            values
        )
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

    def _args_kwargs_repr(self):
        if (
            len(self.children_defs) == 2
            and issubclass(self.children_defs[0].type, (List, Tuple))
            and issubclass(self.children_defs[1].type, Dict)
        ):
            args_def = self.children_defs[0]
            content = ", ".join(repr(i) for i in args_def.children_defs)
            kwargs_def = self.children_defs[1]
            if kwargs_def.aux_data:
                content += ", "
                content += ", ".join(
                    str(i) + "=" + repr(j)
                    for i, j in zip(kwargs_def.aux_data, kwargs_def.children_defs)
                )
            return content
        else:
            return repr(self)

    def __repr__(self):
        format_str = self.type.__name__ + "({})"
        aux_data_delimiter = "="
        if issubclass(self.type, List):
            format_str = "[{}]"
        if issubclass(self.type, Tuple):
            format_str = "({})"
        if issubclass(self.type, Dict):
            format_str = "{{{}}}"
            aux_data_delimiter = ":"
        if self.aux_data:
            content = ", ".join(
                repr(i) + aux_data_delimiter + repr(j)
                for i, j in zip(self.aux_data, self.children_defs)
            )
        else:
            content = ", ".join(repr(i) for i in self.children_defs)
        return format_str.format(content)


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

        return "{}".format(
            self.const_val
            if self.const_val is not None or type(None) in self.type
            else self.type[0].__name__
        )
