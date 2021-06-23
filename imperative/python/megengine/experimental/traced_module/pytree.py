# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import collections
from typing import Callable, NamedTuple

SUPPORTED_TYPE = {}

NodeType = NamedTuple("NodeType", [("flatten", Callable), ("unflatten", Callable)])


def register_supported_type(type, flatten, unflatten):
    SUPPORTED_TYPE[type] = NodeType(flatten, unflatten)


def _dict_flatten(inp):
    aux_data = []
    results = []
    for key, value in sorted(inp.items()):
        results.append(value)
        aux_data.append(key)
    return results, aux_data


def _dict_unflatten(inps, aux_data):
    return dict(zip(aux_data, inps))


register_supported_type(list, lambda x: (x, None), lambda x, aux_data: list(x))
register_supported_type(tuple, lambda x: (x, None), lambda x, aux_data: list(x))
register_supported_type(dict, _dict_flatten, _dict_unflatten)
register_supported_type(
    slice,
    lambda x: ([x.start, x.stop, x.step], None),
    lambda x, aux_data: slice(x[0], x[1], x[2]),
)


def tree_flatten(
    values, leaf_type: Callable = lambda x: type(x), is_leaf: Callable = lambda x: True
):
    if type(values) not in SUPPORTED_TYPE:
        assert is_leaf(values)
        return [values,], LeafDef(leaf_type(values))
    rst = []
    children_defs = []
    children_values, aux_data = SUPPORTED_TYPE[type(values)].flatten(values)
    for v in children_values:
        v_list, treedef = tree_flatten(v, leaf_type)
        rst.extend(v_list)
        children_defs.append(treedef)

    return rst, TreeDef(type(values), aux_data, children_defs)


class TreeDef:
    def __init__(self, type, aux_data, children_defs):
        self.type = type
        self.aux_data = aux_data
        self.children_defs = children_defs
        self.num_leaves = sum(ch.num_leaves for ch in children_defs)

    def unflatten(self, leaves):
        assert len(leaves) == self.num_leaves
        start = 0
        children = []
        for ch in self.children_defs:
            children.append(ch.unflatten(leaves[start : start + ch.num_leaves]))
            start += ch.num_leaves
        return SUPPORTED_TYPE[self.type].unflatten(children, self.aux_data)

    def __eq__(self, other):
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

    def unflatten(self, leaves):
        assert len(leaves) == 1
        assert isinstance(leaves[0], self.type), self.type
        return leaves[0]

    def __repr__(self):
        return "Leaf({})".format(", ".join(t.__name__ for t in self.type))
