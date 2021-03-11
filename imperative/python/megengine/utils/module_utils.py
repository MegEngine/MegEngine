# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from collections import Iterable

from ..module import Sequential
from ..module.module import Module, _access_structure
from ..tensor import Tensor


def get_expand_structure(obj: Module, key: str):
    """
    Gets Module's attribute compatible with complex key from Module's :meth:`~.named_children`.
    Supports handling structure containing list or dict.
    """

    def f(_, __, cur):
        return cur

    return _access_structure(obj, key, callback=f)


def set_expand_structure(obj: Module, key: str, value):
    """
    Sets Module's attribute compatible with complex key from Module's :meth:`~.named_children`.
    Supports handling structure containing list or dict.
    """

    def f(parent, key, cur):
        if isinstance(parent, (Tensor, Module)):
            # cannnot use setattr to be compatible with Sequential's ``__setitem__``
            if isinstance(cur, Sequential):
                parent[int(key)] = value
            else:
                setattr(parent, key, value)
        else:
            parent[key] = value

    _access_structure(obj, key, callback=f)
