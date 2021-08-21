# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import copy
from collections.abc import MutableMapping, MutableSequence
from typing import Dict, Iterable, List, Optional, Sequence

from ...module import Module


def replace_container_with_module_container(container):
    has_module = False
    module_container = None
    if isinstance(container, Dict):
        m_dic = copy.copy(container)
        for key, value in container.items():
            if isinstance(value, Module):
                has_module = True
            elif isinstance(value, (List, Dict)):
                (
                    _has_module,
                    _module_container,
                ) = replace_container_with_module_container(value)
                m_dic[key] = _module_container
                if _has_module:
                    has_module = True
        if not all(isinstance(v, Module) for v in m_dic.values()):
            return has_module, None
        else:
            return has_module, _ModuleDict(m_dic)
    elif isinstance(container, List):
        m_list = copy.copy(container)
        for ind, value in enumerate(container):
            if isinstance(value, Module):
                has_module = True
            elif isinstance(value, (List, Dict)):
                (
                    _has_module,
                    _module_container,
                ) = replace_container_with_module_container(value)
                m_list[ind] = _module_container
                if _has_module:
                    has_module = True
        if not all(isinstance(v, Module) for v in m_list):
            return has_module, None
        else:
            return has_module, _ModuleList(m_list)
    return has_module, module_container


class _ModuleList(Module, MutableSequence):
    r"""
    A List-like container.

    Using a ``ModuleList``, one can visit, add, delete and modify submodules
    just like an ordinary python list.
    """

    def __init__(self, modules: Optional[Iterable[Module]] = None):
        super().__init__()
        self._size = 0
        if modules is None:
            return
        for mod in modules:
            self.append(mod)

    @classmethod
    def _ikey(cls, idx):
        return "{}".format(idx)

    def _check_idx(self, idx):
        L = len(self)
        if idx < 0:
            idx = L + idx
        if idx < 0 or idx >= L:
            raise IndexError("list index out of range")
        return idx

    def __getitem__(self, idx: int):
        if isinstance(idx, slice):
            idx = range(self._size)[idx]
        if not isinstance(idx, Sequence):
            idx = [
                idx,
            ]
        rst = []
        for i in idx:
            i = self._check_idx(i)
            key = self._ikey(i)
            try:
                rst.append(getattr(self, key))
            except AttributeError:
                raise IndexError("list index out of range")
        return rst if len(rst) > 1 else rst[0]

    def __setitem__(self, idx: int, mod: Module):
        if not isinstance(mod, Module):
            raise ValueError("invalid sub-module")
        idx = self._check_idx(idx)
        setattr(self, self._ikey(idx), mod)

    def __delitem__(self, idx):
        idx = self._check_idx(idx)
        L = len(self)
        for orig_idx in range(idx + 1, L):
            new_idx = orig_idx - 1
            self[new_idx] = self[orig_idx]
        delattr(self, self._ikey(L - 1))
        self._size -= 1

    def __len__(self):
        return self._size

    def insert(self, idx, mod: Module):
        assert isinstance(mod, Module)
        L = len(self)
        if idx < 0:
            idx = L - idx
        # clip idx to (0, L)
        if idx > L:
            idx = L
        elif idx < 0:
            idx = 0

        for new_idx in range(L, idx, -1):
            orig_idx = new_idx - 1
            key = self._ikey(new_idx)
            setattr(self, key, self[orig_idx])

        key = self._ikey(idx)
        setattr(self, key, mod)
        self._size += 1

    def forward(self):
        raise RuntimeError("ModuleList is not callable")


class _ModuleDict(Module, MutableMapping):
    r"""
    A Dict-like container.

    Using a ``ModuleDict``, one can visit, add, delete and modify submodules
    just like an ordinary python dict.
    """

    def __init__(self, modules: Optional[Dict[str, Module]] = None):
        super().__init__()
        self._size = 0
        if modules is not None:
            self.update(modules)

    def __delitem__(self, key):
        delattr(self, key)
        self._size -= 1

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        if not isinstance(value, Module):
            raise ValueError("invalid sub-module")
        setattr(self, key, value)
        self._size += 1

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return self._size

    def items(self):
        return dict(self.named_children()).items()

    def values(self):
        return dict(self.named_children()).values()

    def keys(self):
        return dict(self.named_children()).keys()

    def forward(self):
        raise RuntimeError("ModuleList is not callable")
