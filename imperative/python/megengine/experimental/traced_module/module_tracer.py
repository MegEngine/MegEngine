# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from ...module import Module

_active_module_tracer = None


def active_module_tracer():
    return _active_module_tracer


def set_active_module_tracer(tracer):
    global _active_module_tracer
    _active_module_tracer = tracer


class module_tracer:

    _opaque_types = set()

    _active_scopes = None

    def __init__(self):
        self._active_scopes = []

    @classmethod
    def register_as_builtin(cls, mod):
        assert issubclass(mod, Module)
        cls._opaque_types.add(mod)
        return mod

    @classmethod
    def is_builtin(cls, mod):
        return type(mod) in cls._opaque_types

    def push_scope(self, scope):
        self._active_scopes.append(scope)

    def pop_scope(self):
        self._active_scopes.pop()

    def current_scope(self):
        if self._active_scopes:
            return self._active_scopes[-1]
        return None
