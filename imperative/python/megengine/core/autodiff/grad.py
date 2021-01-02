# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import functools
import heapq
import itertools
import typing
import weakref

import numpy as np

import megengine as mge

from .._imperative_rt import core2, ops
from ..ops.builtin import Elemwise, OpDef, RemoteSend
from ..ops.special import Const

""" Some notes:
    1. Initialize the optimizer:
        for each trainable parameter:
            call wrt(param, callback)
        Each parameter tensor will be assciated with a Tracer object saved in Tensor._extra_data
    2. Tracer has one member: node, which is a VariableNode
    3. VariableNode has a OpNode member: opnode
    4. OpNode has four members:
        a. id
        b. inputs, which is made of VariableNode
        c. outputs, which are weakref's to VariableNode
        d. backward: call back function
        e. has_grad_fn: call has_grad_fn(opnode, reached) to check grad exist
        f. backward_allow_noinput: whether backward allow noinput

"""

_grad_count = 0
_grad_manager_dict = weakref.WeakValueDictionary()


def get_grad_managers():
    return [_grad_manager_dict[key] for key in _grad_manager_dict]


class GradKey(core2.GradKey):
    def __init__(self, name=None):
        if name:
            self.name = name

    def backward(self, ys, dys):
        return core2.backward(self, ys, dys)


class Grad:
    def __init__(self, name=None):
        global _grad_count
        if name is None:
            name = "grad_%d" % _grad_count
            _grad_count += 1
        self._refkeeper = []
        self._impl = GradKey(name)
        _grad_manager_dict[self._name] = self

    @property
    def _name(self):
        return self._impl.name

    def _is_attached_to(self, tensor):
        return self._impl.is_attached_to(tensor)

    def wrt(self, *tensors, callback=None):
        for x in tensors:
            self._impl.attach(x, callback)
        return self

    def __call__(self, ys, dys):
        from collections.abc import Sequence

        if not isinstance(ys, Sequence):
            ys = [ys]
        if not isinstance(dys, Sequence):
            dys = [dys]

        self._impl.backward(ys, dys)

        self._refkeeper = None

    def __enter__(self):
        return self

    def __exit__(self, _1, _2, _3):
        self._refkeeper = None
        del self._impl


class Function(ops.PyOpBase):
    def _default_rule(self, *args):
        ret = self.forward(*args)
        self.__single_output = isinstance(ret, core2.Tensor)
        return ret

    def _grad_rule(self, *args):
        return self._default_rule(*args), self.backward

    def __call__(self, *args):
        ret = core2.apply(self, *args)
        if self.__single_output:
            (ret,) = ret
        return ret

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)
