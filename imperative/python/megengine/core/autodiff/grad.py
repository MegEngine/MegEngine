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

from .._imperative_rt import core2
from ..ops.builtin import Elemwise, OpDef, RemoteSend
from ..ops.special import Const
from ..tensor.core import TensorBase, TensorWrapperBase, apply
from ..tensor.function import Function
from . import builtin_op_utils

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


def add(a, b):
    (c,) = apply(Elemwise(Elemwise.Mode.ADD), a, b)
    return c


def get_tensor(x):
    # use recursion to avoid infinite loop
    if isinstance(x, Tensor):
        return x
    try:
        x = x.__wrapped__
    except AttributeError:
        raise TypeError(type(x))
    return get_tensor(x)


class clearable:
    __cleared = False

    def __bool__(self):
        return not self.__cleared

    def clear(self):
        self.__dict__.clear()
        self.__cleared = True


class OpNode(clearable):
    """ OpNode saves all the information to form the computational graph.
    """

    def __init__(self):
        self.id = None
        self.inputs = None  # Could be VariableNode
        self.outputs = None  # Could be VariableNode
        self.backward = None
        self.has_grad_fn = None
        self.backward_allow_noinput = False


class VariableNode(clearable):
    """ VariableNode saves OpNode and callback.
    FIXME!!! Explain manager and owner
    """

    def __init__(self, manager, owner, opnode=None, callback=None):
        # manager is Grad type
        self.manager = weakref.ref(manager)
        # owner is Tensor type
        self.owner = weakref.ref(owner)
        self.opnode = opnode
        self.callback = callback


class Tracer(clearable, TensorBase):
    def __init__(self, node=None):
        """ type(node) is VariableNode
        """
        self.node = node


@functools.singledispatch
def check_backward_allow_noinput(op: OpDef):
    return False


@functools.singledispatch
def get_op_has_grad_fn(op: OpDef):
    assert 0


@get_op_has_grad_fn.register(OpDef)
def _(op: OpDef):
    return default_has_grad_fn


@get_op_has_grad_fn.register(Function)
def _(op: Function):
    return default_has_grad_fn


def default_has_grad_fn(opnode, reached):
    for v in opnode.outputs:
        if v() in reached:
            return True
    return False


@apply.register()
def tracer_apply(op: (OpDef, Function), *args: typing.Optional[Tracer]):
    args = tuple(i if isinstance(i, Tracer) else None for i in args)
    input_requires_grad = list(map(bool, args))
    if not any(input_requires_grad):
        return

    ctx = get_context()
    manager = None
    assert len(ctx.inputs) == len(args)
    for i, j in zip(ctx.inputs, args):
        if j:
            j = j.node
            assert i is j.owner()
            if manager is None:
                manager = j.manager()
                assert manager
            else:
                assert manager is j.manager()

    if not manager._enabled:
        return

    # register backward method
    # tuple of backward functions corresponding to dy / dx_i
    # None means y is not a function of x_i
    backward, output_need_grad = builtin_op_utils.builtin_op_get_backward_fn(
        op, ctx.inputs, ctx.outputs, input_requires_grad
    )
    assert len(ctx.outputs) == len(output_need_grad)
    if not any(output_need_grad):
        return

    opnode, outputs = manager._new_opnode([i and i.node for i in args], ctx.outputs)
    if isinstance(op, RemoteSend):
        manager.remote_send_cache.append(opnode)
    opnode.backward = backward

    outputs = [x if y else None for (x, y) in zip(outputs, output_need_grad)]

    opnode.backward_allow_noinput = check_backward_allow_noinput(op)

    opnode.has_grad_fn = get_op_has_grad_fn(op)

    return tuple(outputs)


@apply.register()
def _(op: Const, *_: typing.Optional[Tracer]):
    return None


class Grad:
    def __init__(self):
        self._impl = core2.GradKey()

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
        core2.backward(self._impl, ys, dys)

    def __enter__(self):
        return self

    def __exit__(self, _1, _2, _3):
        del self._impl
