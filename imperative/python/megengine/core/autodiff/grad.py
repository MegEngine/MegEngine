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
from ..tensor.tensor import Tensor, get_context
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


class Grad:
    def __init__(self, name=None):

        if name is None:
            global _grad_count
            self._name = "grad_" + str(_grad_count)
            _grad_count += 1
        else:
            self._name = name
        assert self._name not in _grad_manager_dict, "grad manager name duplicated"
        _grad_manager_dict[self._name] = self

        # list of all x in partial(y) / partial(x)
        self.xs = []

        # constains weak reference of all OpNode during forward
        # OpNode contains inputs, outputs and its backward
        # ops forms the computational graph
        self.ops = []

        # save remote_send output for backward
        self.remote_send_cache = []

        self._attached_tensors = weakref.WeakSet()
        self._enabled = True

    @property
    def name(self):
        return self._name

    def wrt(self, *args: Tensor, callback=None):
        """ Indicates the loss is a function of the input tensors (usually the net trainable parameters),
        i.e., d (loss) / d (Tensor) != 0

        callback is used to perform additional operations after gradient is obtained in backward.
        e.g., copy the grad to a particular place

        A VariableNode will be created and saved in the tensor/s _extra_data slot.
        """

        for x in map(get_tensor, args):
            v = self._new_variable(x, callback=callback)
            assert self not in x._extra_data
            x._extra_data[self] = Tracer(v)
            self.xs.append(v)

        return self

    def _new_variable(self, owner, opnode=None, callback=None):
        self._attached_tensors.add(owner)
        return VariableNode(self, owner, opnode=opnode, callback=callback)

    def _new_opnode(self, inputs, outputs):
        inputs = tuple(inputs)
        for i in inputs:
            assert i is None or isinstance(i, VariableNode)
        o = OpNode()
        o.inputs = inputs
        o.outputs = []
        tracers = []
        for i in outputs:
            assert isinstance(i, Tensor)
            v = self._new_variable(i, o)
            o.outputs.append(weakref.ref(v))
            tracers.append(Tracer(v))
        self.ops.append(weakref.ref(o))
        return o, tracers

    def copy(self):
        raise NotImplementedError

    def __enter__(self):
        return self

    def _exit(self):
        """clear all resources"""
        self._enabled = False
        for o in self.ops:
            o = o()
            if o:
                o.clear()
        for i in self._attached_tensors:
            i._extra_data.pop(self, None)
        self.remote_send_cache = []

    def __exit__(self, *_):
        self._exit()

    def __call__(self, ys, dys):
        """ Defines Grad().

        :param ys: outputs of forward operators, e.g., the loss tensor
        :type ys: list of Tensor or TensorWrapperBase
        :param dys: delta of outputs, physically equivalent to sensitivity of outputs to the loss,
            e.g., one for the loss itself
        :type dys: list of Tensor or TensorWrapperBase
        """
        assert self._enabled
        self._enabled = False

        def check_wrapper():
            if isinstance(dys, TensorWrapperBase):
                return type(dys)
            if isinstance(dys, TensorBase):
                return
            assert isinstance(dys, (tuple, list))
            for i in dys:
                if isinstance(i, TensorWrapperBase):
                    return type(i)
            # use Tensor as defualt wrapper
            return mge.Tensor

        Wrapper = check_wrapper()

        def aslist(x):
            if isinstance(x, (Tensor, TensorWrapperBase)):
                x = [x]
            else:
                x = list(x)
            x = [i.__wrapped__ if isinstance(i, TensorWrapperBase) else i for i in x]
            for i in x:
                assert isinstance(i, Tensor)
            return x

        ys = aslist(ys)
        dys = aslist(dys)
        assert len(ys) == len(dys)

        ids = [i for i, y in enumerate(ys) if self in y._extra_data.keys()]

        ys = [y for i, y in enumerate(ys) if i in ids]
        dys = [dy for i, dy in enumerate(dys) if i in ids]

        # ys is changed to a list of VariableNode which contains more information
        # such as OpNode, callback, etc.
        ys = [i._extra_data[self].node for i in ys]

        # NOTE: callback is called only if grad is not None

        # the OpNode sequence in backward
        op_seq = []

        # VariableNode -> (i, j), where i is time stamp in backward, j means jth input
        last_written_to = {}

        def schedule():
            reached = set(ys)
            # i is the time stamp in backward
            i = 0
            for o in self.ops[::-1]:
                o = o()
                if o is None:
                    continue

                if not o.has_grad_fn(o, reached):
                    continue
                op_seq.append(o)
                for j, v in enumerate(o.inputs):
                    reached.add(v)
                    last_written_to[v] = i, j
                i += 1

        schedule()

        # VariableNode -> Tensor
        cache = {}

        def initialize():
            for y, dy in zip(ys, dys):
                cache[y] = dy
                if y not in last_written_to and y.callback:
                    y.callback(y.owner(), dy)

        initialize()

        # NOTE: None is used to mark a node has been consumed

        for seqno, opnode in enumerate(op_seq):
            input_nodes = opnode.inputs
            output_nodes = [i() for i in opnode.outputs]
            backward = opnode.backward
            backward_allow_noinput = opnode.backward_allow_noinput
            opnode.clear()

            output_grads = []
            for i in output_nodes:
                if i is not None:
                    if i in cache:
                        assert cache[i] is not None
                        output_grads.append(cache[i])
                    else:
                        output_grads.append(None)
                    # read by backward, mark consumed
                    cache[i] = None
                else:
                    output_grads.append(None)
            if (
                any([grad is not None for grad in output_grads])
                or backward_allow_noinput
            ):
                input_grads = backward(*output_grads)
            else:
                input_grads = [None] * len(input_nodes)

            assert len(input_nodes) == len(input_grads)
            for i, (v, g) in enumerate(zip(input_nodes, input_grads)):
                if v is None:
                    continue
                if v in cache:
                    assert cache[v]
                    if g is not None:
                        cache[v] = add(cache[v], g)
                elif g is not None:
                    cache[v] = g
                if last_written_to[v] == (seqno, i):
                    if v.callback:
                        v.callback(
                            v.owner(), Wrapper(cache[v]) if Wrapper else cache[v]
                        )
                    if v.opnode is None:
                        # won't read by backward, mark consumed
                        cache[v] = None

        for v in cache.values():
            assert v is None

        self._exit()

    def __del__(self):
        self._exit()


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
