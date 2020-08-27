# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import collections
import threading
import weakref
from concurrent.futures import Future, ThreadPoolExecutor

from .. import _imperative_rt
from .._wrap import device as as_device
from ..ops.builtin import OpDef
from .core import OpBase, TensorBase, apply


class CompiledFunction:
    def __init__(self, graph, function):
        self._graph = graph
        self._function = function
        self._future = None

    def execute(self, *args):
        assert self._future is None
        self._future = self._graph._executor.submit(self._function.execute, *args)

    def wait(self):
        assert self._future is not None
        self._future.exception()
        self._function.wait()
        try:
            return self._future.result()
        finally:
            self._future = None

    def __call__(self, *args):
        self.execute(*args)
        return self.wait()


class Graph(_imperative_rt.ComputingGraph):
    def __init__(self):
        super().__init__()
        self._var_cache = weakref.WeakKeyDictionary()
        self._op_cache = weakref.WeakKeyDictionary()
        self._executor = ThreadPoolExecutor(1)

    def _wrap(self, obj):
        if type(obj) is _imperative_rt.VarNode:
            wrapper, cache = VarNode, self._var_cache
        elif type(obj) is _imperative_rt.OperatorNode:
            wrapper, cache = OpNode, self._op_cache
        if obj not in cache:
            cache[obj] = wrapper(obj)
        return cache[obj]

    def compile(self, *args):
        return CompiledFunction(self, super().compile(_unwrap(args)))


class VarNode(TensorBase):
    def __init__(self, node: _imperative_rt.VarNode):
        self._node = node

    @property
    def graph(self) -> Graph:
        return self._node.graph

    @property
    def op(self):
        return self.graph._wrap(self._node.owner)

    @property
    def dtype(self):
        return self._node.dtype

    @property
    def device(self):
        return as_device(self._node.comp_node)


class OpNode:
    def __init__(self, node: _imperative_rt.OperatorNode):
        self._node = node

    @property
    def graph(self) -> Graph:
        return self._node.graph

    @property
    def inputs(self):
        return tuple(map(self.graph._wrap, self._node.inputs))

    @property
    def outputs(self):
        return tuple(map(self.graph._wrap, self._node.outputs))


def _wrap(x):
    if isinstance(x, collections.Sequence):
        return type(x)(map(_wrap, x))
    return x.graph._wrap(x)


def _unwrap(x):
    if isinstance(x, collections.Sequence):
        return type(x)(map(_unwrap, x))
    return x._node


@apply.add
def _(op: OpDef, *args: VarNode):
    outputs = _imperative_rt.invoke_op(op, _unwrap(args))
    return _wrap(outputs)


def input_callback(callback, *args, device=None, dtype=None, graph=None):
    outputs = _imperative_rt.input_callback(
        callback, as_device(device).to_c(), dtype, _unwrap(args), graph=graph
    )
    value, dummy = _wrap(outputs)
    return value, dummy


class InputNode(OpNode):
    def __init__(self, *args: VarNode, device=None, dtype=None, graph=None):
        r = _imperative_rt.DeviceTensorNDRendezvous()
        if device is not None:
            device = as_device(device).to_c()
        outputs = _imperative_rt.input_callback(
            r, device, dtype, _unwrap(args), graph=graph
        )
        super().__init__(outputs[0].owner)
        self._rendezvous = r

    def set_value(self, value):
        assert isinstance(value, _imperative_rt.DeviceTensorND)
        self._rendezvous.set(value)

    def reset(self):
        self._rendezvous.reset()

    @property
    def device(self):
        return self.outputs[0].device

    @property
    def dtype(self):
        return self.outputs[0].dtype


def output_callback(callback, var, *args):
    args = (var,) + args
    dummy = _imperative_rt.output_callback(callback, _unwrap(args))
    return _wrap(dummy)


class OutputNode(OpNode):
    def __init__(self, var, *args):
        args = (var,) + args
        r = _imperative_rt.DeviceTensorNDRendezvous()
        dummy = _imperative_rt.output_callback(r, _unwrap(args))
        super().__init__(dummy.owner)
        self._rendezvous = r

    def get_value(self):
        return self._rendezvous.get()

    def reset(self):
        self._rendezvous.reset()


class TensorAttr:
    def __init__(self, shape, dtype, device):
        self.shape = shape
        self.dtype = dtype
        self.device = device


class AttrOutputNode(OpNode):
    def __init__(self, var, *args):
        args = (var,) + args
        r = _imperative_rt.TensorAttrRendezvous()
        dummy = _imperative_rt.attr_output_callback(r, _unwrap(args))
        super().__init__(dummy.owner)
        self._rendezvous = r

    def get_value(self):
        attr = self._rendezvous.get()
        return TensorAttr(attr.shape, attr.dtype, as_device(attr.comp_node))

    def reset(self):
        self._rendezvous.reset()
