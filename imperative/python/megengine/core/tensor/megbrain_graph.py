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

import numpy as np

from .. import _imperative_rt
from .._imperative_rt import GraphOptimizeOptions
from .._imperative_rt.ops import BackwardGraph
from .._wrap import device as as_device
from ..ops.builtin import OpDef
from .core import OpBase, TensorBase, apply


class Graph(_imperative_rt.ComputingGraph):
    def __init__(self):
        super().__init__()
        self._var_cache = weakref.WeakKeyDictionary()
        self._op_cache = weakref.WeakKeyDictionary()
        self._executor = ThreadPoolExecutor(1)
        self._function = None
        self._future = None

    def _wrap(self, obj):
        if type(obj) is _imperative_rt.VarNode:
            wrapper, cache = VarNode, self._var_cache
        elif type(obj) is _imperative_rt.OperatorNode:
            wrapper, cache = OpNode, self._op_cache
        else:
            raise TypeError(type(obj))
        if obj not in cache:
            cache[obj] = wrapper(obj)
        return cache[obj]

    def compile(self, *args):
        self._function = super().compile(_unwrap(args))
        return self

    def execute(self, *args):
        assert self._future is None
        self._future = self._executor.submit(self._function.execute, *args)

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

    def make_const(self, data, dtype=None, device=None):
        if isinstance(data, _imperative_rt.DeviceTensorND):
            assert dtype is None and device is None
            return self._wrap(_imperative_rt.make_shared(self, data))
        else:
            data = np.asarray(data, dtype=dtype)
            if data.dtype == np.float64:
                data = data.astype(np.float32)
            elif data.dtype == np.int64:
                data = data.astype(np.int32)
            device = as_device(device).to_c()
            return self._wrap(_imperative_rt.make_const(self, data, device, dtype))

    def make_input(self, *args: "VarNode", device=None, dtype=None, shape=None):
        opnode = InputNode(*args, device=device, dtype=dtype, shape=shape, graph=self)
        return opnode.outputs[0]

    def make_h2d(self, *, dtype, device, shape=None, name=None):
        device = as_device(device).to_c()
        return self._wrap(_imperative_rt.make_h2d(self, device, dtype, shape, name))


def optimize_for_inference(dest_vars, **kwargs):
    r"""Applies optimize_for_inference pass for computing graph.

        :param dest_vars: list of output vars in the computing graph

        :Keyword Arguments:

            * enable_io16xc32 --
                whether to use float16 for I/O between oprs and use
                float32 as internal computation precision. Note the output var would be
                changed to float16.
            * enable_ioc16 --
                whether to use float16 for both I/O and computation
                precision.

            * enable_hwcd4 --
                whether to use NHWCD4 data layout. This is faster on some
                OpenCL backend.
            * enable_nchw88 --
                whether to use NCHW88 data layout, currently
                used in X86 AVX backend.
            * enable_nchw44 --
                whether to use NCHW44 data layout, currently
                used in arm backend.
            * enable_nchw44_dot --
                whether to use NCHW44_dot data layout, currently
                used in armv8.2+dotprod backend.
            * enable_nchw4 --
                whether to use NCHW4 data layout, currently
                used in nvidia backend(based on cudnn).
            * enable_nchw32 --
                whether to use NCHW32 data layout, currently
                used in nvidia backend with tensorcore(based on cudnn).
            * enable_chwn4 --
                whether to use CHWN4 data layout, currently
                used in nvidia backend with tensorcore.

            * enable_fuse_conv_bias_nonlinearity: whether to fuse conv+bias+nonlinearty
                into one opr.
            * enable_fuse_conv_bias_with_z: whether to fuse conv_bias with z
                input for inference on nvidia backend(this optimization pass will
                result in mismatch of the precision of output of training and
                inference)
    """
    inference_options = GraphOptimizeOptions()
    if optimize_for_inference:
        inference_optimize_layout_transform_map = {
            "enable_hwcd4": GraphOptimizeOptions.LayoutTransform.NHWCD4,
            "enable_nchw4": GraphOptimizeOptions.LayoutTransform.NCHW4,
            "enable_nchw88": GraphOptimizeOptions.LayoutTransform.NCHW88,
            "enable_nchw32": GraphOptimizeOptions.LayoutTransform.NCHW32,
            "enable_nchw44": GraphOptimizeOptions.LayoutTransform.NCHW44,
            "enable_nchw44_dot": GraphOptimizeOptions.LayoutTransform.NCHW44_DOT,
            "enable_chwn4": GraphOptimizeOptions.LayoutTransform.CHWN4,
        }

        for k, v in inference_optimize_layout_transform_map.items():
            if kwargs.pop(k, False):
                inference_options.layout_transform = v

        if kwargs.pop("enable_io16xc32", False):
            inference_options.f16_io_f32_comp = True
        if kwargs.pop("enable_ioc16", False):
            inference_options.f16_io_comp = True
        if kwargs.pop("enable_fuse_conv_bias_nonlinearity", False):
            inference_options.fuse_conv_bias_nonlinearity = True
        if kwargs.pop("enable_fuse_conv_bias_with_z", False):
            inference_options.fuse_conv_bias_with_z = True

        if kwargs:
            raise ValueError("unknown options: %s" % list(kwargs))

    res_vars = _imperative_rt.optimize_for_inference(
        [i._node for i in dest_vars], inference_options
    )
    return [VarNode(i) for i in res_vars]


def dump(*args):
    return _imperative_rt.dump_graph([i._node for i in args])


class VarNode(TensorBase):
    def __init__(self, node: _imperative_rt.VarNode):
        self._node = node
        self.graph._var_cache[node] = self

    @property
    def graph(self) -> Graph:
        return self._node.graph

    @property
    def op(self):
        return self.graph._wrap(self._node.owner)

    @property
    def name(self):
        return self._node.name

    @name.setter
    def name(self, name):
        self._node.name = name

    @property
    def dtype(self):
        return self._node.dtype

    @property
    def device(self):
        return as_device(self._node.comp_node)

    @property
    def shape(self):
        return self._node.shape

    @property
    def value(self):
        return self._node.value


class OpNode:
    def __init__(self, node: _imperative_rt.OperatorNode):
        self._node = node
        self.graph._op_cache[node] = self

    @property
    def graph(self) -> Graph:
        return self._node.graph

    @property
    def name(self):
        return self._node.name

    @name.setter
    def name(self, name):
        self._node.name = name

    @property
    def inputs(self):
        return tuple(map(self.graph._wrap, self._node.inputs))

    @property
    def outputs(self):
        return tuple(map(self.graph._wrap, self._node.outputs))


def _wrap(x):
    if isinstance(x, collections.abc.Sequence):
        return type(x)(map(_wrap, x))
    return x.graph._wrap(x)


def _unwrap(x):
    if isinstance(x, collections.abc.Sequence):
        return type(x)(map(_unwrap, x))
    return x._node


@apply.register()
def _(op: OpDef, *args: VarNode):
    outputs = _imperative_rt.invoke_op(op, _unwrap(args))
    return _wrap(outputs)


@apply.register()
def _(op: BackwardGraph, *args: VarNode):
    assert args
    graph = args[0].graph
    return op.interpret(lambda op, args: apply(op, *args), graph.make_const, args)


def input_callback(callback, *args, device=None, dtype=None, shape=None, graph=None):
    outputs = _imperative_rt.input_callback(
        callback, as_device(device).to_c(), dtype, shape, _unwrap(args), graph=graph
    )
    value, dummy = _wrap(outputs)
    return value, dummy


class InputNode(OpNode):
    def __init__(self, *args: VarNode, device=None, dtype=None, shape=None, graph=None):
        r = _imperative_rt.DeviceTensorNDRendezvous()
        if device is not None:
            device = as_device(device).to_c()
        outputs = _imperative_rt.input_callback(
            r, device, dtype, shape, _unwrap(args), graph=graph
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

    def drop_value(self):
        self._rendezvous.drop()

    def reset(self):
        self._rendezvous.reset()


class ValueOutputNode(OpNode):
    def __init__(self, var, *args):
        args = (var,) + args
        r = _imperative_rt.HostTensorNDRendezvous()
        dummy = _imperative_rt.value_output_callback(r, _unwrap(args))
        super().__init__(dummy.owner)
        self._rendezvous = r

    def get_value(self):
        hostnd, event = self._rendezvous.get()
        event.wait()
        return hostnd.numpy()

    def drop_value(self):
        self._rendezvous.drop()

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

    def drop_value(self):
        self._rendezvous.drop()

    def reset(self):
        self._rendezvous.reset()
