# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import collections
import json
import os
import threading
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Union

import numpy as np

from ...utils.comp_graph_tools import set_priority_to_id as _set_priority_to_id
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

    def set_priority_to_id(self, dest_vars):
        _set_priority_to_id(_unwrap(dest_vars))

    def compile(self, *args):
        self._function = super().compile(_unwrap(args))
        return self

    def execute(self, *args):
        assert self._future is None

        def wrapped(*args):
            try:
                self._function.execute(*args)
            except Exception as exc:
                for i in self._function._all_rendezvous:
                    i.set_exception(str(exc))
                raise exc

        self._future = self._executor.submit(wrapped, *args)

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

    def _make_const_for_backward(self, data):
        device = as_device(data.comp_node).to_c()
        data = data.numpy()
        return self._wrap(_imperative_rt.make_const(self, data, device, data.dtype))

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


class VarNode(TensorBase):
    def __init__(self, node: _imperative_rt.VarNode, isscalar=False):
        self._node = node
        self._isscalar = isscalar
        if hasattr(self.graph, "_var_cache"):
            self.graph._var_cache[node] = self

    @property
    def graph(self) -> Graph:
        return self._node.graph

    @property
    def op(self):
        if hasattr(self.graph, "_wrap"):
            return self.graph._wrap(self._node.owner)
        else:
            return self._node.owner

    @property
    def name(self):
        return self._node.name

    @property
    def id(self):
        return self._node.id

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
        if hasattr(self.graph, "_op_cache"):
            self.graph._op_cache[node] = self

    @property
    def graph(self) -> Graph:
        return self._node.graph

    @property
    def name(self):
        return self._node.name

    @property
    def id(self):
        return self._node.id

    @name.setter
    def name(self, name):
        self._node.name = name

    @property
    def inputs(self):
        if hasattr(self.graph, "_wrap"):
            return tuple(map(self.graph._wrap, self._node.inputs))
        else:
            return self._node.inputs

    @property
    def outputs(self):
        if hasattr(self.graph, "_wrap"):
            return tuple(map(self.graph._wrap, self._node.outputs))
        else:
            return self._node.outputs

    @property
    def params(self):
        return json.loads(self._node.params)

    @property
    def type(self):
        return self._node.type


def optimize_for_inference(dest_vars, **kwargs):
    r"""
    Applies optimize_for_inference pass for computing graph.

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


CompGraphDumpResult = collections.namedtuple(
    "CompGraphDumpResult",
    [
        "nr_opr",
        "tot_bytes",
        "tensor_value_bytes",
        "content_hash",
        "inputs",
        "outputs",
        "params",
    ],
)


def dump_graph(
    output_vars: Union[Dict[str, VarNode], List[VarNode]],
    *,
    keep_var_name: int = 1,
    keep_param_name: bool = False,
    keep_opr_priority: bool = False,
    strip_info_file=None,
    append_json=False
):
    """
    serialize the computing graph of `output_vars` and get byte result.

    :param output_vars: output variables which are the graph's end point.

        .. note::

            The underlying C++ API only accepts a var list. If a dict is given,
            the vars would be renamed to the given names.

    :param keep_var_name: level for keeping variable names:

        * 0: none of the names are kept
        * 1: (default)keep names of output vars
        * 2: keep names of all (output and internal) vars
    :param keep_param_name: whether to keep param names, so param values can be
        easily manipulated after loading model
    :param keep_opr_priority: whether to keep priority setting for operators
    :param strip_info_file: a string for path or a file handler. if is not None,
        then the dump information for code strip would be written to ``strip_info_file``
    :param append_json: will be check when `strip_info_file` is not None. if set
        true, the information for code strip will be append to strip_info_file.
        if set false, will rewrite strip_info_file
    :return: dump result as byte string, and an instance of namedtuple
        :class:`CompGraphDumpResult`, whose fields are:

            * ``nr_opr`` number of operators dumped
            * ``tot_bytes`` total bytes for the whole graph
            * ``tensor_value_bytes`` bytes consumed for dumping tensor values
            * ``inputs`` names of input tensors
            * ``params`` list of names of dumped params
            * ``outputs`` names of output vars
    """
    ov = []
    if isinstance(output_vars, dict):
        used_vars = set()
        for name, var in output_vars.items():
            assert isinstance(var, VarNode), "bad output var: {!r}".format(var)
            assert var.id not in used_vars, (
                "var name is associated with a var object, so we can not have "
                "two names given to the same var: {}".format(var)
            )
            used_vars.add(var.id)
            var.name = name
            ov.append(var._node)
    else:
        for var in output_vars:
            assert isinstance(var, VarNode), "bad output var: {!r}".format(var)
            ov.append(var._node)

    stat = []
    inputs = []
    outputs = []
    params = []

    dump_content = _imperative_rt.dump_graph(
        ov,
        keep_var_name,
        keep_param_name,
        keep_opr_priority,
        stat,
        inputs,
        outputs,
        params,
    )

    dump_info = CompGraphDumpResult(*stat, inputs, outputs, params)

    if strip_info_file is not None:
        if isinstance(strip_info_file, str):
            if not os.path.exists(strip_info_file):
                os.mknod(strip_info_file)
            strip_info_file = open(strip_info_file, "r+")
        new_strip_dict = json.loads(_imperative_rt.get_info_for_strip(ov))
        ori_strip_dict = new_strip_dict
        json_content = strip_info_file.read()
        if append_json and len(json_content) != 0:
            # if there are contents in json file. Read them first and then append new information
            ori_strip_dict = json.loads(json_content)
            for k in ori_strip_dict:
                new_strip_dict_v = new_strip_dict.get(k)
                if new_strip_dict_v is not None:
                    for value in new_strip_dict_v:
                        if not value in ori_strip_dict[k]:
                            ori_strip_dict[k].append(value)
        ori_strip_dict["hash"] = dump_info.content_hash
        strip_info_file.seek(0)
        strip_info_file.truncate()
        json.dump(ori_strip_dict, strip_info_file)

    return dump_content, dump_info


CompGraphLoadResult = collections.namedtuple(
    "CompGraphLoadResult", ["graph", "output_vars_dict", "output_vars_list"]
)


def load_graph(fpath):
    """
    Load a serialized computing graph from file.

    :param fpath: Path or Handle of the input file
    :return: An instance of namedtuple :class:`CompGraphLoadResult`,
        whose fields are:

            * ``graph`` loaded CompGraph
            * ``output_vars_dict`` A Python dict, mapping name to output SymbolVar
            * ``output_vars_list`` A Python list, containing output vars in the
                                   order passed to serialize_comp_graph_to_file
    """
    output_vars_map = []
    output_vars_list = []
    if isinstance(fpath, str):
        buf = open(fpath, "rb").read()
    else:
        buf = fpath.read()
    cg = _imperative_rt.load_graph(buf, output_vars_map, output_vars_list)
    return CompGraphLoadResult(cg, dict(output_vars_map), output_vars_list)


def _wrap(x):
    if isinstance(x, collections.abc.Sequence):
        return type(x)(map(_wrap, x))
    if hasattr(x.graph, "_wrap"):
        return x.graph._wrap(x)
    else:
        return x


def _unwrap(x):
    if isinstance(x, collections.abc.Sequence):
        return type(x)(map(_unwrap, x))
    if isinstance(x, VarNode):
        return x._node
    else:
        return x


@apply.register()
def _(op: OpDef, *args: VarNode):
    outputs = _imperative_rt.invoke_op(op, _unwrap(args))
    return _wrap(outputs)


@apply.register()
def _(op: BackwardGraph, *args: VarNode):
    assert args
    graph = args[0].graph
    return op.interpret(
        lambda op, args: apply(op, *args), graph._make_const_for_backward, args
    )


def input_callback(callback, *args, device=None, dtype=None, shape=None, graph=None):
    outputs = _imperative_rt.input_callback(
        callback, as_device(device).to_c(), dtype, shape, _unwrap(args), graph=graph
    )
    value, dummy = _wrap(outputs)
    return value, dummy


class InputNode(OpNode):
    def __init__(
        self,
        *args: VarNode,
        device=None,
        dtype=None,
        shape=None,
        graph=None,
        use_static_shape=False
    ):
        r = _imperative_rt.DeviceTensorNDRendezvous()
        if device is not None:
            device = as_device(device).to_c()
        outputs = _imperative_rt.input_callback(
            r,
            device,
            dtype,
            shape,
            _unwrap(args),
            graph=graph,
            use_static_shape=use_static_shape,
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


class VirtualDepNode(OpNode):
    def __init__(self, vars, device=""):
        out = _imperative_rt.virtual_dep(_unwrap(vars), device)
        super().__init__(out)
