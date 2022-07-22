# -*- coding: utf-8 -*-
import collections
import json
import os
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Union

import numpy as np

from .. import _imperative_rt
from .._imperative_rt import GraphOptimizeOptions, SerializationFormat
from .._imperative_rt.core2 import apply
from .._wrap import as_device
from ..ops.builtin import OpDef


def set_priority_to_id(dest_vars):
    r"""For all oprs in the subgraph constructed by dest_vars,
    sets its priority to id if its original priority is zero.

    Args:
        dest_vars: target vars representing the graph.
    """
    dest_vec = []
    for i in dest_vars:
        assert isinstance(i, _imperative_rt.VarNode)
        dest_vec.append(i)
    _imperative_rt.graph._set_priority_to_id(dest_vec)


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

    def _set_priority_to_id(self, dest_vars):
        set_priority_to_id(_unwrap(dest_vars))

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

    def make_const(self, data, dtype=None, device=None, name=None):
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
            return self._wrap(
                _imperative_rt.make_const(self, data, device, dtype, name)
            )

    def make_input(self, *args: "VarNode", device=None, dtype=None, shape=None):
        opnode = InputNode(*args, device=device, dtype=dtype, shape=shape, graph=self)
        return opnode.outputs[0]

    def make_h2d(self, *, dtype, device, shape=None, name=None):
        device = as_device(device).to_c()
        return self._wrap(_imperative_rt.make_h2d(self, device, dtype, shape, name))

    def _to_json(self, filename):
        # debug interface
        if self._function:
            js = json.loads(self._function._to_json())
            json.dump(js, open(filename, "w"))
        else:
            print("this function should be called after compilation.")


class VarNode:
    def __init__(self, node: _imperative_rt.VarNode):
        self._node = node
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
    r"""Applies optimize_for_inference pass for computing graph.

    Args:
        dest_vars: list of output vars in the computing graph

    Keyword Arguments:

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
        * enable_nchw64 --
          whether to use NCHW64 data layout, used for fast int4
          support on Nvidia GPU.
        * enable_fuse_conv_bias_nonlinearity: whether to fuse conv+bias+nonlinearty
          into one opr.
        * enable_fuse_conv_bias_with_z: whether to fuse conv_bias with z
          input for inference on nvidia backend(this optimization pass will
          result in mismatch of the precision of output of training and
          inference
        * enable_fuse_grain: fuse grain will be enable by default to fuse grain operator to huge operator, you can disable it.
          )
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
        "enable_nchw64": GraphOptimizeOptions.LayoutTransform.NCHW64,
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
    if kwargs.pop("enable_fuse_preprocess", False):
        inference_options.fuse_preprocess = True
    if kwargs.pop("enable_fuse_grain", True):
        inference_options.fuse_grain = True

    if kwargs:
        raise ValueError("unknown options: %s" % list(kwargs))

    dest_vars = _unwrap(dest_vars)
    res_vars = _imperative_rt.optimize_for_inference(dest_vars, inference_options)
    return _wrap(res_vars), inference_options.serialize()


def deserialize_infer_option(x: int) -> Dict[str, bool]:
    r"""Deserailize optimize options generated by ``imperative_rt.GraphOptimizeOptions``.

    Args:
        x: inference options represented by int.

    Returns:
        inference options represented by dict.
    """

    inference_options = GraphOptimizeOptions.deserialize(x)

    inference_optimize_layout_transform_map = {
        GraphOptimizeOptions.LayoutTransform.NHWCD4: "enable_hwcd4",
        GraphOptimizeOptions.LayoutTransform.NCHW4: "enable_nchw4",
        GraphOptimizeOptions.LayoutTransform.NCHW88: "enable_nchw88",
        GraphOptimizeOptions.LayoutTransform.NCHW32: "enable_nchw32",
        GraphOptimizeOptions.LayoutTransform.NCHW44: "enable_nchw44",
        GraphOptimizeOptions.LayoutTransform.NCHW44_DOT: "enable_nchw44_dot",
        GraphOptimizeOptions.LayoutTransform.CHWN4: "enable_chwn4",
        GraphOptimizeOptions.LayoutTransform.NCHW64: "enable_nchw64",
    }

    ret = dict()

    layout = inference_options.layout_transform
    if layout != GraphOptimizeOptions.LayoutTransform.DEFAULT:
        ret[inference_optimize_layout_transform_map[layout]] = True

    if inference_options.f16_io_f32_comp:
        ret["enable_io16xc32"] = True
    if inference_options.f16_io_comp:
        ret["enable_ioc16"] = True
    if inference_options.fuse_conv_bias_nonlinearity:
        ret["enable_fuse_conv_bias_nonlinearity"] = True
    if inference_options.fuse_conv_bias_with_z:
        ret["enable_fuse_conv_bias_with_z"] = True
    if inference_options.fuse_preprocess:
        ret["enable_fuse_preprocess"] = True
    if inference_options.fuse_grain:
        ret["enable_fuse_grain"] = True

    return ret


def modify_opr_algo_strategy_inplace(dest_vars, strategy: str):
    r"""C++ graph version of :func:`~.set_execution_strategy`. Used to inplacely modify
    dumped graph's fast-run strategy.

    Args:
        dest_vars: list of output vars in the computing graph.
        strategy: fast-run algorithms strategy.
    """
    dest_vars = _unwrap(dest_vars)
    _imperative_rt.modify_opr_algo_strategy_inplace(dest_vars, strategy)


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
    keep_opr_name: bool = False,
    keep_param_name: bool = False,
    keep_opr_priority: bool = False,
    no_change_graph: bool = False,
    strip_info_file=None,
    append_json=False,
    metadata=None,
    dump_format=None,
    model_version: int = 2
) -> Tuple[bytes, CompGraphDumpResult]:
    r"""serialize the computing graph of `output_vars` and get byte result.

    Args:
        output_vars: output variables which are the graph's end point.
        keep_var_name: level for keeping variable names:

            * 0: none of the names are kept
            * 1: (default)keep names of output vars
            * 2: keep names of all (output and internal) vars

        keep_opr_name: whether to keep operator names.
        keep_param_name: whether to keep param names, so param values can be
            easily manipulated after loading model
        keep_opr_priority: whether to keep priority setting for operators
        no_change_graph: whether to change the compute graph when dump, for
            model compatibility, some operators will convert to its compatible
            format in this version.

            * if set False, some operators maybe convert to other operator for
              compatibility, all operators will ensure compatibility.
            * if set True, no operator will change in the graph when dump.

        strip_info_file: a string for path or a file handler. if is not None,
            then the dump information for code strip would be written to ``strip_info_file``
        append_json: will be check when `strip_info_file` is not None. if set
            true, the information for code strip will be append to strip_info_file.
            if set false, will rewrite strip_info_file
        dump_format: using different dump formats.
        model_version: the model version of "FBS_V2", begin with version 2, this
            works only when dump format is "FBS_V2".

    Note:
        The underlying C++ API only accepts a var list. If a dict is given,
        the vars would be renamed to the given names.

    Returns:
        dump result as byte string, and an instance of namedtuple
        :class:`CompGraphDumpResult`, whose fields are:

        * ``nr_opr`` number of operators dumped
        * ``tot_bytes`` total bytes for the whole graph
        * ``tensor_value_bytes`` bytes consumed for dumping tensor values
        * ``inputs`` names of input tensors
        * ``params`` list of names of dumped params
        * ``outputs`` names of output vars
    """
    if isinstance(output_vars, dict):
        used_vars = set()
        for name, var in output_vars.items():
            assert var.id not in used_vars, (
                "var name is associated with a var object, so we can not have "
                "two names given to the same var: {}".format(var)
            )
            used_vars.add(var.id)
            var.name = name
        output_vars = list(output_vars.values())
    else:
        output_vars = list(output_vars)

    ov = _unwrap(output_vars)

    stat = []
    inputs = []
    outputs = []
    params = []

    dump_format_map = {
        None: None,
        "FBS_V2": SerializationFormat.FBS_V2,
        "FBS": SerializationFormat.FBS,
    }
    dump_format = dump_format_map[dump_format]

    dump_content = _imperative_rt.dump_graph(
        ov,
        keep_var_name,
        keep_opr_name,
        keep_param_name,
        keep_opr_priority,
        no_change_graph,
        metadata,
        dump_format,
        model_version,
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
    "CompGraphLoadResult", ["graph", "output_vars_dict", "output_vars_list", "metadata"]
)


def load_graph(fpath) -> CompGraphLoadResult:
    r"""Load a serialized computing graph from file.

    Args:
        fpath: Path or Handle of the input file

    Returns:
        An instance of namedtuple :class:`CompGraphLoadResult`,
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
    cg, metadata = _imperative_rt.load_graph(buf, output_vars_map, output_vars_list)
    return CompGraphLoadResult(cg, dict(output_vars_map), output_vars_list, metadata)


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
    return x


def apply_normal_varnode(op: OpDef, *args: VarNode):
    outputs = _imperative_rt.invoke_op(op, _unwrap(args))
    return _wrap(outputs)


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
        var = self.outputs[0]
        if isinstance(var, VarNode):
            return var.device
        else:
            return var.comp_node

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
