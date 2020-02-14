# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""the megbrain python package

Note that all the submodules are automatically imported, so you usually only
need to ``import megengine._internal as mgb``.
"""

import collections
import json
import os
import sys

import numpy as np

from . import comp_graph_tools as cgtools
from . import config, craniotome, dtype
from . import global_init as _global_init
from . import helper as _helper
from . import mgb as _detail
from . import opr, opr_param_defs, plugin
from .exc import MegBrainError
from .logconf import get_logger
from .mgb import (
    CompGraph,
    CompNode,
    SharedND,
    SharedScalar,
    SymbolVar,
    TensorValueDumperContext,
    TensorValueLoaderContext,
)
from .mgb import as_comp_node as comp_node
from .mgb_helper import SharedNDLazyInitializer, callback_lazycopy, copy_output
from .plugin import CompGraphProfiler
from .plugin import GlobalInfkernFinder as _GlobalInfkernFinder
from .plugin import NumRangeChecker
from .version import __version__, version_info

if sys.version_info.major < 3:
    raise ImportError("megbrain requires python 3")


class ProxySharedNDAndSymbolVar(_detail.SymbolVar):
    """this is a :class:`.SymbolVar` with a corresponding :class:`.SharedND`.
    It can participate in graph computating and also provides :meth:`set_value`
    and :meth:`get_value`.  It should be constructed by :func:`make_shared`.
    """

    __shared_nd = None
    __kwargs = None

    def __init__(self, snd, comp_graph, name, **kwargs):
        self.__shared_nd = snd
        self.__kwargs = kwargs
        self.this = snd.symvar(comp_graph=comp_graph, name=name, **kwargs).this

    def set_value(self, v, **kwargs):
        ret = self.__shared_nd.set_value(v, **kwargs)
        self._reeval_if_eager_eval()
        return ret

    def get_value(self):
        return self.__shared_nd.get_value()

    def reset_zero(self):
        self.__shared_nd.reset_zero()


def make_shared(
    comp_node,
    *,
    dtype=None,
    shape=None,
    value=None,
    comp_graph=None,
    name=None,
    volatile=None
):
    """make a shared tensor which is stored on device and could be modified
    later, either as a :class:`.SymbolVar` or a :class:`.SharedND` object

    :param comp_node: computing node
    :type comp_node: :class:`.CompNode`
    :param dtype: data type; if it is None, then dtype of value would be used
        if value is not None, and float32 would be used as default dtype if
        value is None
    :type dtype: :class:`numpy.dtype` compatible
    :param value: initializing value
    :type value: None or :class:`numpy.ndarray`
    :param comp_graph: the computing graph to which this shared value should
        belong; if provided, the retuned object could be used as a
        :class:`.SymbolVar`
    :type comp_graph: None or :class:`.CompGraph`
    :param name: node name to be used in computing graph; only meaningful if
        *comp_graph* is provided
    :param volatile: if *comp_graph* is given then *volatile* indicates whether
        shape or mem ptr of this SharedND can be changed
    :rtype: :class:`.SharedND` if *comp_graph* is not given; or
        :class:`ProxySharedNDAndSymbolVar` otherwise
    """
    if dtype is None:
        if value is not None:
            value = np.ascontiguousarray(value)
            dtype = to_mgb_supported_dtype(value.dtype)
        else:
            dtype = np.float32
    comp_node = _detail.as_comp_node(comp_node)
    rst = _detail.SharedND(comp_node, dtype)
    if value is not None:
        assert shape is None, "could not provide both value and shape"
        rst.set_value(value)
    elif shape is not None:
        rst._set_init_shape(shape)
    if comp_graph is None:
        assert name is None and volatile is None
        return rst
    assert isinstance(comp_graph, CompGraph), "expect CompGraph but got {}".format(
        comp_graph
    )
    if volatile is None:
        volatile = False
    else:
        assert isinstance(volatile, bool)
    return ProxySharedNDAndSymbolVar(rst, comp_graph, name, volatile=volatile)


def make_immutable(comp_node, comp_graph, value, *, dtype=None, name=None):
    """make a graph node containing an immutable tensor from host tensor value

    :param dtype: required data type; if not None, the data would be converted
        to that type; otherwise
    """

    comp_node = _detail.as_comp_node(comp_node)
    assert isinstance(
        comp_graph, _detail.CompGraph
    ), "expect CompGraph but got {!r}".format(comp_graph)

    config = _detail.make_opr_config(name, comp_node)
    return _helper.cvt_opr_result(
        _detail._make_immutable(comp_graph, value, dtype, config)
    )


def make_arg(
    comp_node,
    comp_graph,
    *,
    dtype=np.float32,
    shape=None,
    name=None,
    value=None,
    enable_static_infer=True
):
    """make an argument to be passed to compiled function during runtime;

    :type shape: None or tuple of int
    :param shape: expected tensor shape to be used for shape inferring; actual
        tesor shape could be different
    :type name: str
    :param name: name of the generated var node
    :type value: None or ndarray-compatible
    :param value: initial value used for static inference; if not given, static
        infer would be deferred to first graph execution
    :param enable_static_infer: whether to enable static inference for this var
    """
    host_val = mgb._HostSharedND(comp_node, dtype)

    if value is not None:
        value = np.ascontiguousarray(value, dtype=dtype)
        if shape is None:
            shape = value.shape
        else:
            assert shape == value.shape
    if shape is not None:
        host_val._resize(shape)

    if value is not None:
        host_val.set_value(value)

    return _helper.cvt_opr_result(
        ProxySharedNDAndSymbolVar(
            host_val, comp_graph, name, enable_static_infer=enable_static_infer
        )
    )


def comp_graph(*, extra_opts=None, check_env_var=True):
    """allocate a new computing graph

    :param extra_opts: extra options to be set; would be updated (modified
        inplace) from ``MGB_COMP_GRAPH_OPT`` environment var. See
        :func:`.set_comp_graph_option` for list of supported options.
    :type extra_opts: dict
    :param check_env_var: whether to check environment vars
    :type check_env_var: bool

    :return: the comp graph object
    :rtype: :class:`.CompGraph`
    """
    cg = _detail.CompGraph()
    if extra_opts is None:
        extra_opts = {}
    if check_env_var:
        setting = os.getenv("MGB_COMP_GRAPH_OPT")
        if setting:
            for item in setting.split(";"):
                k, v = item.split("=", 1)
                extra_opts.setdefault(k, v)
            get_logger().warning(
                "set comp graph option from env: {}".format(extra_opts)
            )
        user_data = os.getenv("MGB_COMP_GRAPH_USER_DATA")
        if user_data:
            storage = cg.user_data
            for ud in user_data.split(";"):
                k, v = ud.split("=", 1)
                storage[k] = eval(v)
        _GlobalInfkernFinder.add_graph(cg)
    for k, v in extra_opts.items():
        cg.set_option(k, v)
    return cg


def grad(
    target, wrt, warn_mid_wrt=True, use_virtual_grad=None, return_zero_for_nodep=True
):
    r"""compute symbolic grad

    :param target: grad target var
    :type target: :class:`.SymbolVar`
    :param wrt: with respect to which to compute the grad
    :type wrt: :class:`.SymbolVar` or Iterable[SymbolVar]
    :param warn_mid_wrt: whether to give warning if *wrt* is not endpoint
    :type warn_mid_wrt: bool
    :param use_virtual_grad: whether to use virtual grad opr, so fwd graph can
        be optimized before applying grad; if ``None`` is given, then virtual
        grad would be used if ``graph_opt_level >= 2``
    :type use_virtual_grad: :class:`bool` or ``None``
    :param return_zero_for_nodep: if *target* does not depend on *wrt*, set to True to return
        a zero-valued `.SymbolVar` rather than ``None``; can't be set to False when using
        virtual grad opr.
    :type return_zero_for_nodep: bool
    :rtype: :class:`.SymbolVar` or None
    :return: :math:`\frac{\partial\text{target}}{\partial\text{wrt}}`
    """
    if use_virtual_grad is None:
        use_virtual_grad = -1
    else:
        use_virtual_grad = 1 if use_virtual_grad else 0

    if isinstance(wrt, SymbolVar):
        wrts = [
            wrt,
        ]
    else:
        wrts = wrt

    assert isinstance(wrts, collections.Iterable)
    # return a invalid SymbolVar (with nullptr VarNode*) when return_zero_for_nodep is False
    # and target doesn't depend on wrt
    grads = _detail._grad(
        target, wrts, bool(warn_mid_wrt), use_virtual_grad, return_zero_for_nodep
    )
    grads = list(grads)

    for i in range(len(grads)):
        if not grads[i].valid:
            assert (
                not return_zero_for_nodep
            ), "invalid grad SymbolVar: target={}, wrt={}".format(target, wrts[i])
            grads[i] = None

    if len(grads) == 1:
        grads = grads[0]

    return grads


def current_grad_target(comp_graph):
    """get current target var to compute grad, used for implementing custom
    gradient"""
    return _detail._current_grad_target(comp_graph)


def inter_graph_trans_var(dest_graph, src):
    """get the corresponding var of *src* in *dest_graph*; assuming
    *dest_graph* is a copy of owner graph of *src*; usually used in callback of
    set_grad to get grad of vars in loop

    :param dest_graph: target computing graph
    :type dest_graph: :class:`.CompGraph`
    :param src: source var node
    :type src: :class:`.SymbolVar`
    :return: corresponding var in *dest_graph*
    :rtype: :class:`.SymbolVar`
    """
    return _detail._inter_graph_trans_var(dest_graph, src)


def get_graph_optimizer_replaced_var(src):
    """get optimized var corresponding to given var; usually used in callback
    of set_grad to get grad w.r.t. some var

    :param src: source var node
    :type src: :class:`.SymbolVar`
    :rtype: :class:`.SymbolVar`
    """
    return _detail._get_graph_optimizer_replaced_var(src)


CompGraphSerializationResult = collections.namedtuple(
    "CompGraphSerializationResult",
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


def serialize_comp_graph_to_file(
    fpath,
    output_vars,
    *,
    keep_var_name=1,
    keep_param_name=False,
    keep_opr_priority=False,
    tensor_value_dumper=None,
    output_strip_info=False,
    append=False,
    format=None,
    **kwargs
):
    """serialize this computing graph and write result to a file. Note:
    ``kwargs`` exists for backward compatibility; there is no additional
    arguments.

    :parma fpath: path for the output file
    :type fpath: ``str``
    :param output_vars: output variables that need to be retrieved when
        deserializing

        .. note::

            The underlying C++ API only accepts a var list. If a dict is given,
            the vars would be renamed to given names.

    :type output_vars: dict(name => :class:`.SymbolVar`), or a list of vars
    :param keep_var_name: level for keeping variable names:

        * 0: none of the names are kept
        * 1: keep names of output vars
        * 2: keep names of all (output and internal) vars
    :param keep_param_name: whether to keep param names, so param values can be
        easily manipulated after loading model
    :param keep_opr_priority: whether to keep priority setting for operators
    :param tensor_value_dumper: a callable to dump tensor values; it should
        only write the tensor value without layout information. It would be
        given a :class:`.TensorValueDumperContext` object as its sole argument.
    :param output_strip_info: if set to True, then a json file containing
        information for code strip would be written to ``fpath+'.json'``
    :param append: whether to open output file in append mode
    :return: an instance of namedtuple :class:`CompGraphSerializationResult`,
        whose fields are:

            * ``nr_opr`` number of operators dumped
            * ``tot_bytes`` total bytes for the whole graph
            * ``tensor_value_bytes`` bytes consumed for dumping tensor values
            * ``inputs`` names of input tensors
            * ``params`` list of names of dumped params
            * ``outputs`` names of output vars
    :param format: serialization format of the resulting model, should be either
        "mdl" or "fbs"; none means default.
    :type format: ``str``
    """

    assert isinstance(fpath, str), "bad file path: {!r}".format(fpath)
    ov = _detail._VectorSymbolVar()
    SUPPORTED_FORMATS = {
        # default
        None: _detail.GraphDumpFormat_FLATBUFFERS,
        "fbs": _detail.GraphDumpFormat_FLATBUFFERS,
    }
    resolved_fmt = SUPPORTED_FORMATS.get(format, None)
    if resolved_fmt is None:
        raise ValueError(
            "unknown format {} requested, supported ones are {}".format(
                format, list(filter(None, SUPPORTED_FORMATS.keys()))
            )
        )
    if isinstance(output_vars, dict):
        used_vars = set()
        for name, var in output_vars.items():
            assert isinstance(var, _detail.SymbolVar), "bad output var: {!r}".format(
                var
            )
            assert var.id not in used_vars, (
                "var name is associated with a var object, so we can not have "
                "two names given to the same var: {}".format(var)
            )
            used_vars.add(var.id)
            var.rename(name)
            ov.push_back(var)
    else:
        for i in output_vars:
            assert isinstance(i, _detail.SymbolVar), "bad output var: {!r}".format(i)
            ov.push_back(i)

    if tensor_value_dumper is not None:
        assert isinstance(tensor_value_dumper, collections.Callable)

        class Callback(_detail._TensorValueDumperCallback):
            def call(self, ctx, *, _f=tensor_value_dumper):
                _f(ctx)

        tensor_value_dumper = Callback()

    # for backward compatibility
    mangle_opr_name = kwargs.pop("mangle_opr_name", ov)
    if mangle_opr_name is not ov:
        get_logger().warning("mangle_opr_name is deprecated; use keep_var_name instead")
        keep_var_name = 1 if mangle_opr_name else 2
    mangle_param_name = kwargs.pop("mangle_param_name", ov)
    assert (
        not kwargs
    ), "extra kwargs provided to serialize_comp_graph_to_file: {}".format(kwargs)

    if mangle_param_name is not ov:
        get_logger().warning(
            "mangle_param_name is deprecated; use keep_param_name instead"
        )
        keep_param_name = not mangle_param_name

    inputs = _detail._VectorString()
    outputs = _detail._VectorString()
    params = _detail._VectorString()
    stat = _detail._VectorSizeT()

    _detail._serialize_comp_graph_to_file(
        fpath,
        append,
        resolved_fmt,
        ov,
        keep_var_name,
        keep_param_name,
        keep_opr_priority,
        tensor_value_dumper,
        stat,
        inputs,
        outputs,
        params,
    )

    dump_ret = CompGraphSerializationResult(
        *stat, list(inputs), list(outputs), list(params)
    )

    if output_strip_info:
        with open(fpath + ".json", "w") as fout:
            strip_info = _detail._get_info_for_strip(ov)
            strip_info_dict = json.loads(strip_info)
            strip_info_dict["hash"] = dump_ret.content_hash
            json.dump(strip_info_dict, fout)

    return dump_ret


CompGraphLoadResult = collections.namedtuple(
    "CompGraphLoadResult", ["graph", "output_vars_dict", "output_vars_list"]
)


def load_comp_graph_from_file(
    fpath, *, comp_node_mapper=None, tensor_value_loader=None
):
    """Load a serialized computing graph from file.

    :parma fpath: Path for the output file
    :type fpath: ``str``
    :param comp_node_mapper: A callable to modify comp node locator, takes old
        locator as argument and returns new locator.
    :type comp_node_mapper: Callable[[str], str]
    :param tensor_value_loader: A callable to load tensor values. It should
        read the tensor value with the given shape and dtype and return it as
        NumPy ndarray. It would be given a :class:`.TensorValueLoaderContext`
        object as its sole argument.
    :type tensor_value_loader: Callable[[TensorValueLoaderContext], numpy.ndarray]
    :return: An instance of namedtuple :class:`CompGraphLoadResult`,
        whose fields are:

            * ``graph`` loaded CompGraph
            * ``output_vars_dict`` A Python dict, mapping name to output SymbolVar
            * ``output_vars_list`` A Python list, containing output vars in the
                                   order passed to serialize_comp_graph_to_file
    """
    assert isinstance(fpath, str), "bad file path: {!r}".format(fpath)

    if comp_node_mapper is not None:
        assert isinstance(comp_node_mapper, collections.Callable)

        class Callback(_detail._CompNodeMapperCallback):
            def call(self, desc, *, _f=comp_node_mapper):
                return _f(desc)

        comp_node_mapper = Callback()
    if tensor_value_loader is not None:
        assert isinstance(tensor_value_loader, collections.Callable)

        class Callback(_detail._TensorValueLoaderCallback):
            def call(self, ctx, *, _f=tensor_value_loader):
                return _f(ctx)

        tensor_value_loader = Callback()
    output_vars_map = _detail._VectorPairStringSymbolVar()
    output_vars_list = _detail._VectorSymbolVar()
    cg = _detail._load_comp_graph_from_file(
        fpath, comp_node_mapper, tensor_value_loader, output_vars_map, output_vars_list
    )
    return CompGraphLoadResult(cg, dict(list(output_vars_map)), list(output_vars_list))


def optimize_for_inference(
    output_vars,
    *,
    f16_io_f32_comp=False,
    f16_io_comp=False,
    use_nhwcd4=False,
    fuse_conv_bias_nonlinearity=False,
    use_tensor_core=False,
    fuse_conv_bias_with_z=False,
    use_nchw88=False
):
    """optimize computing graph for inference

    This applies a predefined set of optimization passes. Refer to the mnist
    sdk example and C++ code for fine-grained control.

    :param output_vars: output symvars
    :type output_vars: list of :class:`.SymbolVar`
    :param f16_io_f32_comp: whether to use float16 for I/O between oprs and use
        float32 as internal computation precision. Note the output var would be
        changed to float16
    :param f16_io_comp: whether to use float16 for both I/O and computation
        precision
    :param use_nhwcd4: whether to use NHWCD4 data format. This is faster on some
        OpenCL devices
    :param fuse_conv_bias_nonlinearity: whether to fuse conv+bias+nonlinearty
        into one opr. This is supported only in NHWCD4 format.
    :param use_nchw88: whether to use NCHW4 tensor format. This maybe faster some
        times.


    :return: list of transformed vars corresponding to given output vars
    """

    assert isinstance(output_vars, (list, tuple))
    opt = _detail._OptimizeForInferenceOptions()
    settings = locals()
    for i in [
        "f16_io_f32_comp",
        "f16_io_comp",
        "use_nhwcd4",
        "fuse_conv_bias_nonlinearity",
        "use_tensor_core",
        "fuse_conv_bias_with_z",
        "use_nchw88",
    ]:
        if settings[i]:
            getattr(opt, "enable_{}".format(i))()
    vec = _detail._VectorSymbolVar()
    for i in output_vars:
        assert isinstance(i, _detail.SymbolVar), "bad var: {}".format(i)
        vec.push_back(i)
    return list(_detail._optimize_for_inference(vec, opt))


def get_opr_fp_graph_exec(comp_graph, output_vars):
    """get opr footprint and graph exec info

    This function will recompile the compute graph, the AsyncExecutable compiled
    before will be invalid.

    :param comp_graph: ComputingGraph
    :param output_vars: list of :class:'.SymbolVar'
    """
    assert isinstance(output_vars, (list, tuple))
    vec = _detail._VectorSymbolVar()
    for i in output_vars:
        assert isinstance(i, _detail.SymbolVar), "bad var: {}".format(i)
        vec.push_back(i)
    return json.loads(_detail._get_opr_fp_graph_exec(comp_graph, output_vars))


def to_mgb_supported_dtype(dtype_):
    """get the dtype supported by megbrain nearest to given dtype"""
    if dtype.is_lowbit(dtype_) or dtype.is_quantize(dtype_):
        return dtype_
    return _detail._to_mgb_supported_dtype(dtype_)
