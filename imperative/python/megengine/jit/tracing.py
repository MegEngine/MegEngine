# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import collections
import contextlib
import functools
import itertools
import json
import os
import pickle
from typing import Any

import numpy as np

from ..core._imperative_rt import GraphProfiler, GraphProfiler2, SerializationMetadata
from ..core._imperative_rt.core2 import Tensor as RawTensor
from ..core._imperative_rt.core2 import (
    TensorWeakRef,
    apply,
    set_tracing,
    skip_tracing,
    unset_tracing,
)
from ..core._imperative_rt.ops import (
    AssertEqual,
    CollectiveComm,
    RemoteRecv,
    RemoteSend,
)
from ..core._trace_option import set_symbolic_shape
from ..core._wrap import device as as_device
from ..core.ops.builtin import BatchNorm, OpDef
from ..core.ops.special import Const
from ..core.tensor import megbrain_graph as G
from ..core.tensor.utils import setscalar
from ..utils.naming import AutoNaming
from ..utils.profiler import is_profiling
from .dtr_config import DTRConfig
from .graph_opt_config import GraphOptimizationConfig
from .sublinear_memory_config import SublinearMemoryConfig


def _input_node_use_static_shape():
    return os.environ.get("MEGENGINE_INPUT_NODE_USE_STATIC_SHAPE") is not None


class TraceMismatchError(RuntimeError):
    pass


active_trace = None


def is_tracing():
    if active_trace is None:
        return False
    else:
        return not skip_tracing


@contextlib.contextmanager
def exclude_from_trace():
    global skip_tracing
    if skip_tracing:
        yield
        return
    try:
        skip_tracing = True
        unset_tracing()
        if active_trace is not None:
            active_trace._begin_excluded_region()
        yield
    finally:
        skip_tracing = False
        set_tracing()


class TensorInfo:
    __slots__ = (
        # collected attributes
        "name",
        "external",
        "data_read",
        "shape_read",
        "value_read",
        "exported",
        "device",
        "dtype",
        "shape",
        "is_const",
        "bound_data",
        # resources for execution
        "varnode",
        "data_setter",
        "shape_reader",
        "value_reader",
        "data_reader",
    )

    def __init__(self):
        self.name = None
        self.exported = None
        self.data_read = None
        self.shape_read = None
        self.value_read = None
        self.bound_data = None

        self.data_setter = None
        self.shape_reader = None
        self.value_reader = None
        self.data_reader = None


_io_op_types = {AssertEqual, CollectiveComm, RemoteSend, RemoteRecv}


class trace:
    """
    Wraps a callable and provide:

    * tracing via :meth:`.trace` and :meth:`.dump`
    * accelerated evalutaion via :meth:`.__call__`

    :param function: the function will be traced.
    :param symbolic: whether to apply symbolic execution for tracing. Default: False
    :param capture_as_const: capture global vars or closures as const value. Default: False
    :param sublinear_memory_config: configuration for sublinear memory optimization.
        If not None, it enables sublinear memory optimization with given setting.
    :param profiling: whether to profile compiled trace. Default: False
    :param opt_level: optimization level for compiling trace. Default: 2
    :param graph_opt_config: configuration for graph optimization. Default: None
    :param symbolic_shape: whether to use symbolic shape for tracing. Default: True
    """

    def __new__(cls, *args, **kwargs):
        if not args:
            return functools.partial(cls, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        function,
        symbolic=False,
        capture_as_const=False,
        sublinear_memory_config: SublinearMemoryConfig = None,
        dtr_config: DTRConfig = None,
        profiling: bool = False,
        opt_level: int = 2,
        graph_opt_config: GraphOptimizationConfig = None,
        symbolic_shape: bool = True,
    ):
        self.__wrapped__ = function
        self._symbolic = symbolic
        self._capture_as_const = capture_as_const
        self._sublinear_memory_config = sublinear_memory_config
        self._dtr_config = dtr_config
        self._profiling = profiling
        self._profiler = None
        self._profiler2 = None
        self._graph_opt_level = opt_level
        self._graph_opt_config = graph_opt_config
        self._symbolic_shape = symbolic_shape
        self._output_handles = set()

        self._reset()

    def _reset(self):
        self._untraced = True
        self._tinfo = []  # handle -> TensorInfo
        self._seq = []
        self._pc = 0
        self._graph = None
        self._need_reset_nodes = None
        self._lazy_eval_graph = None
        self._lazy_eval_tensors = set()
        self._lazy_eval_links = None
        self._active_tensors = set()
        self._tensor_remaps = None
        self._inputs_to_restore = None
        self._arg_bindings = None
        self._kwarg_bindings = None
        self._output_bindings = None
        self._output_names = None

    def _new_handle(self):
        handle = len(self._tinfo)
        info = TensorInfo()
        self._tinfo.append(info)
        return handle, info

    def _apply_op(self, op, args):
        assert not self._untraced
        # check against trace
        if self._pc >= len(self._seq):
            raise TraceMismatchError("trace should end here, but more op observed")
        record = self._seq[self._pc]
        op_, ihandles, ohandles = record
        if (isinstance(op_, str) and op_ == "Const") or (op != op_):
            raise TraceMismatchError("op different from last time")
        if len(ihandles) != len(args):
            raise TraceMismatchError("op input size different from last time")

        # check all inputs of crrent op
        for h, x in zip(ihandles, args):
            info = self._tinfo[h]
            if info.external:
                if (
                    x._compiled_info is not None
                    and not self._tinfo[x._mixin_handle].exported
                ):
                    raise TraceMismatchError(
                        "failed to capture: input was an external tensor "
                        "last time, got an internal tensor this time"
                    )
                if info.bound_data:
                    if x._compiled_info is not None:
                        raise TraceMismatchError(
                            "const capture violated: was an external tensor "
                            "last time, got an internal tensor this time"
                        )
                    if x._handle != info.bound_data._handle:
                        if not np.array_equal(x.numpy(), info.bound_data.numpy()):
                            raise TraceMismatchError(
                                "const capture violated: got "
                                "a different tensor this time"
                            )
                else:
                    if info.dtype != x.dtype:
                        raise TraceMismatchError(
                            "failed to capture: different dtype from last time"
                        )
                    if info.device != x.device:
                        raise TraceMismatchError(
                            "failed to capture: different device from last time"
                        )
                    info.data_setter.set_value(x._dev_tensor())
            else:
                if x._mixin_handle == -1:
                    if x._handle not in self._tensor_remaps:
                        raise TraceMismatchError(
                            "unexpected capture: trying to use an external tensor as "
                            "input, but that input was an internal tensor last time"
                        )
                    else:
                        x._mixin_handle = self._tensor_remaps[
                            x._handle
                        ]._CompiledTensorProxy__handle
                if x._mixin_handle != h:
                    raise TraceMismatchError(
                        "mis-wiring: input edge to an data flow "
                        "graph node is different from last time"
                    )

        self._pc += 1
        outputs = []
        for h in ohandles:
            info = self._tinfo[h]
            # generate output tensor and create compied info
            y = RawTensor(info.varnode)
            y._compiled_info = CompiledTensorProxy(h)
            y._mixin_handle = h
            outputs += [y]
            self._active_tensors.add(TensorWeakRef(y))
        self._output_handles.update(ohandles)
        return outputs

    def _apply_const(self, value, dtype, device):
        assert not self._untraced
        # check against trace
        if self._pc >= len(self._seq):
            raise TraceMismatchError("trace should end here, but more op observed")
        record = self._seq[self._pc]
        op_, ihandles, ohandles = record
        # Const op is represented by a str
        assert isinstance(op_, str) and op_ == "Const"

        eq = np.all(np.atleast_1d(value) == self._tinfo[ohandles[0]].bound_data.numpy())
        if not eq:
            raise TraceMismatchError(
                "const tensor violated: got a different tensor this time"
            )

        self._pc += 1
        (h,) = ohandles
        outputs = [self._tinfo[h].bound_data]
        return outputs

    # run in first step, record information for trace
    def _record_op(self, op, inputs, outputs):
        if skip_tracing:
            for x in inputs:
                h = getattr(x, "_mixin_handle", -1)
                if h >= 0:
                    self._tinfo[h].data = True
            return

        ihandles = []
        for x in inputs:
            h = getattr(x, "_mixin_handle", -1)
            if h < 0 or (not self._capture_as_const and self._tinfo[h].exported):
                h, info = self._new_handle()
                name = AutoNaming.gen_name(x)
                info.name = name
                info.external = True
                info.device = x.device
                info.dtype = x.dtype
                info.shape = x.shape
                if self._capture_as_const:
                    info.bound_data = RawTensor(
                        x.numpy(), x.dtype, x.device, False, name
                    )

            ihandles.append(h)

        ohandles = []
        for x in outputs:
            h, info = self._new_handle()
            ohandles.append(h)
            info.external = False
            x._mixin_handle = h
            x._recording = True
            x._trace_mixin_info = info
            self._active_tensors.add(TensorWeakRef(x))
            if self._symbolic:
                self._lazy_eval_tensors.add(TensorWeakRef(x))

        self._seq.append((op, tuple(ihandles), tuple(ohandles)))

    def _record_const(self, outputs):
        if skip_tracing:
            (x,) = outputs
            h = getattr(x, "_mixin_handle", -1)
            if h >= 0:
                self._tinfo[h].data_read = True
            return

        (x,) = outputs
        h, info = self._new_handle()
        ohandles = [h]
        info.external = True
        info.device = x.device
        info.dtype = x.dtype
        info.shape = x.shape
        info.bound_data = x
        info.is_const = True
        x._mixin_handle = h
        x._recording = True
        x._trace_mixin_info = info
        if self._symbolic:
            self._lazy_eval_tensors.add(TensorWeakRef(x))
        self._seq.append(("Const", tuple(), tuple(ohandles)))

    def _set_active(self, active: bool):
        global active_trace
        if active:
            if active_trace:
                raise NotImplementedError("sorry, not implemented: nested trace")
            active_trace = self
        else:
            assert active_trace is self
            active_trace = None

    def _init_trace(self, symbolic: bool):
        if symbolic:
            self._lazy_eval_graph = G.Graph()
            self._apply_graph_options(self._lazy_eval_graph)
            self._lazy_eval_links = ()

    def _take_escaped_tensors(self):
        escaped_tensors = tuple(filter(lambda x: x() is not None, self._active_tensors))
        self._active_tensors.clear()
        return escaped_tensors

    def _lazy_eval(self, lazy_eval_graph, lazy_eval_tensors, lazy_eval_links):
        lazy_eval_tensors = [x() for x in lazy_eval_tensors]
        lazy_eval_tensors = [x for x in lazy_eval_tensors if x is not None]
        readers = [G.OutputNode(x._varnode).outputs[0] for x in lazy_eval_tensors]
        self._apply_graph_options(lazy_eval_graph)
        lazy_eval_graph.options.graph_opt_level = self._graph_opt_level
        lazy_eval_graph._set_priority_to_id([*lazy_eval_links, *readers])
        lazy_eval_graph.compile(*lazy_eval_links, *readers)
        self._execute_graph(lazy_eval_graph)
        lazy_eval_graph.wait()
        for r, x in zip(readers, lazy_eval_tensors):
            # get values from lazy_eval_graph and assign to lazy_eval tensor
            x._handle = RawTensor(r.op.get_value())._handle
            x._reset_varnode()

    @contextlib.contextmanager
    def _setup(self):
        interrupted = False

        def do_enter():
            set_tracing()
            self._save_symbolic_shape = set_symbolic_shape(self._symbolic_shape)
            self._set_active(True)
            if self._untraced:
                self._init_trace(self._symbolic)
            else:
                if self._graph is None:
                    self._compile()
                self._execute_graph(self._graph)

        def do_finalize():
            escaped_tensors = self._take_escaped_tensors()
            if self._untraced:
                for x in escaped_tensors:
                    if x():
                        info = self._tinfo[x()._mixin_handle]
                        info.data_read = True
                        x()._mixin_handle = -1
                        x()._recording = False
                if self._inputs_to_restore:
                    for x in self._inputs_to_restore:
                        x._mixin_handle = -1
                        x._recording = False
                if self._symbolic and (
                    self._lazy_eval_tensors or self._lazy_eval_links
                ):
                    # eval lazy eval tensors
                    self._lazy_eval(
                        self._lazy_eval_graph,
                        self._lazy_eval_tensors,
                        self._lazy_eval_links,
                    )
                    self._lazy_eval_graph = None
                    self._lazy_eval_tensors = None
                    self._lazy_eval_links = None
                self._untraced = False
            else:
                # compiled_tensor leaks
                if self._pc == len(self._seq):
                    for x in escaped_tensors:
                        try:
                            assign_raw_tensor(x(), RawTensor(x()._dev_tensor()))
                        except RuntimeError:
                            # TraceMismatchError thrown in do_exit
                            pass
                    self._graph.wait()
                    self._reset_exec_env()

            # reset status
            self._pc = 0
            self._tensor_remaps = None
            self._set_active(False)
            set_symbolic_shape(self._save_symbolic_shape)
            unset_tracing()

        def do_exit():
            unset_tracing()
            if not self._untraced and self._pc != len(self._seq):
                raise TraceMismatchError("premature end")
            if not self._symbolic or not self._untraced:
                # reset output tensors
                for x in self._active_tensors.copy():
                    strong_x = x()
                    if strong_x is not None:
                        strong_x._dev_tensor()
                        strong_x._reset_varnode()
                        strong_x._mixin_handle = -1
                        strong_x._recording = False
                        strong_x._trace_mixin_info = None

        try:
            do_enter()
            yield
            do_exit()
        except:
            interrupted = True
            raise
        finally:
            do_finalize()
            if interrupted:
                self._reset()

    def _begin_excluded_region(self):
        if self._capture_as_const:
            raise RuntimeError(
                "exclude_from_trace cannot be used with capture_as_const"
            )
        if self._untraced:
            # conditionally reading a compiled tensor in excluded region
            # is permitted, so we have to assume every tensor might be read
            for x in self._active_tensors:
                strong_x = x()
                if strong_x:
                    info = self._tinfo[strong_x._mixin_handle]
                    info.exported = True
                    info.data_read = True
        else:
            for x in self._active_tensors:
                strong_x = x()
                if strong_x:
                    strong_x._dev_tensor()

    def _apply_graph_options(self, graph):

        graph.options.no_force_inplace = True
        graph.options.seq_opt.enable_seq_comp_node_opt = False
        graph.options.graph_opt_level = self._graph_opt_level
        if self._dtr_config is not None:
            graph.options.enable_dtr_memory_opt = True
            graph.options.dtr_config.eviction_threshold = (
                self._dtr_config.eviction_threshold
            )
            graph.options.dtr_config.evictee_minimum_size = (
                self._dtr_config.evictee_minimum_size
            )
        # graph optimization
        if self._graph_opt_config is not None:
            mapping = {None: 0, False: 1, True: 2}
            jit_config = graph.options.graph_opt.jit_config
            jit_config.fuse_dimshuffle = mapping[
                self._graph_opt_config.jit_fuse_dimshuffle
            ]
            jit_config.fuse_reduce = mapping[self._graph_opt_config.jit_fuse_reduce]
        # sublinear
        if self._sublinear_memory_config is not None:
            graph.options.enable_sublinear_memory_opt = True
            sublinear_config = graph.options.sublinear_mem_config
            sublinear_config.lb_memory = self._sublinear_memory_config.lb_memory
            sublinear_config.genetic_nr_iter = (
                self._sublinear_memory_config.genetic_nr_iter
            )
            sublinear_config.genetic_pool_size = (
                self._sublinear_memory_config.genetic_pool_size
            )
            sublinear_config.thresh_nr_try = self._sublinear_memory_config.thresh_nr_try
            sublinear_config.num_worker = self._sublinear_memory_config.num_worker
        # profile
        if self._profiling:
            self._profiler = GraphProfiler(graph)
        self._profiler2 = None
        if int(os.getenv("MEGENGINE_INPLACE_UPDATE", "0")):
            graph.options.var_sanity_check_first_run = False

    def _execute_graph(self, graph: G.Graph, *args):
        if is_profiling() and (self._profiler2 is None):
            self._profiler2 = GraphProfiler2(graph)
        elif not is_profiling() and (self._profiler2 is not None):
            self._profiler2 = None
        graph.execute(*args)

    def _compile(self):
        graph = self._graph = G.Graph()
        graph.options.async_exec_level = 0b100
        self._apply_graph_options(graph)
        need_reset_nodes = self._need_reset_nodes = []
        # links enforce ordering of I/O nodes
        in_out_links = ()
        io_links = ()
        readers = []

        if self._capture_as_const:
            for h in itertools.chain(self._arg_bindings, self._kwarg_bindings.values()):
                info = self._tinfo[h]
                opnode = info.data_setter = G.InputNode(
                    device=info.device,
                    dtype=info.dtype,
                    shape=info.shape or (1,),
                    graph=graph,
                    use_static_shape=_input_node_use_static_shape(),
                )
                need_reset_nodes.append(opnode)
                info.varnode = opnode.outputs[0]
                in_out_links += opnode.outputs[1:]

        for op, ihandles, ohandles in self._seq:
            if isinstance(op, str) and op == "Const":
                assert len(ihandles) == 0
                (h,) = ohandles
                info = self._tinfo[h]
                if not hasattr(info, "varnode"):
                    assert info.external
                    assert info.bound_data
                    info.varnode = graph.make_const(
                        info.bound_data.numpy(),
                        info.bound_data.dtype,
                        info.bound_data.device,
                    )
                continue

            require_links = type(op) in _io_op_types
            ivars = []
            for i, h in enumerate(ihandles):
                info = self._tinfo[h]
                if not hasattr(info, "varnode"):
                    assert info.external
                    if info.bound_data:
                        if getattr(info, "is_const", False):
                            info.varnode = graph.make_const(
                                info.bound_data.numpy(),
                                info.bound_data.dtype,
                                info.bound_data.device,
                            )
                        else:
                            info.varnode = graph.make_const(
                                info.bound_data._dev_tensor()
                                # info.bound_data.numpy()
                            )
                    else:
                        opnode = info.data_setter = G.InputNode(
                            *in_out_links,
                            device=info.device,
                            dtype=info.dtype,
                            shape=info.shape or (1,),
                            graph=graph,
                            use_static_shape=_input_node_use_static_shape(),
                        )
                        need_reset_nodes.append(opnode)
                        info.varnode, *in_out_links = opnode.outputs
                if require_links and i == 0 and len(io_links) > 0:
                    opnode = G.VirtualDepNode(
                        [info.varnode, *io_links], str(io_links[0].device)
                    )
                    info.varnode = opnode.outputs[0]
                    io_links = (info.varnode,)

                ivars.append(info.varnode)

            ovars = G.apply_normal_varnode(op, *ivars)

            if require_links and len(ovars) > 0:
                io_links = (ovars[0],)
            assert len(ovars) == len(ohandles)
            for h, v in zip(ohandles, ovars):
                info = self._tinfo[h]
                info.varnode = v

                def add_reader(opnode):
                    nonlocal in_out_links
                    need_reset_nodes.append(opnode)
                    readers.append(opnode.outputs[0])
                    in_out_links = opnode.outputs

                if info.data_read:
                    # Shape can be obtained from data so doesn't need its own
                    # output node. On the other hand, value is read separately
                    # to leverage eager h2d copy
                    info.shape_read = False
                    opnode = info.data_reader = G.OutputNode(v, *in_out_links)
                    add_reader(opnode)
                if info.value_read:
                    opnode = info.value_reader = G.ValueOutputNode(v, *in_out_links)
                    add_reader(opnode)
                if info.shape_read:
                    opnode = info.shape_reader = G.AttrOutputNode(v, *in_out_links)
                    add_reader(opnode)

        graph.options.graph_opt_level = self._graph_opt_level
        graph._set_priority_to_id([*readers, *in_out_links, *io_links])
        graph.compile(*readers, *in_out_links, *io_links)

    def _reset_exec_env(self):
        for opnode in self._need_reset_nodes:
            opnode.reset()

    def __call__(self, *args, **kwargs):
        with self._setup():
            if self._capture_as_const:
                self._process_inputs(*args, **kwargs)
            outputs = self.__wrapped__(*args, **kwargs)
            if self._capture_as_const:
                self._process_outputs(outputs)
            return outputs

    def dump(
        self,
        file,
        *,
        arg_names=None,
        output_names=None,
        append=False,
        keep_var_name: int = 1,
        keep_opr_name: bool = False,
        keep_param_name: bool = False,
        keep_opr_priority: bool = False,
        strip_info_file=None,
        append_json=False,
        optimize_for_inference=True,
        user_info: Any = None,
        enable_metadata: bool = True,
        **kwargs
    ):
        r"""
        Serializes trace to file system.

        :param file: output file, could be file object or filename.
        :param arg_names: names of the input tensors in the traced function.
        :param output_names: names of the output tensors in the traced function,
            use the default name if not specified.
        :param append: whether output is appended to ``file``.
            Only works when ``file`` is str.
        :param keep_var_name: level for keeping variable names:

            * 0: none of the names are kept
            * 1: (default)keep names of output vars
            * 2: keep names of all (output and internal) vars
        :param keep_opr_name: whether to keep operator names.
        :param keep_param_name: whether to keep param names, so param values can be
            easily manipulated after loading model
        :param keep_opr_priority: whether to keep priority setting for operators
        :param strip_info_file: a string for path or a file handler. if is not None,
            then the dump information for code strip would be written to ``strip_info_file``
        :param append_json: will be check when `strip_info_file` is not None. if set
            true, the information for code strip will be append to strip_info_file.
            if set false, will rewrite strip_info_file
        :param optimize_for_inference: enbale optmizations,
            will skip all optimize options if this is False. Default: True
        :param user_info: any type object, which will be pickled to bytes.
        :param enable_metadata: whether to save metadata into output file.

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
            * enable_nchw64 --
                whether to use NCHW64 data layout, used for fast int4
                support on Nvidia GPU.

            * enable_fuse_conv_bias_nonlinearity: whether to fuse conv+bias+nonlinearty
                into one opr.
            * enable_fuse_conv_bias_with_z: whether to fuse conv_bias with z
                input for inference on nvidia backend(this optimization pass will
                result in mismatch of the precision of output of training and
                inference)
        """
        if not self._capture_as_const:
            raise ValueError(
                "you must specify capture_as_const=True at __init__ to use dump"
            )
        if self._untraced:
            raise RuntimeError("should run at least once before calling dump")
        if self._output_names and output_names:
            raise TypeError(
                "cannot specify output_names when output is already in dict format"
            )
        if output_names and not isinstance(output_names, collections.abc.Sequence):
            output_names = (output_names,)
        if output_names and len(output_names) != len(self._output_bindings):
            raise ValueError(
                "wrong number of output_names, should be {} values".format(
                    len(self._output_bindings)
                )
            )
        without_arg_names = arg_names is None
        if without_arg_names:
            arg_names = ["arg_%d" % i for i in range(len(self._arg_bindings))]
        if arg_names and not isinstance(arg_names, collections.abc.Sequence):
            arg_names = (arg_names,)
        if arg_names and len(arg_names) != len(self._arg_bindings):
            raise ValueError(
                "wrong number of arg_names, should be {} values".format(
                    len(self._arg_bindings)
                )
            )
        output_names = output_names or self._output_names

        def dumped_device(info):
            device_name = info.device.logical_name
            if device_name[:3] in ("cpu", "gpu", "xpu"):
                return as_device("xpux")
            return info.device

        h2v = {}
        graph = G.Graph()

        # apply graph_opt_level in dump
        if self._graph_opt_level is not None:
            graph.options.graph_opt_level = self._graph_opt_level
        for i, h in enumerate(self._arg_bindings):
            info = self._tinfo[h]
            h2v[h] = graph.make_h2d(
                dtype=info.dtype,
                device=dumped_device(info),
                shape=info.shape or (1,),
                name=info.name if without_arg_names and info.name else arg_names[i],
            )
        for k, h in self._kwarg_bindings.items():
            info = self._tinfo[h]
            h2v[h] = graph.make_h2d(
                dtype=info.dtype,
                device=dumped_device(info),
                shape=info.shape or (1,),
                name=k,
            )

        for op, ihandles, ohandles in self._seq:
            if isinstance(op, str) and op == "Const":
                assert len(ihandles) == 0
                (h,) = ohandles
                info = self._tinfo[h]
                if h not in h2v:
                    assert info.external
                    assert info.bound_data
                    h2v[h] = graph.make_const(
                        info.bound_data.numpy(),
                        dtype=info.dtype,
                        device=info.device,
                        name=info.name,
                    )
                continue
            ivars = []
            for h in ihandles:
                info = self._tinfo[h]
                if h not in h2v:
                    assert info.external
                    assert info.bound_data
                    h2v[h] = graph.make_const(
                        info.bound_data.numpy(),
                        dtype=info.dtype,
                        device=dumped_device(info),
                        name=info.name,
                    )
                ivars.append(h2v[h])
            if isinstance(op, BatchNorm):
                assert (
                    op.fwd_mode == BatchNorm.FwdMode.INFERENCE
                ), "can not dump BatchNorm in training mode, maybe you forget to do model.eval()?"
            ovars = G.apply_normal_varnode(op, *ivars)

            AutoNaming.record_opnode(ovars[0].op)

            assert len(ovars) == len(ohandles)
            h2v.update(zip(ohandles, ovars))

            for i in ohandles:
                name = AutoNaming.get_var_name(i)
                if name is not None:
                    h2v[i].name = name

        AutoNaming.remove_duplicate_names()

        dest_vars = []
        for i, h in enumerate(self._output_bindings):
            v = h2v[h]
            if output_names:
                v.name = output_names[i]
            dest_vars.append(v)

        if optimize_for_inference:
            dest_vars, optimize_options = G.optimize_for_inference(dest_vars, **kwargs)

        metadata = SerializationMetadata()
        if enable_metadata:
            metadata.user_info = pickle.dumps(user_info)
            metadata.is_valid = True
            metadata.graph_modified = False
            if optimize_for_inference:
                metadata.optimize_options = optimize_options

        if isinstance(file, str):
            permission = "wb" if append == False else "ab"
            file = open(file, permission)
        dump_content, dump_info = G.dump_graph(
            dest_vars,
            keep_var_name=keep_var_name,
            keep_opr_name=keep_opr_name,
            keep_param_name=keep_param_name,
            keep_opr_priority=keep_opr_priority,
            strip_info_file=strip_info_file,
            append_json=append_json,
            metadata=metadata,
        )
        file.write(dump_content)
        return dump_info

    def _process_inputs(self, *args, **kwargs):
        if self._untraced:
            self._inputs_to_restore = []

            def record_input(x):
                if x is None:
                    return
                h, info = self._new_handle()
                info.external = False
                info.name = x.c_name
                info.device = x.device
                info.dtype = x.dtype
                info.shape = x.numpy().shape
                x._mixin_handle = h
                x._recording = True
                x._trace_mixin_info = info
                self._inputs_to_restore.append(x)
                return h

            self._arg_bindings = []
            for i, x in enumerate(args):
                if not isinstance(x, RawTensor):
                    raise TypeError(
                        "positional arguments should all be tensor "
                        "but args[%d] cannot be recognized as one" % i
                    )
                self._arg_bindings.append(record_input(x))

            self._kwarg_bindings = {}
            for k, x in kwargs.items():
                if isinstance(x, RawTensor):
                    self._kwarg_bindings[k] = record_input(x)
        else:
            if len(args) != len(self._arg_bindings):
                raise TraceMismatchError("positional argument length mismatch")

            self._tensor_remaps = {}

            for i, (h, x) in enumerate(zip(self._arg_bindings, args)):
                if not isinstance(x, RawTensor):
                    raise TypeError(
                        "positional arguments should all be tensor "
                        "but args[%d] cannot be recognized as one" % i
                    )
                info = self._tinfo[h]
                if x.dtype != info.dtype:
                    raise TypeError("args[%d].dtype different from last time" % i)
                if x.device != info.device:
                    raise TypeError("args[%d].device different from last time" % i)
                info.data_setter.set_value(x._dev_tensor())
                self._tensor_remaps[x._handle] = CompiledTensorProxy(h)

            kwargs_tensors = {}
            for k, x in kwargs.items():
                if isinstance(x, RawTensor):
                    kwargs_tensors[k] = x
            if set(kwargs_tensors) != set(self._kwarg_bindings):
                too_many = set(kwargs_tensors) - set(self._kwarg_bindings)
                too_few = set(self._kwarg_bindings) - set(kwargs_tensors)
                if too_many:
                    raise TraceMismatchError(
                        "keyword arguments found to be tensor this time "
                        "but were non-tensor previously: %s" % " ".join(too_many)
                    )
                if too_few:
                    raise TraceMismatchError(
                        "keyword arguments found to be non-tensor this time "
                        "but were tensor previously: %s" % " ".join(too_few)
                    )
            for k, h in self._kwarg_bindings.items():
                x = kwargs_tensors[k]
                info = self._tinfo[h]
                if x.dtype != info.dtype:
                    raise TypeError("kwargs[%s].dtype different from last time" % k)
                if x.device != info.device:
                    raise TypeError("kwargs[%s].device different from last time" % k)
                info.data_setter.set_value(x._dev_tensor())
                self._tensor_remaps[x._handle] = CompiledTensorProxy(h)

    def _process_outputs(self, outputs):
        output_names = None
        if isinstance(outputs, collections.abc.Mapping):
            output_names, outputs = zip(*sorted(outputs.items()))
        elif not isinstance(outputs, collections.abc.Sequence):
            outputs = (outputs,)

        if not self._untraced:
            if output_names != self._output_names:
                too_many = set(output_names) - set(self._output_names)
                too_few = set(self._output_names) - set(output_names)
                if too_many:
                    raise TraceMismatchError(
                        "output has more keys than last time: %s" % " ".join(too_many)
                    )
                if too_few:
                    raise TraceMismatchError(
                        "output has less keys than last time: %s" % " ".join(too_few)
                    )
            if len(outputs) != len(self._output_bindings):
                raise TraceMismatchError("output size differs from last time")
        else:
            self._output_names = output_names
            self._output_bindings = []

        for i, x in enumerate(outputs):
            if not isinstance(x, RawTensor):
                raise TypeError("every item of return value should be tensor")
            if self._untraced:
                h = x._mixin_handle
                if h < 0:
                    raise RuntimeError("output is not computed from inputs")
                self._output_bindings.append(h)
            else:
                h = x._mixin_handle
                if h not in self._output_handles:
                    raise RuntimeError("output is not computed from inputs")
                if h != self._output_bindings[i]:
                    raise TraceMismatchError(
                        "retval[%s] is a different tensor than last time"
                        % (output_names and output_names[i] or i)
                    )

    def get_profile(self):
        """
        Get profiling result for compiled trace.

        :return: a json compatible object.
        """
        if not self._profiler:
            raise RuntimeError("trace is not set with profiling=True")
        return json.loads(self._profiler.get())

    def trace(self, *args, **kwargs):
        raise NotImplementedError(
            "trace is deemed unbeneficial with the new "
            "tracing mechanism. You should alwasy use __call__."
        )


class CompiledTensorProxy:
    """
    Duck-typed RawTensor
    """

    def __init__(self, handle):
        self.__handle = handle
        self._isscalar = False
        self.__info = active_trace._tinfo[handle]
        self.__shape = None
        self.__data = None
        self.__value = None

    @property
    def dtype(self):
        return self.__info.varnode.dtype

    @property
    def device(self):
        return self.__info.varnode.device

    @property
    def shape(self):
        if self._isscalar:
            return ()
        if self.__shape is None:
            if self.__info.shape_read:
                self.__shape = self.__info.shape_reader.get_value().shape
            elif self.__info.data_read:
                self.__shape = self._dev_tensor().shape
            else:
                # c++ will throw TraceReadError
                return None
        return self.__shape

    def numpy(self):
        if self.__value is None:
            if self.__info.value_read:
                self.__value = self.__info.value_reader.get_value()
            elif self.__info.data_read:
                self.__value = self._dev_tensor().numpy()
            else:
                # c++ will throw TraceReadError
                return None
        # c++ side will handle scalar case
        return self.__value

    def _dev_tensor(self):
        if self.__data is None:
            if not self.__info.data_read:
                # c++ will throw TraceReadError
                return None
            self.__data = self.__info.data_reader.get_value()
        return self.__data

    def __del__(self):
        if self.__info.shape_read and self.__shape is not None:
            self.__info.shape_reader.drop_value()
        if self.__info.value_read and self.__value is not None:
            self.__info.value_reader.drop_value()
        if self.__info.data_read and self.__data is not None:
            self.__info.data_reader.drop_value()


def assign_raw_tensor(lhs, rhs):
    lhs.__init__(rhs)


def apply_symbolic_mode(op: OpDef, *args: RawTensor):
    graph = active_trace._lazy_eval_graph
    ivars = []
    for x in args:
        var = getattr(x, "_varnode", None)
        if var:
            ivars.append(var)
        else:
            data_setter = G.InputNode(
                device=x.device,
                dtype=x.dtype,
                shape=x.numpy().shape or (1,),
                graph=graph,
                use_static_shape=True,
            )
            var = data_setter.outputs[0]
            ivars.append(var)
            data_setter.set_value(x._dev_tensor())

    require_links = type(op) in _io_op_types

    if require_links and active_trace._lazy_eval_links:
        assert len(ivars) > 0, "op should has at least one input"
        opnode = G.VirtualDepNode(
            [ivars[0], *active_trace._lazy_eval_links],
            str(active_trace._lazy_eval_links[0].device),
        )
        ivars[0] = opnode.outputs[0]
        active_trace._lazy_eval_links = (ivars[0],)

    ovars = G.apply_normal_varnode(op, *ivars)
    outputs = [RawTensor(o) for o in ovars]

    if require_links:
        active_trace._lazy_eval_links = (G.VarNode(outputs[0]._varnode),)

    return outputs


def apply_const_symbolic_mode(value, dtype, device, name):
    graph = active_trace._lazy_eval_graph
    # don't need to unset tracing
    # because varnode construction will ignore tracing flag
    ret = RawTensor(graph.make_const(value, dtype=dtype, device=device, name=name))
    if np.array(value).ndim == 0:
        setscalar(ret)
    return (ret,)


def apply_compiled_mode(op: OpDef, *args: RawTensor):
    if skip_tracing:
        args = [
            RawTensor(x._dev_tensor()) if x.__class__ is CompiledTensorProxy else x
            for x in args
        ]
        unset_tracing()
        ret = apply(op, *args)
        set_tracing()
        return ret
    return active_trace._apply_op(op, args)


def apply_const_compiled_mode(value, dtype, device, is_const, no_cache, name):
    if skip_tracing:
        unset_tracing()
        ret = RawTensor(value, dtype, device, False, name)
        set_tracing()
        return ret
    return active_trace._apply_const(value, dtype, device)


def apply_with_tracing(op: OpDef, *args: RawTensor):
    if active_trace._graph:
        # if member _graph exits, then is_compiled
        return apply_compiled_mode(op, *args)
    if hasattr(op, "scope"):
        op.scope = AutoNaming.get_scope()
    if active_trace._symbolic:
        outputs = apply_symbolic_mode(op, *args)
    else:
        unset_tracing()
        outputs = apply(op, *args)
        set_tracing()

    active_trace._record_op(op, args, outputs)
    return list(outputs)


def apply_const_with_tracing(value, dtype, device, is_const, no_cache, name):
    if active_trace._graph:
        return apply_const_compiled_mode(value, dtype, device, is_const, no_cache, name)
    if active_trace._symbolic:
        outputs = apply_const_symbolic_mode(value, dtype, device, name)
    else:
        unset_tracing()
        outputs = RawTensor(value, dtype, device, False, name)
        if np.array(value).ndim == 0:
            setscalar(outputs)
        outputs = (outputs,)
        set_tracing()
    active_trace._record_const(outputs)
    return list(outputs)
