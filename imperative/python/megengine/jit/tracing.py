# -*- coding: utf-8 -*-
import collections
import contextlib
import functools
import itertools
import json
import os
import pickle
import re
import struct
import sys
from typing import Any

import cv2
import numpy as np

from .. import tensor
from ..core import _imperative_rt as rt
from ..core._imperative_rt import GraphProfiler, GraphProfiler2, SerializationMetadata
from ..core._imperative_rt.core2 import Tensor as RawTensor
from ..core._imperative_rt.core2 import Trace, TraceError, name_tensor  # skip_tracing,
from ..core._imperative_rt.graph import _set_priority_to_id
from ..core._imperative_rt.ops import (
    AssertEqual,
    CollectiveComm,
    ExternOpr,
    RemoteRecv,
    RemoteSend,
    set_jit_enabled,
)
from ..core._trace_option import set_symbolic_shape
from ..core.tensor import megbrain_graph as G
from ..logger import get_logger
from ..utils import comp_graph_tools as cgtools
from ..utils.naming import AutoNaming
from ..utils.profiler import is_profiling
from .dtr_config import DTRConfig
from .graph_opt_config import GraphOptimizationConfig
from .sublinear_memory_config import SublinearMemoryConfig

logger = get_logger(__name__)


def _input_node_use_static_shape():
    return os.environ.get("MEGENGINE_INPUT_NODE_USE_STATIC_SHAPE") is not None


active_trace = None
skip_tracing = False


def is_tracing():
    if active_trace is None:
        return False
    else:
        return not skip_tracing


@contextlib.contextmanager
def exclude_from_trace():
    global skip_tracing
    if skip_tracing or (active_trace is None):
        yield
        return
    try:
        skip_tracing = True
        if active_trace is not None:
            active_trace._begin_excluded_region()
        yield
        if active_trace is not None:
            active_trace._end_excluded_region()
    finally:
        skip_tracing = False


def array_comparator(lhs, rhs):
    return np.all(lhs == rhs)


class trace:
    """Wraps a callable and provide:

    * tracing via :meth:`.trace` and :meth:`.dump`
    * accelerated evalutaion via :meth:`.__call__`

    Args:
        function: the function will be traced.
        symbolic: whether to apply symbolic execution for tracing. Default: False
        capture_as_const: capture global vars or closures as const value. Default: False
        record_only: if True, won't run even if call the function. Default: False
        sublinear_memory_config: configuration for sublinear memory optimization.
            If not None, it enables sublinear memory optimization with given setting.
        profiling: whether to profile compiled trace. Default: False
        opt_level: optimization level for compiling trace. Default: 2
        graph_opt_config: configuration for graph optimization. Default: None
        symbolic_shape: whether to use symbolic shape for tracing. Default: True
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
        record_only=False,
        sublinear_memory_config: SublinearMemoryConfig = None,
        dtr_config: DTRConfig = None,
        profiling: bool = False,
        opt_level: int = 2,
        graph_opt_config: GraphOptimizationConfig = None,
        symbolic_shape: bool = True,
    ):
        self.__wrapped__ = function
        self._capture_as_const = capture_as_const or record_only
        self._arg_bindings = None
        self._kwarg_bindings = None
        self._output_bindings = None
        self._symbolic_shape = symbolic_shape
        self._graph_options = {
            "no_force_inplace": True,
            "graph_opt_level": opt_level,
            "seq_opt.enable_seq_comp_node_opt": False,
        }

        # prevent cyclic reference
        graph_options = self._graph_options
        if dtr_config is not None:
            graph_options["enable_dtr_memory_opt"] = True
            graph_options[
                "dtr_config.eviction_threshold"
            ] = dtr_config.eviction_threshold
            graph_options[
                "dtr_config.evictee_minimum_size"
            ] = dtr_config.evictee_minimum_size
            graph_options[
                "dtr_config.recomp_memory_factor"
            ] = dtr_config.recomp_memory_factor
            graph_options[
                "dtr_config.recomp_time_factor"
            ] = dtr_config.recomp_time_factor
        if graph_opt_config is not None:
            mapping = {None: 0, False: 1, True: 2}
            graph_options["graph_opt.jit_config.fuse_dimshuffle"] = mapping[
                graph_opt_config.jit_fuse_dimshuffle
            ]
            graph_options["graph_opt.jit_config.fuse_reduce"] = mapping[
                graph_opt_config.jit_fuse_reduce
            ]
        if sublinear_memory_config is not None:
            graph_options["enable_sublinear_memory_opt"] = True
            graph_options[
                "sublinear_mem_config.lb_memory_mb"
            ] = sublinear_memory_config.lb_memory_mb
            graph_options[
                "sublinear_mem_config.genetic_nr_iter"
            ] = sublinear_memory_config.genetic_nr_iter
            graph_options[
                "sublinear_mem_config.genetic_pool_size"
            ] = sublinear_memory_config.genetic_pool_size
            graph_options[
                "sublinear_mem_config.thresh_nr_try"
            ] = sublinear_memory_config.thresh_nr_try
            graph_options[
                "sublinear_mem_config.num_worker"
            ] = sublinear_memory_config.num_worker
        if int(os.getenv("MEGENGINE_INPLACE_UPDATE", "0")):
            graph_options["var_sanity_check_first_run"] = False

        def apply_options(options):
            for k, v in graph_options.items():
                words = k.split(".")
                suboptions = options
                for word in words[:-1]:
                    suboptions = getattr(suboptions, word)
                setattr(suboptions, words[-1], v)

        self._trace = Trace()
        self._trace.symbolic = symbolic or record_only
        self._trace.capture_as_const = capture_as_const or record_only
        self._trace.no_exec = record_only
        self._trace.options_visitor = apply_options
        self._trace.profile = profiling
        self._trace.array_comparator = array_comparator
        self._trace.record_input_shapes = _input_node_use_static_shape()

    def __call__(self, *args, **kwargs):
        global active_trace
        symbolic_shape = None
        outputs = None
        try:
            active_trace = self
            self._trace.enter()
            if self._capture_as_const:
                self._process_inputs(*args, **kwargs)
            symbolic_shape = set_symbolic_shape(self._symbolic_shape)
            outputs = self.__wrapped__(*args, **kwargs)
        finally:
            handling_exc = sys.exc_info() != (None,) * 3
            active_trace = None
            if symbolic_shape is not None:
                symbolic_shape = set_symbolic_shape(symbolic_shape)
                assert symbolic_shape == self._symbolic_shape
            if self._capture_as_const and (outputs is not None):
                self._process_outputs(outputs)
            try:
                # may raise TraceError
                self._trace.exit()
            except TraceError:
                if not handling_exc:
                    raise
        return outputs

    def _process_inputs(self, *args, **kwargs):
        for i, arg in enumerate(args):
            name_tensor("arg_{}".format(i), arg)

        # TODO: mark kwargs in order
        for k, kwarg in kwargs.items():
            if isinstance(kwarg, RawTensor):
                name_tensor("kwarg_{}".format(k), kwarg)

        if self._arg_bindings is None:
            self._arg_bindings = [
                ("arg_{}".format(i), arg._tuple_shape) for i, arg in enumerate(args)
            ]

        if self._kwarg_bindings is None:
            self._kwarg_bindings = {
                "kwarg_{}".format(k): (k, kwarg._tuple_shape)
                for k, kwarg in kwargs.items()
                if isinstance(kwarg, RawTensor)
            }

    def _process_outputs(self, outputs):
        if isinstance(outputs, RawTensor):
            outputs = [outputs]
        if isinstance(outputs, collections.abc.Mapping):
            output_names, outputs = zip(*sorted(outputs.items()))
        else:
            # output_names = ["output_{}".format(i) for i in range(len(outputs))]
            output_names = None
        self._output_names = output_names
        for i, output in enumerate(outputs):
            name_tensor("output_{}".format(i), output)
        if self._output_bindings is None:
            self._output_bindings = ["output_{}".format(i) for i in range(len(outputs))]

    def _begin_excluded_region(self):
        self._trace.begin_excluded_region()

    def _end_excluded_region(self):
        self._trace.end_excluded_region()

    def _make_feed(
        self,
        graph,
        outputs,
        input_data,
        repeat,
        silent,
        no_assert,
        maxerr,
        resize_input,
        input_transform,
    ):
        def auto_reformat_image(path, data, dst_shape):
            """reformat image to target shape

            :param data: image data as numpy array
            :param dst_shape: target shape
            """
            dim3_format = False  # required input format does not contain batch
            hwc_format = False  # required input format is NHWC

            if not dst_shape:  # input tensor shape is not predefined
                if len(data.shape) == 2:
                    chl = 1
                    h = data.shape[0]
                    w = data.shape[1]
                else:
                    assert (
                        len(data.shape) == 3
                    ), "Input image must be of dimension 2 or 3"
                    h, w, chl = data.shape
                dst_shape = (1, chl, h, w)

            if len(dst_shape) == 3:
                dst_shape = (1,) + dst_shape
                dim3_format = True

            assert len(dst_shape) == 4, "bad dst_shape: {}".format(dst_shape)
            chl = dst_shape[1]
            if chl in [1, 3]:
                n, c, h, w = dst_shape
                dst_shape = (n, h, w, c)
            else:
                chl = dst_shape[3]
                assert chl in [
                    1,
                    3,
                ], "can not infer input format from shape: {}".format(dst_shape)
                hwc_format = True

            # dst_shape has now been normalized to NHWC format

            if resize_input:
                h, w = dst_shape[1:3]
                data = cv2.resize(data, (w, h))
                logger.info("input {} resized to {}".format(path, data.shape))

            if chl == 1:
                data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
                data = data[:, :, np.newaxis]

            assert data.ndim == 3
            data = data[np.newaxis]
            # data normalized to NHWC format

            if not hwc_format:
                data = np.transpose(data, (0, 3, 1, 2))

            if dim3_format:
                data = np.squeeze(data, 0)

            return data

        def read_input_data(dst_shape, dtype, path):
            def check_shape_equal(dst_shape, data_shape):
                if len(dst_shape):
                    assert len(data_shape) == len(
                        dst_shape
                    ), "input/data shapes mismatch: {} vs {}".format(
                        dst_shape, data_shape
                    )

                    if data_shape[1:] != dst_shape[1:]:
                        logger.warning(
                            "dst_shape is {}; data_shape is {}".format(
                                dst_shape, data_shape
                            )
                        )

            if path.startswith("#"):
                assert not resize_input
                assert not input_transform
                spec = path
                m = re.match(
                    r"^#rand\(([-0-9.]*)\s*,\s*([-0-9.]*)\s*(,[^\)]+)?\)$", spec
                )
                assert m, "bad spec {}".format(spec)

                rng_min = float(m.group(1))
                rng_max = float(m.group(2))
                if m.group(3):
                    shape_str = m.group(3)
                    try:
                        shape = shape_str[1:].split(",")
                        if shape[-1].strip() == "...":
                            shape = shape[:-1]
                            shape.extend(list(dst_shape[len(shape) :]))
                        data_shape = tuple(map(int, shape))
                    except ValueError as e:
                        raise ValueError("bad spec {}: {}".format(spec, e.args))
                else:
                    data_shape = dst_shape

                check_shape_equal(dst_shape, data_shape)
                return np.random.uniform(rng_min, rng_max, data_shape).astype(dtype)

            # try to load image
            data = cv2.imread(path, cv2.IMREAD_COLOR)
            if data is None:
                assert not resize_input
                data = np.load(path)
                assert isinstance(data, np.ndarray)
            else:
                # load image succeeds, so we expect input format is image format
                data = auto_reformat_image(path, data, dst_shape)

            data = np.repeat(data, repeat, axis=0)
            if repeat > 1:
                logger.info(
                    "repeat input for {} times, data shape is {}".format(
                        repeat, data.shape
                    )
                )

            check_shape_equal(dst_shape, data.shape)

            if input_transform:
                data = eval(input_transform, {"data": data, "np": np})

            return data

        def gen_one_testcase(inputs, spec):
            paths = spec.split(";")
            if len(paths) != len(inputs):
                if len(paths) == 1 and paths[0].startswith("#"):
                    paths = ["{}:{}".format(name, paths[0]) for name in inputs.keys()]
            assert len(paths) == len(
                inputs
            ), "required inputs: {}; data paths: {}".format(inputs.keys(), paths)
            if len(paths) == 1 and ":" not in paths[0]:
                paths[0] = next(iter(inputs.keys())) + ":" + paths[0]

            ret = {}
            for path in paths:
                var, path = path.split(":")
                ret[var] = read_input_data(inputs[var].shape, inputs[var].dtype, path)
            return ret

        inputs = cgtools.get_dep_vars(outputs, "Host2DeviceCopy")
        inputs = {i.name: i for i in inputs}

        if not no_assert:

            replace_varmap = {}
            inp_map = {}
            # replace var use InputNode
            for name, var in inputs.items():
                inp = G.InputNode(
                    device="xpux", dtype=var.dtype, shape=var.shape, graph=graph
                )
                replace_varmap[var] = inp.outputs[0]._node
                inp_map[name] = inp

            new = cgtools.replace_vars(outputs, replace_varmap)
            if isinstance(new, rt.VarNode):
                new = list(new)

            output_nodes = [G.OutputNode(var) for var in new]
            func = graph.compile(*[node.outputs[0]._node for node in output_nodes])

            def make_dev_tensor(value, dtype=None, device=None):
                return tensor(value, dtype=dtype, device=device)._dev_tensor()

            def calculate(*args, **kwargs):
                output_val = []
                # set inputs value
                for name, var in inputs.items():
                    val = kwargs.pop(name, None)
                    assert val is not None, "miss input name{}".format(name)
                    dev_tensor = make_dev_tensor(val, dtype=var.dtype, device="xpux")
                    inp_map[name].set_value(dev_tensor)

                func.execute()

                for res in output_nodes:
                    output_val.append(res.get_value().numpy())
                return output_val

            def expect_name(var):
                return "{}:expect".format(var.name)

        testcases = []

        np.set_printoptions(precision=2, threshold=4, suppress=True)

        data_list = []
        for item in input_data:
            if item.startswith("@"):
                with open(item[1:], "r") as f:
                    data_list.extend(
                        [line.rstrip() for line in f if line.rstrip() != ""]
                    )
            else:
                data_list.append(item)

        for inp_spec in data_list:
            cur_testcase = gen_one_testcase(inputs, inp_spec)
            assert len(cur_testcase) == len(
                inputs
            ), "required inputs: {}; given data: {}".format(
                inputs.keys(), cur_testcase.keys()
            )

            if not no_assert:
                outputs_get = calculate(**cur_testcase)
                for var, val in zip(outputs, outputs_get):
                    cur_testcase[expect_name(var)] = val
                    logger.info(
                        "generate test groundtruth: var={} shape={} range=({}, {})"
                        " mean={} var={}".format(
                            var,
                            val.shape,
                            val.min(),
                            val.max(),
                            np.mean(val),
                            np.var(val),
                        )
                    )
            testcases.append(cur_testcase)
            logger.info(
                "add testcase: \n {}".format(
                    "\n ".join(
                        "{}: shape={} dtype={} range=({:.2f},{:.2f}) "
                        "mean={:.2f} sd={:.2f}".format(
                            k, v.shape, v.dtype, v.min(), v.max(), np.mean(v), np.std(v)
                        )
                        for k, v in sorted(cur_testcase.items())
                    )
                )
            )

        if not no_assert:

            def expect_shp(var):
                ret = var.shape
                if ret:
                    return ret
                return testcases[0][expect_name(var)].shape

            def assert_equal(expect, real, **kwargs):
                op = AssertEqual(**kwargs)
                (res,) = G.apply_normal_varnode(op, expect, real)
                return res._node

            verbose = not silent

            outputs_new = []
            for i in outputs:
                device = rt.CompNode("xpux")
                dtype = i.dtype
                name = expect_name(i)
                shape = expect_shp(i)
                # make expect output as one input of model.
                expect_get = rt.make_h2d(graph, device, dtype, shape, name)
                # insert assert opr to check expect and real.
                outputs_new.append(
                    assert_equal(expect_get, i, verbose=verbose, maxerr=maxerr,)
                )
                inputs[expect_name(i)] = expect_get
            outputs = outputs_new

        return {"outputs": outputs, "testcases": testcases}

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
        no_change_graph: bool = False,
        strip_info_file=None,
        append_json=False,
        optimize_for_inference=True,
        user_info: Any = None,
        enable_metadata: bool = True,
        input_data=None,
        repeat=1,
        silent=False,
        no_assert=False,
        maxerr=1e-4,
        resize_input=False,
        input_transform=None,
        dump_format: str = None,
        model_version: int = 2,
        **kwargs
    ):
        r"""Serializes trace to file system.

        Args:
            file: output file, could be file object or filename.
            arg_names: names of the input tensors in the traced function.
            output_names: names of the output tensors in the traced function,
                use the default name if not specified.
            append: whether output is appended to ``file``.
                Only works when ``file`` is str.
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
            optimize_for_inference: enbale optmizations,
                will skip all optimize options if this is False. Default: True
            user_info: any type object, which will be pickled to bytes.
            enable_metadata: whether to save metadata into output file.
            input_data: input test data and current network output would be used as groundtruth.
                The format is "var0:file0;var1:file1..." to specify data files for input vars.
                It can also be "#rand(min,max,shape...)" for generating random input data, for
                example, "#rand(0,255)", "#rand(0,255,1,3,224,224)" or "#rand(0, 255, 1, ...)"
                where `...` means the remaining part of the original shape. If the shape is not
                specified, the shape of corresponding input tensors in the network will be used.
                If there is only one input var, its name can be omitted. Each data file can either
                be an image which can be loaded by opencv, or a pickled numpy.ndarray. This option
                can be given multiple times to add multiple testcases. If you start the data
                with the letter @, the rest should be a filename, and each line in the file should
                be a single datum in the format described above. *NOTE* If `input_data` is not None,
                you can only use load-and-run to run the output file.
            repeat: how many times the input image is repeated. Useful when running benchmark for
                batch size other than one. Have no effect on randomly generated input data.
            silent: whether set verbose to False in assert_equal opr.
            no_assert: whether insert assert_equal opr to check result; this option is useful for
                benchmarking.
            maxerr: max error for assert_equal check during runtime.
            resize_input: whether resize input image to fit input var shape.
            input_transform: a python expression to transform the input data.
                Example: data / np.std(data)
            dump_format: using different dump formats. the open source MegEngine
                defaults to the FBS_V2 format, there are two format FBS_V2 and FBS to choose,
                internal MegEngine have an other choice of internal proprietary formats
            model_version: the model version of FBS_V2, begin with version 2, this
                works only when dump format is FBS_V2.


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
          inference)
        * enable_fuse_preprocess: whether to fuse astype\pad_channel\dimshuffle and
          etc opr
        """
        if not self._capture_as_const:
            raise ValueError(
                "you must specify capture_as_const=True at __init__ to use dump"
            )
        if self._output_names and output_names:
            raise TypeError(
                "cannot specify output_names when output is already in dict format"
            )
        if output_names and isinstance(output_names, str):
            output_names = (output_names,)
        if output_names and len(output_names) != len(self._output_bindings):
            raise ValueError(
                "wrong number of output_names, should be {} values".format(
                    len(self._output_bindings)
                )
            )
        prefer_input_names = arg_names is not None
        if arg_names is None:
            arg_names = ["arg_%d" % i for i in range(len(self._arg_bindings))]
        if isinstance(arg_names, str):
            arg_names = (arg_names,)
        arg_names = [arg_name if arg_name is not None else "" for arg_name in arg_names]
        if arg_names and len(arg_names) != len(self._arg_bindings):
            raise ValueError(
                "wrong number of arg_names, should be {} values".format(
                    len(self._arg_bindings)
                )
            )
        output_names = output_names or self._output_names

        if output_names is None:
            output_names = [""] * len(self._output_bindings)
            # output_names = ["output_{}".format(i) for i in range(len(self._output_bindings))]

        input_bindings = []

        def normalize_shape(shape):
            return (1,) if shape == () else shape

        for arg_name, (arg_id, arg_shape) in zip(arg_names, self._arg_bindings):
            input_bindings.append((arg_id, arg_name, normalize_shape(arg_shape)))

        for kwarg_id, (kwarg_name, kwarg_shape) in self._kwarg_bindings.items():
            input_bindings.append((kwarg_id, kwarg_name, normalize_shape(kwarg_shape)))

        graph = G.Graph()

        jit_enabled = set_jit_enabled(False)
        dest_vars = self._trace.dump(
            graph,
            input_bindings,
            [*zip(self._output_bindings, output_names)],
            prefer_input_names,
        )
        set_jit_enabled(jit_enabled)

        # dest_vars = [i._node for i in dest_vars]

        if input_data is not None:
            feeds = self._make_feed(
                graph,
                dest_vars,
                input_data,
                repeat,
                silent,
                no_assert,
                maxerr,
                resize_input,
                input_transform,
            )
            assert (
                isinstance(feeds, dict) and feeds["testcases"]
            ), "testcases can not be empty"
            dest_vars = feeds["outputs"]

        if optimize_for_inference:
            dest_vars, optimize_options = G.optimize_for_inference(dest_vars, **kwargs)
            dest_vars = [i._node for i in dest_vars]

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

        if keep_opr_priority:
            _set_priority_to_id(dest_vars)

        if input_data is not None:
            file.write(b"mgbtest0")
            file.write(struct.pack("I", len(feeds["testcases"])))
        dump_content, dump_info = G.dump_graph(
            dest_vars,
            keep_var_name=keep_var_name,
            keep_opr_name=keep_opr_name,
            keep_param_name=keep_param_name,
            keep_opr_priority=keep_opr_priority,
            no_change_graph=no_change_graph,
            strip_info_file=strip_info_file,
            append_json=append_json,
            metadata=metadata,
            dump_format=dump_format,
            model_version=model_version,
        )
        file.write(dump_content)

        if input_data is not None:
            inputs = cgtools.get_dep_vars(dest_vars, "Host2DeviceCopy")
            inputs = sorted((i.name, i.dtype) for i in inputs)

            def make_dev_tensor(value, dtype=None, device=None):
                return tensor(value, dtype=dtype, device=device)._dev_tensor()

            for testcase in feeds["testcases"]:
                assert isinstance(testcase, dict)
                cg = G.Graph()
                output_mgbvars = []
                for name, dtype in inputs:
                    output_mgbvars.append(
                        cg.make_const(
                            make_dev_tensor(
                                testcase.pop(name), dtype=dtype, device="cpux"
                            )
                        )
                    )
                assert not testcase, "extra inputs provided in testcase: {}".format(
                    testcase.keys()
                )
                dump_content, _ = G.dump_graph(
                    output_mgbvars, strip_info_file=strip_info_file, append_json=True,
                )
                file.write(dump_content)

        return dump_info

    def get_profile(self):
        return json.loads(self._trace.get_profile())
