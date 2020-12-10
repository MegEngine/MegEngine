# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse
import os
import re
import struct

import cv2
import numpy as np

import megengine as mge
import megengine.core._imperative_rt as rt
import megengine.core.tensor.megbrain_graph as G
from megengine.utils import comp_graph_tools as cgtools
from megengine.core.ops import builtin
from megengine.core.tensor.core import apply
from megengine.core.tensor.megbrain_graph import VarNode
from megengine.core.tensor.raw_tensor import as_raw_tensor

logger = mge.get_logger(__name__)


def auto_reformat_image(args, path, data, dst_shape):
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
            assert len(data.shape) == 3, "Input image must be of dimension 2 or 3"
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
        assert chl in [1, 3], "can not infer input format from shape: {}".format(
            dst_shape
        )
        hwc_format = True

    # dst_shape has now been normalized to NHWC format

    if args.resize_input:
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


def read_input_data(args, dst_shape, dtype, path, repeat):
    def check_shape_equal(dst_shape, data_shape):
        if len(dst_shape):
            assert len(data_shape) == len(
                dst_shape
            ), "input/data shapes mismatch: {} vs {}".format(dst_shape, data_shape)

            if data_shape[1:] != dst_shape[1:]:
                logger.warning(
                    "dst_shape is {}; data_shape is {}".format(dst_shape, data_shape)
                )

    if path.startswith("#"):
        assert not args.resize_input
        assert not args.input_transform
        spec = path
        m = re.match(r"^#rand\(([-0-9.]*)\s*,\s*([-0-9.]*)\s*(,[^\)]+)?\)$", spec)
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
        assert not args.resize_input
        data = np.load(path)
        assert isinstance(data, np.ndarray)
    else:
        # load image succeeds, so we expect input format is image format
        data = auto_reformat_image(args, path, data, dst_shape)

    data = np.repeat(data, repeat, axis=0)
    if repeat > 1:
        logger.info(
            "repeat input for {} times, data shape is {}".format(repeat, data.shape)
        )

    check_shape_equal(dst_shape, data.shape)

    if args.input_transform:
        data = eval(args.input_transform, {"data": data, "np": np})

    return data


def gen_one_testcase(args, inputs, spec):
    paths = spec.split(";")
    if len(paths) != len(inputs):
        if len(paths) == 1 and paths[0].startswith("#"):
            paths = ["{}:{}".format(name, paths[0]) for name in inputs.keys()]
    assert len(paths) == len(inputs), "required inputs: {}; data paths: {}".format(
        inputs.keys(), paths
    )
    if len(paths) == 1 and ":" not in paths[0]:
        paths[0] = next(iter(inputs.keys())) + ":" + paths[0]

    ret = {}
    for path in paths:
        var, path = path.split(":")
        if args.repeat:
            repeat = args.repeat
        else:
            repeat = 1
        ret[var] = read_input_data(
            args, inputs[var].shape, inputs[var].dtype, path, repeat
        )
    return ret


def make_feeds(args):
    cg_rt, _, outputs = G.load_graph(args.input)
    inputs = cgtools.get_dep_vars(outputs, "Host2DeviceCopy")

    inputs = {i.name: i for i in inputs}
    if not args.no_assert:

        replace_varmap = {}
        inp_map = {}
        # replace var use InputNode
        for name, var in inputs.items():
            inp = G.InputNode(
                device="xpux", dtype=var.dtype, shape=var.shape, graph=cg_rt
            )
            replace_varmap[var] = inp.outputs[0]
            inp_map[name] = inp

        new = cgtools.replace_vars(outputs, replace_varmap)
        if isinstance(new, rt.VarNode):
            new = list(new)

        output_nodes = [G.OutputNode(var) for var in new]
        func = cg_rt.compile([node.outputs[0] for node in output_nodes])

        def make_dev_tensor(value, dtype=None, device=None):
            return as_raw_tensor(value, dtype=dtype, device=device)._dev_tensor()

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
    for item in args.data:
        if item.startswith("@"):
            with open(item[1:], "r") as f:
                data_list.extend([line.rstrip() for line in f if line.rstrip() != ""])
        else:
            data_list.append(item)

    for inp_spec in data_list:
        cur_testcase = gen_one_testcase(args, inputs, inp_spec)
        assert len(cur_testcase) == len(
            inputs
        ), "required inputs: {}; given data: {}".format(
            inputs.keys(), cur_testcase.keys()
        )

        if not args.no_assert:
            outputs_get = calculate(**cur_testcase)
            for var, val in zip(outputs, outputs_get):
                cur_testcase[expect_name(var)] = val
                logger.info(
                    "generate test groundtruth: var={} shape={} range=({}, {})"
                    " mean={} var={}".format(
                        var, val.shape, val.min(), val.max(), np.mean(val), np.var(val)
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

    if not args.no_assert:

        def expect_shp(var):
            ret = var.shape
            if ret:
                return ret
            return testcases[0][expect_name(var)].shape

        def assert_equal(expect, real, **kwargs):
            op = builtin.AssertEqual(**kwargs)
            (res,) = apply(op, expect, real)
            return res

        verbose = not args.silent

        outputs_new = []
        for i in outputs:
            device = rt.CompNode("xpux")
            dtype = i.dtype
            name = expect_name(i)
            shape = expect_shp(i)
            # make expect output as one input of model.
            expect_get = rt.make_h2d(cg_rt, device, dtype, shape, name)
            # insert assert opr to check expect and real.
            outputs_new.append(
                assert_equal(
                    G.VarNode(expect_get),
                    G.VarNode(i),
                    verbose=verbose,
                    maxerr=args.maxerr,
                )
            )
            inputs[expect_name(i)] = expect_get
        outputs = outputs_new

    return {"outputs": outputs, "testcases": testcases}


def optimize_for_inference(args, outputs):
    args_map = {
        "enable_io16xc32": "f16_io_f32_comp",
        "enable_ioc16": "f16_io_comp",
        "enable_hwcd4": "use_nhwcd4",
        "enable_nchw4": "use_nchw4",
        "enable_nchw88": "use_nchw88",
        "enable_nchw44": "use_nchw44",
        "enable_nchw44_dot": "use_nchw44_dot",
        "enable_nchw32": "use_nchw32",
        "enable_chwn4": "use_chwn4",
        "enable_fuse_conv_bias_nonlinearity": "fuse_conv_bias_nonlinearity",
        "enable_fuse_conv_bias_with_z": "fuse_conv_bias_with_z",
    }
    kwargs = {}
    for k, v in args_map.items():
        if getattr(args, k):
            assert (
                args.optimize_for_inference
            ), "optimize_for_inference should be set when {} is given".format(k)
            kwargs[v] = True

    if args.optimize_for_inference:
        outputs = [i._node for i in G.optimize_for_inference(outputs, **kwargs)]

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Pack computing graph, input values and expected output "
        "values into one file for checking correctness. README.md gives more "
        "details on the usage",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="MegEngine dumped model file")
    parser.add_argument("-o", "--output", help="output file", required=True)
    parser.add_argument(
        "-d",
        "--data",
        default=[],
        action="append",
        required=True,
        help="Given input test data when input file is a network, "
        "and current network output would be used as groundtruth. "
        "The format is var0:file0;var1:file1... to specify data files for "
        "input vars. It can also be #rand(min,max,shape...) for generating "
        "random input data, for example, #rand(0,255), "
        "#rand(0,255,1,3,224,224) or #rand(0, 255, 1, ...) where `...` means "
        "the remaining part of the original shape. "
        "If the shape is not specified, the shape of "
        "corresponding input tensors in the network will be used. "
        "If there is only one input var, its name can be omitted. "
        "Each data file can either be an image which can be loaded by opencv, "
        "or a pickled numpy.ndarray. "
        "This option can be given multiple times to add multiple testcases. "
        " *NOTE* "
        "If you start the data with the letter @, the rest should be a "
        "filename, and each line in the file should be a single datum in "
        "the format described above. ",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Specify how many times the input image is repeated. "
        "Useful when running benchmark for batch size other than one. "
        "Have no effect on randomly generated input data.",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="set verbose to False in asserti_equal opr",
    )
    parser.add_argument(
        "--optimize-for-inference",
        action="store_false",
        help="enbale optimization for inference",
    )
    parser.add_argument(
        "--no-assert",
        action="store_true",
        help="do not insert assert_equal opr to check result; "
        "this option is useful for benchmarking",
    )
    parser.add_argument(
        "--maxerr",
        type=float,
        default=1e-4,
        help="max error for assert_equal check during runtime",
    )
    parser.add_argument(
        "--resize-input",
        action="store_true",
        help="resize input image to fit input var shape",
    )
    parser.add_argument(
        "--input-transform",
        help="a python expression to transform the input data. "
        "Example: data / np.std(data)",
    )
    parser.add_argument(
        "--discard-var-name",
        action="store_true",
        help="discard variable and param names in the " "generated output",
    )
    parser.add_argument(
        "--output-strip-info", action="store_true", help="output code strip information"
    )
    parser.add_argument(
        "--enable-io16xc32",
        action="store_true",
        help="transform the mode to float16 io float32 compute",
    )
    parser.add_argument(
        "--enable-ioc16",
        action="store_true",
        help="transform the dtype of the model to float16 io " "and compute",
    )
    parser.add_argument(
        "--enable-fuse-conv-bias-nonlinearity",
        action="store_true",
        help="fuse convolution bias and nonlinearity opr to a "
        "conv_bias opr and compute",
    )
    parser.add_argument(
        "--enable-hwcd4",
        action="store_true",
        help="transform the model format from NCHW to NHWCD4 "
        "for inference; you may need to disable CUDA and set "
        "MGB_USE_MEGDNN_DBG=2",
    )
    parser.add_argument(
        "--enable-nchw4",
        action="store_true",
        help="transform the model format from NCHW to NCHW4 " "for inference",
    )
    parser.add_argument(
        "--enable-nchw88",
        action="store_true",
        help="transform the model format from NCHW to NCHW88 " "for inference",
    )
    parser.add_argument(
        "--enable-nchw44",
        action="store_true",
        help="transform the model format from NCHW to NCHW44 " "for inference",
    )
    parser.add_argument(
        "--enable-nchw44-dot",
        action="store_true",
        help="transform the model format from NCHW to NCHW44_DOT "
        "for optimizing armv8.2 dot in inference",
    )
    parser.add_argument(
        "--enable-nchw32",
        action="store_true",
        help="transform the model format from NCHW4 to NCHW32 "
        "for inference on nvidia TensoCore",
    )
    parser.add_argument(
        "--enable-chwn4",
        action="store_true",
        help="transform the model format to CHWN4 "
        "for inference, mainly used for nvidia tensorcore",
    )
    parser.add_argument(
        "--enable-fuse-conv-bias-with-z",
        action="store_true",
        help="fuse conv_bias with z input for inference on "
        "nvidia GPU (this optimization pass will result in mismatch "
        "of the precision of output of training and inference)",
    )
    args = parser.parse_args()

    feeds = make_feeds(args)

    assert isinstance(feeds, dict) and feeds["testcases"], "testcases can not be empty"

    output_mgbvars = feeds["outputs"]
    output_mgbvars = optimize_for_inference(args, output_mgbvars)

    inputs = cgtools.get_dep_vars(output_mgbvars, "Host2DeviceCopy")
    inputs = sorted((i.name, i.dtype) for i in inputs)

    if args.discard_var_name:
        sereg_kwargs = dict(keep_var_name=0, keep_param_name=False)
    else:
        sereg_kwargs = dict(keep_var_name=2, keep_param_name=True)

    strip_info_file = args.output + ".json" if args.output_strip_info else None

    with open(args.output, "wb") as fout:
        fout.write(b"mgbtest0")
        fout.write(struct.pack("I", len(feeds["testcases"])))
        if isinstance(output_mgbvars, dict):
            wrap_output_vars = dict([(i, VarNode(j)) for i, j in output_mgbvars])
        else:
            wrap_output_vars = [VarNode(i) for i in output_mgbvars]
        dump_content, stat = G.dump_graph(
            wrap_output_vars,
            append_json=True,
            strip_info_file=strip_info_file,
            **sereg_kwargs
        )
        fout.write(dump_content)

        logger.info(
            "graph dump sizes: tot_size={:.3f}KiB overhead={:.3f}KiB".format(
                stat.tot_bytes / 1024, (stat.tot_bytes - stat.tensor_value_bytes) / 1024
            )
        )

    def make_dev_tensor(value, dtype=None, device=None):
        return as_raw_tensor(value, dtype=dtype, device=device)._dev_tensor()

    for testcase in feeds["testcases"]:
        assert isinstance(testcase, dict)
        cg = G.Graph()
        output_mgbvars = []
        for name, dtype in inputs:
            output_mgbvars.append(
                cg.make_const(
                    make_dev_tensor(testcase.pop(name), dtype=dtype, device="cpux")
                )
            )
        assert not testcase, "extra inputs provided in testcase: {}".format(
            testcase.keys()
        )
        with open(args.output, "ab") as fout:
            dump_content, _ = G.dump_graph(
                output_mgbvars, strip_info_file=strip_info_file, append_json=True
            )
            fout.write(dump_content)


if __name__ == "__main__":
    main()
