import argparse
import logging
import time
from collections import OrderedDict

import numpy as np

import megengine as mge
from megengine.core.ops import custom
from megengine.core.tensor import megbrain_graph as G
from megengine.device import get_device_count, set_default_device
from megengine.functional.debug_param import set_execution_strategy
from megengine.logger import enable_debug_log, get_logger, set_log_file
from megengine.utils import comp_graph_tools as tools

logger = get_logger(__name__)


def make_data_given_desc(args, inputs, shape0_multiply=1):
    if args.load_input_data:
        logger.info("load data from {}".format(args.load_input_data))
        data = mge.load(args.load_input_data)
        data_names = [inp.name for inp in inputs]

        if isinstance(data, np.ndarray):
            assert len(data_names) == 1, (
                "data is given as a single numpy array, so there should be "
                "exactly one input in the graph; got: {}".format(data_names)
            )
            data = {data_names[0]: data}

        assert isinstance(data, dict)
        for v in data.values():
            assert isinstance(
                v, np.ndarray
            ), "data should provide ndarray; got {} instead".format(v)

        if args.batchsize:
            for k, v in list(data.items()):
                assert (
                    args.batchsize % v.shape[0] == 0
                ), "current batch size must divide given batch size: {} {}".format(
                    args.batchsize, v.shape[0]
                )
                data[k] = np.repeat(v, args.batchsize // v.shape[0], axis=0)
        return data

    def iter_inpdesc(desc):
        if not desc:
            return
        for pair in desc.split(";"):
            name, value = pair.split(":")
            if name not in data_shapes:
                logger.warning("rng name {} not in data provider".format(name))
            yield name, value

    rng = np.random.RandomState(args.seed)

    data_shapes = OrderedDict((inp.name, list(inp.shape)) for inp in inputs)
    data_dtypes = OrderedDict((inp.name, inp.dtype) for inp in inputs)

    for name, shape in iter_inpdesc(args.input_desc):
        data_shapes[name] = list(map(int, shape.split(",")))

    if args.batchsize:
        for i in data_shapes.values():
            i[0] = args.batchsize

    data_rngs = dict(iter_inpdesc(args.rng))

    result = OrderedDict()
    for name, shape in data_shapes.items():
        shape[0] *= shape0_multiply
        rng_expr = data_rngs.get(name)
        if rng_expr:
            value = eval("rng.{}".format(rng_expr).format(shape), {"rng": rng})
        else:
            value = rng.uniform(size=shape)

        value = np.ascontiguousarray(value, dtype=data_dtypes[name])
        assert value.shape == tuple(shape)
        result[name] = value

    return result


def get_execution_strategy(args):
    if not args.fast_run:
        logger.warning("--fast-run not enabled; execution may be slow")
        strategy = "HEURISTIC"
    else:
        logger.warning("--fast-run enabled; compile may be slow")
        strategy = "PROFILE"
    if args.reproducible:
        strategy += "_REPRODUCIBLE"
    return strategy


def get_opt_kwargs(args):
    args_list = [
        "enable_io16xc32",
        "enable_ioc16",
        "enable_hwcd4",
        "enable_nchw4",
        "enable_nchw88",
        "enable_nchw44",
        "enable_nchw44_dot",
        "enable_nchw32",
        "enable_chwn4",
        "enable_fuse_conv_bias_nonlinearity",
        "enable_fuse_conv_bias_with_z",
    ]
    kwargs = {}
    for k in args_list:
        if getattr(args, k):
            kwargs[k] = True
    return kwargs


def run_model(args, graph, inputs, outputs, data):
    # must use level0 to avoid unintended opr modification
    graph.options.graph_opt_level = 0

    if args.weight_preprocess:
        graph.enable_weight_preprocess()

    logger.info("input tensors: ")
    for k, v in data.items():
        logger.info("  {}: {}".format(k, v.shape))

    G.modify_opr_algo_strategy_inplace(outputs, get_execution_strategy(args))

    if args.optimize_for_inference:
        opt_kwargs = get_opt_kwargs(args)
        outputs = G.optimize_for_inference(outputs, **opt_kwargs)

    # embed inputs must be on the last, to avoid const fold
    if args.embed_input:
        outputs, inp_dict = tools.embed_inputs(outputs, data.values(), inputs=inputs)
    else:
        outputs, inp_dict = tools.convert_inputs(outputs, inputs=inputs)

    if args.dump_cpp_model:
        dump_content, _ = G.dump_graph(outputs, keep_var_name=2)
        with open(args.dump_cpp_model, "wb") as file:
            file.write(dump_content)
        logger.info("C++ model written to {}".format(args.dump_cpp_model))

    outputs, output_dict = tools.convert_outputs(outputs)

    if args.profile:
        profiler = tools.GraphProfiler(graph)

    func = graph.compile(outputs)

    if args.get_static_mem_info:
        func.get_static_memory_alloc_info(args.get_static_mem_info)

    def run():
        if not args.embed_input:
            for key in inp_dict:
                inp_dict[key].set_value(mge.Tensor(data[key])._dev_tensor())
        func.execute()
        func.wait()
        return [oup_node.get_value().numpy() for oup_node in output_dict.values()]

    for i in range(args.warm_up):
        logger.info("warming up {}".format(i))
        run()

    total_time = 0

    for i in range(args.iter):
        logger.info("iter {}".format(i))
        start_time = time.time()
        retval = run()
        cur_time = time.time() - start_time
        total_time += cur_time

        avg_speed = (i + 1) / total_time
        if "data" in data:
            avg_speed *= data["data"].shape[0]
            avg_speed_txt = "{:.3f}sample/s".format(avg_speed)
        else:
            avg_speed_txt = "{:.3f}batch/s".format(avg_speed)

        msg = (
            "iter {}: duration={:.4f}({:.4f})s average={:.4f}s "
            "avg_speed={} time={:.4f}s"
        ).format(
            i,
            cur_time,
            func.get_prev_exec_time(),
            total_time / (i + 1),
            avg_speed_txt,
            total_time,
        )
        if args.calc_output_rms:
            rms = []
            for v in retval:
                rms.append("{:.3g}".format(float(((v ** 2).mean()) ** 0.5)))
            msg += " output_rms=[{}]".format(", ".join(rms))
        if logger.level > logging.INFO:
            print(msg)
        else:
            logger.info(msg)

    if args.focused_nvprof:
        if get_device_count("gpu") < 1:
            logger.warning(
                "No cuda device detected. ``focused_nvprof`` will be ignored."
            )
        else:
            try:
                import pycuda.driver as D

                D.start_profiler()
                func.execute()
                func.wait()
                D.stop_profiler()
            except ImportError:
                logger.error("`focused_nvprof need pycuda`", exc_info=True)

    if args.profile:
        with open(args.profile, "w") as fout:
            fout.write(profiler.get())

    return avg_speed


def main():
    parser = argparse.ArgumentParser(
        description="load a network and run inference on random data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("net")
    parser.add_argument(
        "--device", "-d", help="set defult device, like 'gpux' or 'cpux'"
    )
    parser.add_argument(
        "--calc-output-rms",
        action="store_true",
        help="compute RMS of outputs; useful for comparing computing results",
    )
    parser.add_argument(
        "--output-name",
        nargs="*",
        help="Specify output name. This option can be"
        " specified multiple times. We will look for opr/var"
        " in the graph",
    )
    parser.add_argument(
        "--load-input-data",
        help="load input data from pickle file; it should be"
        " a numpy array or a dict of numpy array",
    )
    parser.add_argument("--profile", help="profiler output file")
    parser.add_argument(
        "--fast-run",
        action="store_true",
        help="enable fast running by profiling conv algorithms during compiling.",
    )
    parser.add_argument(
        "--reproducible", action="store_true", help="use reproducible kernels"
    )
    parser.add_argument(
        "--input-desc",
        help="specifiy input names and shapes manually in"
        " format: <name>:<shape>[;<name>:<shape>, ...], where"
        " name is a string and shape is a comma separated"
        ' string. e.g., "data:128,1,28,28,label:128".'
        " different input tensor are separated by semicolon.",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        help="change batchsize; the first dimension of each"
        " input is assumed to be batch size",
    )
    parser.add_argument(
        "--warm-up",
        type=int,
        default=0,
        help="times of warm up model before do timing " " for better estimation",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="verbose output, logging in debug mode",
    )
    parser.add_argument(
        "--iter", type=int, default=1, help="number of iters to run the model"
    )
    parser.add_argument("--log", help="give a file path to duplicate log to")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for random number generator for input data",
    )
    parser.add_argument(
        "--rng",
        help="special RNG options to generate input data in"
        " format: <name>:func[;<name>:func, ...] where name is"
        " a string and func is a python expression containing"
        ' "{}" for the size param, e.g. '
        ' "label:randint(low=0,high=1000,size={})"',
    )
    parser.add_argument(
        "--focused-nvprof",
        action="store_true",
        help="only profile last iter for `nvprof --profile-from-start off`",
    )
    parser.add_argument(
        "--optimize-for-inference",
        action="store_true",
        help="optimize model for inference",
    )
    parser.add_argument(
        "--enable-io16xc32",
        action="store_true",
        help="transform the mode to float16 io float32 compute",
    )
    parser.add_argument(
        "--enable-ioc16",
        action="store_true",
        help="transform the dtype of the model to float16 io and compute",
    )
    parser.add_argument(
        "--enable-hwcd4",
        action="store_true",
        help="transform the model format from NCHW to NHWCD4 for inference",
    )
    parser.add_argument(
        "--enable-nchw4",
        action="store_true",
        help="transform the model format from NCHW to NCHW4 for inference",
    )
    parser.add_argument(
        "--enable-nchw88",
        action="store_true",
        help="transform the model format from NCHW to NCHW88 for inference",
    )
    parser.add_argument(
        "--enable-nchw44",
        action="store_true",
        help="transform the model format from NCHW to NCHW44 for inference",
    )
    parser.add_argument(
        "--enable-nchw44-dot",
        action="store_true",
        help="transform the model format from NCHW to NCHW44_DOT "
        "for optimizing armv8.2 dot in inference",
    )
    parser.add_argument(
        "--enable-chwn4",
        action="store_true",
        help="transform the model format to CHWN4 "
        "for inference, mainly used for nvidia tensorcore",
    )
    parser.add_argument(
        "--enable-nchw32",
        action="store_true",
        help="transform the model format from NCHW4 to NCHW32 "
        "for inference on nvidia TensoCore",
    )
    parser.add_argument(
        "--enable-fuse-conv-bias-nonlinearity",
        action="store_true",
        help="fuse convolution bias and nonlinearity opr to a "
        "conv_bias opr and compute",
    )
    parser.add_argument(
        "--enable-fuse-conv-bias-with-z",
        action="store_true",
        help="fuse conv_bias with z input for inference on "
        "nvidia GPU (this optimization pass will result in mismatch "
        "of the precision of output of training and inference)",
    )
    parser.add_argument(
        "--dump-cpp-model",
        help="write a C++ model that can be loaded by "
        "megbrain/lite/load_and_run; "
        "this implies --embed-input",
    )
    parser.add_argument(
        "--embed-input",
        action="store_true",
        help="embed input data as SharedDeviceTensor in model, "
        "to remove memory copy for inputs",
    )
    parser.add_argument(
        "--get-static-mem-info",
        type=str,
        help="Record the static graph's static memory info.",
    )
    parser.add_argument(
        "--custom-op-lib", type=str, help="path of the custom op",
    )
    parser.add_argument(
        "--weight-preprocess",
        action="store_true",
        help="Execute operators with weight preprocess, which can"
        "optimize the operator execution time with algo of winograd,"
        "im2col ,etc.,but it may consume more memory.",
    )

    args = parser.parse_args()

    if args.verbose:
        enable_debug_log()
    if args.log:
        set_log_file(args.log)

    if args.device:
        set_default_device(args.device)

    if args.dump_cpp_model:
        args.embed_input = True
    if args.custom_op_lib is not None:
        custom.load(args.custom_op_lib)

    logger.info("loading model ...")
    ret = G.load_graph(args.net)
    graph, output_vars = ret.graph, ret.output_vars_list
    input_vars = tools.get_dep_vars(output_vars, "Host2DeviceCopy")

    if args.output_name is not None:
        output_vars = tools.find_vars_by_name(output_vars, args.output_name)

    data = make_data_given_desc(args, input_vars)

    run_model(args, graph, input_vars, output_vars, data)


if __name__ == "__main__":
    main()
