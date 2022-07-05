#! /usr/bin/env python3
import argparse
import logging
import re
from collections import namedtuple

import numpy as np
from tqdm import tqdm

import megengine as mge
from megengine.core.tensor.dtype import is_quantize
from megengine.logger import _imperative_rt_logger, get_logger, set_mgb_log_level
from megengine.utils.module_stats import (
    enable_receptive_field,
    get_activation_stats,
    get_op_stats,
    get_param_stats,
    print_activations_stats,
    print_op_stats,
    print_param_stats,
    print_summary,
    sizeof_fmt,
    sum_activations_stats,
    sum_op_stats,
    sum_param_stats,
)
from megengine.utils.network import Network

logger = get_logger(__name__)


def visualize(
    model_path: str,
    log_path: str,
    input: np.ndarray = None,
    inp_dict: dict = None,
    cal_params: bool = True,
    cal_flops: bool = True,
    cal_activations: bool = True,
    logging_to_stdout: bool = True,
    bar_length_max: int = 20,
):
    r"""Load megengine dumped model and visualize graph structure with tensorboard log files.
    Can also record and print model's statistics like :func:`~.module_stats`

    Args:
      model_path: dir path for megengine dumped model.
      log_path: dir path for tensorboard graph log.
      input: user defined input data for running model and calculating stats,
        alternative with inp_dict, used when the model has only one input.
      inp_dict: input dict for running model and calculating stats, alternative with
        input, used when the model has more than one input.
        When both input and inp_dict are None, a random input will be used.
      cal_params: whether calculate and record params size.
      cal_flops: whether calculate and record op flops.
      cal_activations: whether calculate and record op activations.
      logging_to_stdout: whether print all calculated statistic details.
      bar_length_max: size of bar indicating max flops or parameter size in net stats.
      model_path: str:
      log_path: str:
      input: np.ndarray:
      inp_dict: dict:
      cal_params: bool:
      cal_flops: bool:
      cal_activations: bool:
      logging_to_stdout: bool:
      bar_length_max: int:
    """
    if log_path:
        try:
            from tensorboard.compat.proto.attr_value_pb2 import AttrValue
            from tensorboard.compat.proto.config_pb2 import RunMetadata
            from tensorboard.compat.proto.graph_pb2 import GraphDef
            from tensorboard.compat.proto.node_def_pb2 import NodeDef
            from tensorboard.compat.proto.step_stats_pb2 import (
                AllocatorMemoryUsed,
                DeviceStepStats,
                NodeExecStats,
                StepStats,
            )
            from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
            from tensorboard.compat.proto.versions_pb2 import VersionDef
            from tensorboardX import SummaryWriter
        except ImportError:
            logger.error(
                "TensorBoard and TensorboardX are required for visualize.",
                exc_info=True,
            )
            return

    enable_receptive_field()

    graph = Network.load(model_path)
    graph.reset_batch_size(1)

    has_input = False
    if input is not None or inp_dict is not None:
        has_input = True
        repl_dict = {}
        inp_vars = graph.input_vars
        if inp_dict is not None:
            assert len(inp_dict) == len(
                inp_vars
            ), "Inputs are not sufficient for calculation."
            for v in inp_vars:
                new_input = graph.make_const(inp_dict[v.name], name=v.name)
                repl_dict[v] = new_input
        else:
            assert len(inp_vars) == 1, "The graph needs more than one input."
            inp_var = inp_vars[0]
            repl_dict[inp_var] = graph.make_const(input, name=inp_var.name)
        graph.replace_vars(repl_dict=repl_dict)

    graph._compile()

    def process_name(name):
        # nodes that start with point or contain float const will lead to display bug
        if not re.match(r"^[+-]?\d*\.\d*", name):
            name = name.replace(".", "/")
        return name.encode(encoding="utf-8")

    summary = [["item", "value"]]
    node_list = []
    flops_list = []
    params_list = []
    activations_list = []
    total_stats = namedtuple(
        "total_stats", ["param_size", "param_dims", "flops", "act_size", "act_dims"]
    )
    stats_details = namedtuple("module_stats", ["params", "flops", "activations"])

    disable_stats = False
    for node in tqdm(graph.all_oprs):
        if hasattr(node, "output_idx"):
            node_oup = node.outputs[node.output_idx]
        else:
            if len(node.outputs) != 1:
                logger.warning(
                    "OpNode {} has more than one output and not has 'output_idx' attr.".format(
                        node
                    )
                )
            node_oup = node.outputs[0]

        inp_list = [process_name(var.owner.name) for var in node.inputs]
        if log_path:
            # detail format see tensorboard/compat/proto/attr_value.proto
            attr = {
                "params": AttrValue(s=str(node.params).encode(encoding="utf-8")),
                "dtype": AttrValue(s=str(node_oup.dtype).encode(encoding="utf-8")),
            }
            if node_oup.shape:
                attr["_output_shapes"] = AttrValue(
                    list=AttrValue.ListValue(
                        shape=[
                            TensorShapeProto(
                                dim=[
                                    TensorShapeProto.Dim(size=d) for d in node_oup.shape
                                ]
                            )
                        ]
                    )
                )
            else:
                disable_stats = True
                logger.warning(
                    f"OpNode {node.name} do not has shape attr, would not calculate flops/params/activations for this net."
                )

        if not disable_stats:
            if cal_flops:
                flops_stats = get_op_stats(node, node.inputs, node.outputs)
                if flops_stats is not None:
                    # add op flops attr
                    if log_path and hasattr(flops_stats, "flops_num"):
                        attr["flops"] = AttrValue(
                            s=sizeof_fmt(flops_stats["flops"]).encode(encoding="utf-8")
                        )
                    flops_stats["name"] = node.name
                    flops_stats["class_name"] = node.type
                    flops_list.append(flops_stats)

            if cal_activations:
                acts = get_activation_stats(node_oup, has_input=has_input)
                acts["name"] = node.name
                acts["class_name"] = node.type
                activations_list.append(acts)

            if cal_params:
                if node.type == "ImmutableTensor":
                    param_stats = get_param_stats(node_oup)
                    # add tensor size attr
                    if log_path:
                        attr["size"] = AttrValue(
                            s=sizeof_fmt(param_stats["size"]).encode(encoding="utf-8")
                        )
                    param_stats["name"] = node.name
                    params_list.append(param_stats)

        if log_path:
            node_list.append(
                NodeDef(
                    name=process_name(node.name),
                    op=node.type,
                    input=inp_list,
                    attr=attr,
                )
            )
    # summary
    extra_info = {
        "#ops": len(graph.all_oprs),
        "#params": len(params_list),
    }

    (
        total_flops,
        total_param_dims,
        total_param_size,
        total_act_dims,
        total_act_size,
    ) = (0, 0, 0, 0, 0)

    if not disable_stats:
        if cal_params:
            total_param_dims, total_param_size, params_list = sum_param_stats(
                params_list, bar_length_max
            )
            extra_info["total_param_dims"] = sizeof_fmt(total_param_dims, suffix="")
            extra_info["total_param_size"] = sizeof_fmt(total_param_size)
            if logging_to_stdout:
                print_param_stats(params_list)

        if cal_flops:
            total_flops, flops_list = sum_op_stats(flops_list, bar_length_max)
            extra_info["total_flops"] = sizeof_fmt(total_flops, suffix="OPs")
            if logging_to_stdout:
                print_op_stats(flops_list)

        if cal_activations:
            total_act_dims, total_act_size, activations_list = sum_activations_stats(
                activations_list, bar_length_max
            )
            extra_info["total_act_dims"] = sizeof_fmt(total_act_dims, suffix="")
            extra_info["total_act_size"] = sizeof_fmt(total_act_size)
            if logging_to_stdout:
                print_activations_stats(activations_list, has_input=has_input)

        if cal_flops and cal_params:
            extra_info["flops/param_size"] = "{:3.3f}".format(
                total_flops / total_param_size
            )

        print_summary(**extra_info)

    if log_path:
        graph_def = GraphDef(node=node_list, versions=VersionDef(producer=22))

        device = "/device:CPU:0"
        stepstats = RunMetadata(
            step_stats=StepStats(dev_stats=[DeviceStepStats(device=device)])
        )
        writer = SummaryWriter(log_path)
        writer._get_file_writer().add_graph((graph_def, stepstats))

    return (
        total_stats(
            param_size=total_param_size,
            param_dims=total_param_dims,
            flops=total_flops,
            act_size=total_act_size,
            act_dims=total_act_dims,
        ),
        stats_details(
            params=params_list, flops=flops_list, activations=activations_list
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description="load a megengine dumped model and export log file for tensorboard visualization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_path", help="dumped model path.")
    parser.add_argument("--log_path", help="tensorboard log path.")
    parser.add_argument(
        "--load_input_data",
        help="load input data from pickle file; it should be a numpy array or a dict of numpy array",
    )
    parser.add_argument(
        "--bar_length_max",
        type=int,
        default=20,
        help="size of bar indicating max flops or parameter size in net stats.",
    )
    parser.add_argument(
        "--cal_params",
        action="store_true",
        help="whether calculate and record params size.",
    )
    parser.add_argument(
        "--cal_flops",
        action="store_true",
        help="whether calculate and record op flops.",
    )
    parser.add_argument(
        "--cal_activations",
        action="store_true",
        help="whether calculate and record op activations.",
    )
    parser.add_argument(
        "--logging_to_stdout",
        action="store_true",
        help="whether print all calculated statistic details.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="whether print all stats. Tensorboard logs will be placed in './log' if not specified.",
    )
    args = parser.parse_args()
    if args.load_input_data:
        logger.info("load data from {}".format(args.load_input_data))
        data = mge.load(args.load_input_data)
        if isinstance(data, dict):
            for v in data.values():
                assert isinstance(
                    v, np.ndarray
                ), "data should provide ndarray; got {} instead".format(v)
            args.inp_dict = data
        elif isinstance(data, np.ndarray):
            args.input = data
        else:
            logger.error("input data should be a numpy array or a dict of numpy array")
    if args.all:
        args.cal_params = True
        args.cal_flops = True
        args.cal_activations = True
        args.logging_to_stdout = True
        if not args.log_path:
            args.log_path = "./log"
    kwargs = vars(args)
    kwargs.pop("all")
    kwargs.pop("load_input_data")
    visualize(**kwargs)


if __name__ == "__main__":
    main()
