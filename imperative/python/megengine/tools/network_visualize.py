#! /usr/bin/env python3
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse
import logging
import re
from collections import namedtuple

import numpy as np

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
    bar_length_max: int = 20,
    log_params: bool = True,
    log_flops: bool = True,
    log_activations: bool = True,
):
    r"""
    Load megengine dumped model and visualize graph structure with tensorboard log files.
    Can also record and print model's statistics like :func:`~.module_stats`

    :param model_path: dir path for megengine dumped model.
    :param log_path: dir path for tensorboard graph log.
    :param bar_length_max: size of bar indicating max flops or parameter size in net stats.
    :param log_params: whether print and record params size.
    :param log_flops: whether print and record op flops.
    :param log_activations: whether print and record op activations.
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
    total_stats = namedtuple("total_stats", ["param_size", "flops", "act_size"])
    stats_details = namedtuple("module_stats", ["params", "flops", "activations"])

    for node in graph.all_oprs:
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
                "_output_shapes": AttrValue(
                    list=AttrValue.ListValue(
                        shape=[
                            TensorShapeProto(
                                dim=[
                                    TensorShapeProto.Dim(size=d) for d in node_oup.shape
                                ]
                            )
                        ]
                    )
                ),
                "params": AttrValue(s=str(node.params).encode(encoding="utf-8")),
                "dtype": AttrValue(s=str(node_oup.dtype).encode(encoding="utf-8")),
            }
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

            acts = get_activation_stats(node_oup)
            acts["name"] = node.name
            acts["class_name"] = node.type
            activations_list.append(acts)

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
        total_param_size,
    ) = (0, 0, 0, 0, 0)

    total_param_dims, total_param_size, params = sum_param_stats(
        params_list, bar_length_max
    )
    extra_info["total_param_dims"] = sizeof_fmt(total_param_dims, suffix="")
    extra_info["total_param_size"] = sizeof_fmt(total_param_size)
    if log_params:
        print_param_stats(params)

    total_flops, flops = sum_op_stats(flops_list, bar_length_max)
    extra_info["total_flops"] = sizeof_fmt(total_flops, suffix="OPs")
    if log_flops:
        print_op_stats(flops)

    total_act_dims, total_act_size, activations = sum_activations_stats(
        activations_list, bar_length_max
    )
    extra_info["total_act_dims"] = sizeof_fmt(total_act_dims, suffix="")
    extra_info["total_act_size"] = sizeof_fmt(total_act_size)
    if log_activations:
        print_activations_stats(activations)

    extra_info["flops/param_size"] = "{:3.3f}".format(total_flops / total_param_size)

    if log_path:
        graph_def = GraphDef(node=node_list, versions=VersionDef(producer=22))

        device = "/device:CPU:0"
        stepstats = RunMetadata(
            step_stats=StepStats(dev_stats=[DeviceStepStats(device=device)])
        )
        writer = SummaryWriter(log_path)
        writer._get_file_writer().add_graph((graph_def, stepstats))

    print_summary(**extra_info)

    return (
        total_stats(
            param_size=total_param_size, flops=total_flops, act_size=total_act_size,
        ),
        stats_details(params=params, flops=flops, activations=activations),
    )


def main():
    parser = argparse.ArgumentParser(
        description="load a megengine dumped model and export log file for tensorboard visualization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_path", help="dumped model path.")
    parser.add_argument("--log_path", help="tensorboard log path.")
    parser.add_argument(
        "--bar_length_max",
        type=int,
        default=20,
        help="size of bar indicating max flops or parameter size in net stats.",
    )
    parser.add_argument(
        "--log_params",
        action="store_true",
        help="whether print and record params size.",
    )
    parser.add_argument(
        "--log_flops", action="store_true", help="whether print and record op flops.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="whether print all stats. Tensorboard logs will be placed in './log' if not specified.",
    )
    args = parser.parse_args()
    if args.all:
        args.log_params = True
        args.log_flops = True
        if not args.log_path:
            args.log_path = "./log"
    kwargs = vars(args)
    kwargs.pop("all")
    visualize(**kwargs)


if __name__ == "__main__":
    main()
