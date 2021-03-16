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

import numpy as np

from megengine.core.tensor.dtype import is_quantize
from megengine.logger import _imperative_rt_logger, get_logger, set_mgb_log_level
from megengine.utils.module_stats import (
    print_flops_stats,
    print_params_stats,
    sizeof_fmt,
)
from megengine.utils.network import Network

logger = get_logger(__name__)


def visualize(
    model_path: str,
    log_path: str,
    bar_length_max: int = 20,
    log_params: bool = True,
    log_flops: bool = True,
):
    r"""
    Load megengine dumped model and visualize graph structure with tensorboard log files.
    Can also record and print model's statistics like :func:`~.module_stats`

    :param model_path: dir path for megengine dumped model.
    :param log_path: dir path for tensorboard graph log.
    :param bar_length_max: size of bar indicating max flops or parameter size in net stats.
    :param log_params: whether print and record params size.
    :param log_flops: whether print and record op flops.
    """
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
            "TensorBoard and TensorboardX are required for visualize.", exc_info=True
        )
        return
    # FIXME: remove this after resolving "span dist too large" warning
    old_level = set_mgb_log_level(logging.ERROR)

    graph = Network.load(model_path)
    writer = SummaryWriter(log_path)

    def process_name(name):
        return name.replace(".", "/").encode(encoding="utf-8")

    node_list = []
    flops_list = []
    params_list = []
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
        attr = {
            "_output_shapes": AttrValue(
                list=AttrValue.ListValue(
                    shape=[
                        TensorShapeProto(
                            dim=[TensorShapeProto.Dim(size=d) for d in node_oup.shape]
                        )
                    ]
                )
            ),
        }
        if hasattr(node, "calc_flops"):
            flops_num = node.calc_flops()
            # add op flops attr
            attr["flops"] = AttrValue(s=sizeof_fmt(flops_num).encode(encoding="utf-8"))
            flops_list.append(
                dict(
                    name=node.name,
                    class_name=node.type,
                    input_shapes=[i.shape for i in node.inputs],
                    output_shapes=[o.shape for o in node.outputs],
                    flops_num=flops_num,
                    flops_cum=0,
                )
            )
        if node.type == "ImmutableTensor":
            param_dim = np.prod(node_oup.shape)
            # TODO: consider other quantize dtypes
            param_bytes = 1 if is_quantize(node_oup.dtype) else 4
            # add tensor size attr
            attr["size"] = AttrValue(
                s=sizeof_fmt(param_dim * param_bytes).encode(encoding="utf-8")
            )
            params_list.append(
                dict(
                    name=node.name,
                    shape=node_oup.shape,
                    param_dim=param_dim,
                    bits=param_bytes * 8,
                    size=param_dim * param_bytes,
                    size_cum=0,
                    mean="{:.2g}".format(node.numpy().mean()),
                    std="{:.2g}".format(node.numpy().std()),
                )
            )
        # FIXME(MGE-2165): nodes outside network module may lead to unknown display bug
        if not len(node.name.split(".")) > 2 and not node in graph.input_vars:
            continue
        node_list.append(
            NodeDef(
                name=process_name(node.name), op=node.type, input=inp_list, attr=attr,
            )
        )

    total_flops, total_params = 0, 0
    if log_params:
        total_params = print_params_stats(params_list, bar_length_max)
    if log_flops:
        total_flops = print_flops_stats(flops_list, bar_length_max)

    graph_def = GraphDef(node=node_list, versions=VersionDef(producer=22))

    device = "/device:CPU:0"
    stepstats = RunMetadata(
        step_stats=StepStats(dev_stats=[DeviceStepStats(device=device)])
    )
    writer._get_file_writer().add_graph((graph_def, stepstats))

    # FIXME: remove this after resolving "span dist too large" warning
    _imperative_rt_logger.set_log_level(old_level)

    return total_params, total_flops


def main():
    parser = argparse.ArgumentParser(
        description="load a megengine dumped model and export log file for tensorboard visualization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_path", help="dumped model path.")
    parser.add_argument("log_path", help="tensorboard log path.")
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
    visualize(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
