# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
from resnet50 import Resnet50

import megengine as mge
import megengine.distributed as dist
import megengine.functional as F
from megengine._internal.plugin import CompGraphProfiler
from megengine.core import Graph, tensor
from megengine.core.graph import get_default_graph
from megengine.functional.debug_param import (
    get_conv_execution_strategy,
    set_conv_execution_strategy,
)
from megengine.jit import trace
from megengine.module import BatchNorm2d, Conv2d, Linear, MaxPool2d, Module
from megengine.optimizer import SGD

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples"))


def init_profiler(comp_graph=get_default_graph()):
    profiler = CompGraphProfiler(comp_graph)
    return profiler


def dump_profiler(profiler, filename):
    with open(filename, "w") as fout:
        json.dump(profiler.get(), fout, indent=2)


def print_gpu_usage():
    stdout = subprocess.getoutput("nvidia-smi")
    for line in stdout.split("\n"):
        for item in line.split(" "):
            if "MiB" in item:
                print("Finish with GPU Usage", item)
                break


def run_perf(
    batch_size=64,
    warm_up=True,
    dump_prof=None,
    opt_level=2,
    conv_fastrun=False,
    run_step=True,
    track_bn_stats=True,
    warm_up_iter=20,
    run_iter=100,
    num_gpu=None,
    device=0,
    server=None,
    port=None,
    scale_batch_size=False,
    eager=False,
):

    if conv_fastrun:
        set_conv_execution_strategy("PROFILE")

    if num_gpu:
        dist.init_process_group(args.server, args.port, num_gpu, device, device)
        if scale_batch_size:
            batch_size = batch_size // num_gpu
        print("Run with data parallel, batch size = {} per GPU".format(batch_size))

    data = tensor(np.random.randn(batch_size, 3, 224, 224).astype("float32"))
    label = tensor(np.random.randint(1000, size=[batch_size,], dtype=np.int32))

    net = Resnet50(track_bn_stats=track_bn_stats)
    opt = SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    def train_func(data, label):
        logits = net(data)
        loss = F.cross_entropy_with_softmax(logits, label)

        if num_gpu:
            loss = loss / num_gpu

        opt.zero_grad()
        opt.backward(loss)
        return loss

    train_func = trace(
        train_func,
        symbolic=(not eager),
        opt_level=opt_level,
        profiling=not (dump_prof is None),
    )

    if warm_up:
        print("Warm up ...")
        for _ in range(warm_up_iter):
            opt.zero_grad()
            train_func(data, label)
            if run_step:
                opt.step()
    print_gpu_usage()
    print("Running train ...")
    start = time.time()
    for _ in range(run_iter):
        opt.zero_grad()
        train_func(data, label)
        if run_step:
            opt.step()

    time_used = time.time() - start

    if dump_prof:
        with open(dump_prof, "w") as fout:
            json.dump(train_func.get_profile(), fout, indent=2)

    return time_used / run_iter


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Running regression test on Resnet 50",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch-size", type=int, default=64, help="batch size ")
    parser.add_argument(
        "--warm-up", type=str2bool, default=True, help="whether to warm up"
    )
    parser.add_argument(
        "--dump-prof",
        type=str,
        default=None,
        help="pass the json file path to dump the profiling result",
    )
    parser.add_argument("--opt-level", type=int, default=2, help="graph opt level")
    parser.add_argument(
        "--conv-fastrun",
        type=str2bool,
        default=False,
        help="whether to use conv fastrun mode",
    )
    parser.add_argument(
        "--run-step",
        type=str2bool,
        default=True,
        help="whether to run optimizer.step()",
    )
    parser.add_argument(
        "--track-bn-stats",
        type=str2bool,
        default=True,
        help="whether to track bn stats",
    )
    parser.add_argument(
        "--warm-up-iter", type=int, default=20, help="number of iters to warm up"
    )
    parser.add_argument(
        "--run-iter", type=int, default=100, help="number of iters to collect wall time"
    )
    parser.add_argument("--server", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument(
        "--scale-batch-size",
        type=str2bool,
        default=False,
        help="whether to divide batch size by number of GPUs",
    )
    parser.add_argument(
        "--eager", type=str2bool, default=False, help="whether to use eager mode"
    )

    # Data parallel related
    parser.add_argument("--num-gpu", type=int, default=None)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    print(vars(args))

    os.environ["MGB_JIT_BACKEND"] = "NVRTC"

    t = run_perf(**vars(args))

    print("**********************************")
    print("Wall time per iter {:.0f} ms".format(t * 1000))
    print("**********************************")
    get_default_graph().clear_device_memory()
