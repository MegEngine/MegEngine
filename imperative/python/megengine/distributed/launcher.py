# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import functools
import multiprocessing as mp
import os
import queue

from .. import _exit
from ..core._imperative_rt.core2 import sync
from ..logger import get_logger
from .group import group_barrier, init_process_group
from .helper import get_device_count_by_fork
from .server import Client, Server

WARN_SUBPROCESS_EXIT_WITHOUT_RETURN = (
    "subprocess exited with code 0 but did not return a value"
)


def _run_wrapped(
    func,
    is_multimachine,
    master_ip,
    port,
    world_size,
    rank,
    dev,
    device_type,
    args,
    kwargs,
    queue: mp.Queue,
):
    """Init distributed process group and run wrapped function."""
    init_process_group(
        master_ip=master_ip,
        port=port,
        world_size=world_size,
        rank=rank,
        device=dev,
        device_type=device_type,
    )
    # set NCCL_LAUNCH_MODE to avoid deadlock
    os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"
    if is_multimachine:
        group_barrier()
    ret = func(*args, **kwargs)
    queue.put((dev, ret))
    sync()
    if is_multimachine:
        group_barrier()
    _exit(0)


class launcher:
    """Decorator for launching multiple processes in single-machine multi-gpu training.

    :param func: the function you want to launch in distributed mode.
    :param n_gpus: how many devices each node.
    :param world_size: how many devices totally.
    :param rank_start: start number for rank.
    :param master_ip: ip address for master node (where the rank 0 is).
    :param port: server port for distributed server.
    """

    def __new__(cls, *args, **kwargs):
        if not args:
            return functools.partial(cls, **kwargs)
        return super().__new__(cls)

    def __init__(
        self,
        func,
        n_gpus=None,
        world_size=None,
        rank_start=0,
        master_ip="localhost",
        port=0,
        device_type="xpu",
    ):
        self.func = func
        self.n_gpus = (
            n_gpus if n_gpus is not None else get_device_count_by_fork(device_type)
        )
        self.world_size = world_size if world_size is not None else self.n_gpus
        self.rank_start = rank_start
        self.master_ip = master_ip
        self.port = port
        self.device_type = device_type
        # master node create server
        if self.rank_start == 0:
            self.server = Server(self.port)
            self.port = self.server.py_server_port
        else:
            assert self.port != 0, "you have to assign a port for distributed server"

    def __call__(self, *args, **kwargs):
        procs = []
        queue = mp.Queue(self.n_gpus)
        results = [None] * self.n_gpus
        for dev in range(self.n_gpus):
            p = mp.Process(
                target=_run_wrapped,
                args=(
                    self.func,
                    self.world_size > self.n_gpus,
                    self.master_ip,
                    self.port,
                    self.world_size,
                    dev + self.rank_start,
                    dev,
                    self.device_type,
                    args,
                    kwargs,
                    queue,
                ),
            )
            p.start()
            procs.append(p)

        devs = list(range(self.n_gpus))

        def terminate():
            for dev in devs:
                procs[dev].terminate()
            devs.clear()

        result_count = 0
        while len(devs) > 0:
            left = []
            # check all processes in one second
            time_to_wait = 1.0 / len(devs)
            for dev in devs:
                procs[dev].join(time_to_wait)
                code = procs[dev].exitcode
                # terminate processes if one of them has failed
                if code != 0 and code != None:
                    terminate()
                assert (
                    code == 0 or code == None
                ), "subprocess {} exit with code {}".format(dev + self.rank_start, code)
                if code == None:
                    left.append(dev)

                # DO NOT delete it, multiprocess.Queue has small buffer
                # fetch data early to avoid dead lock
                if not queue.empty():
                    result_count += 1
                    dev, ret = queue.get_nowait()
                    results[dev] = ret
            devs = left

        while not queue.empty():
            result_count += 1
            dev, ret = queue.get_nowait()
            results[dev] = ret

        if result_count < self.n_gpus:
            get_logger().warning(WARN_SUBPROCESS_EXIT_WITHOUT_RETURN)

        return results
