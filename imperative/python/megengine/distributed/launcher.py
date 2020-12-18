# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import functools
import multiprocessing as mp

from ..core._imperative_rt.core2 import sync
from .group import group_barrier, init_process_group
from .helper import get_device_count_by_fork
from .server import Server
from .util import get_free_ports


def _run_wrapped(
    func, is_multimachine, master_ip, port, world_size, rank, dev, args, kwargs
):
    """Init distributed process group and run wrapped function."""
    init_process_group(
        master_ip=master_ip, port=port, world_size=world_size, rank=rank, device=dev
    )
    if is_multimachine:
        group_barrier()
    func(*args, **kwargs)
    sync()
    if is_multimachine:
        group_barrier()


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
    ):
        self.func = func
        self.n_gpus = n_gpus if n_gpus is not None else get_device_count_by_fork("gpu")
        self.world_size = world_size if world_size is not None else self.n_gpus
        self.rank_start = rank_start
        self.master_ip = master_ip
        self.port = port
        # master node create server
        if self.rank_start == 0:
            self.server = Server(self.port)
            self.port = self.server.py_server_port
        else:
            assert self.port != 0, "you have to assign a port for distributed server"

    def __call__(self, *args, **kwargs):
        procs = []
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
                    args,
                    kwargs,
                ),
            )
            p.start()
            procs.append(p)

        devs = list(range(self.n_gpus))

        while len(devs) > 0:
            left = []
            # check all processes in one second
            time_to_wait = 1.0 / len(devs)
            for dev in devs:
                procs[dev].join(time_to_wait)
                code = procs[dev].exitcode
                # terminate processes if one of them has failed
                if code != 0 and code != None:
                    for i in devs:
                        procs[i].terminate()
                assert (
                    code == 0 or code == None
                ), "subprocess {} exit with code {}".format(dev + self.rank_start, code)
                if code == None:
                    left.append(dev)
            devs = left
