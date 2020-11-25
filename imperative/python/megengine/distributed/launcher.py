# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import multiprocessing as mp

from .group import init_process_group
from .helper import get_device_count_by_fork
from .server import Server
from .util import get_free_ports


def _run_wrapped(func, master_ip, port, world_size, rank, dev, args, kwargs):
    """Init distributed process group and run wrapped function."""
    init_process_group(
        master_ip=master_ip, port=port, world_size=world_size, rank=rank, device=dev
    )
    func(*args, **kwargs)


def launcher(func):
    """Decorator for launching multiple processes in single-machine multi-gpu training."""

    n_gpus = get_device_count_by_fork("gpu")

    def wrapper(*args, **kwargs):
        master_ip = "localhost"
        server = Server()
        port = server.py_server_port

        procs = []
        for rank in range(n_gpus):
            p = mp.Process(
                target=_run_wrapped,
                args=(func, master_ip, port, n_gpus, rank, rank, args, kwargs),
            )
            p.start()
            procs.append(p)

        ranks = [rank for rank in range(n_gpus)]

        while len(ranks) > 0:
            left = []
            # check all processes in one second
            time_to_wait = 1.0 / len(ranks)
            for rank in ranks:
                procs[rank].join(time_to_wait)
                code = procs[rank].exitcode
                # terminate processes if one of them has failed
                if code != 0 and code != None:
                    for i in ranks:
                        procs[i].terminate()
                assert (
                    code == 0 or code == None
                ), "subprocess {} exit with code {}".format(rank, code)
                if code == None:
                    left.append(rank)
            ranks = left

    return wrapper
