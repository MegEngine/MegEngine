# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import multiprocessing as mp

from ..device import get_device_count
from .group import init_process_group
from .server import Server
from .util import get_free_ports


def _get_device_count():
    """use subprocess to avoid cuda environment initialization in the main process"""

    def run(q):
        count = get_device_count("gpu")
        q.put(count)

    q = mp.Queue()
    p = mp.Process(target=run, args=(q,))
    p.start()
    p.join()
    return q.get()


def _run_wrapped(func, master_ip, port, world_size, rank, dev, args, kwargs):
    """init distributed process group and run wrapped function"""
    init_process_group(
        master_ip=master_ip, port=port, world_size=world_size, rank=rank, device=dev
    )
    func(*args, **kwargs)


def launcher(n_gpus):
    """decorator for launching multiple processes in single-machine multi-gpu training"""

    count = _get_device_count()
    assert isinstance(n_gpus, int) and n_gpus > 1, "invalid n_gpus"
    assert n_gpus <= count, "{} gpus required, {} gpus provided".format(n_gpus, count)

    def decorator(func):
        def wrapper(*args, **kwargs):
            master_ip = "localhost"
            port = get_free_ports(1)[0]
            server = Server(port)

            procs = []
            for rank in range(n_gpus):
                p = mp.Process(
                    target=_run_wrapped,
                    args=(func, master_ip, port, n_gpus, rank, rank, args, kwargs),
                )
                p.start()
                procs.append(p)

            for rank in range(n_gpus):
                procs[rank].join()
                code = procs[rank].exitcode
                assert code == 0, "subprocess {} exit with code {}".format(rank, code)

        return wrapper

    return decorator
