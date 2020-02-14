# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import functools
from typing import Callable, Optional

import megengine._internal as mgb

from ..core import set_default_device

_master_ip = None
_master_port = 0
_world_size = 0
_rank = 0
_backend = None


def init_process_group(
    master_ip: str,
    master_port: int,
    world_size: int,
    rank: int,
    dev: int,
    backend: Optional[str] = "nccl",
) -> None:
    """Initialize the distributed process group, and also specify the device used in the current process.

    :param master_ip: IP address of the master node.
    :param master_port: Port available for all processes to communicate.
    :param world_size: Total number of processes participating in the job.
    :param rank: Rank of the current process.
    :param dev: The GPU device id to bind this process to.
    :param backend: Communicator backend, currently support 'nccl' and 'ucx'
    """
    global _master_ip  # pylint: disable=global-statement
    global _master_port  # pylint: disable=global-statement
    global _world_size  # pylint: disable=global-statement
    global _rank  # pylint: disable=global-statement
    global _backend  # pylint: disable=global-statement

    if not isinstance(master_ip, str):
        raise TypeError("Expect type str but got {}".format(type(master_ip)))
    if not isinstance(master_port, int):
        raise TypeError("Expect type int but got {}".format(type(master_port)))
    if not isinstance(world_size, int):
        raise TypeError("Expect type int but got {}".format(type(world_size)))
    if not isinstance(rank, int):
        raise TypeError("Expect type int but got {}".format(type(rank)))
    if not isinstance(backend, str):
        raise TypeError("Expect type str but got {}".format(type(backend)))

    _master_ip = master_ip
    _master_port = master_port
    _world_size = world_size
    _rank = rank
    _backend = backend

    set_default_device(mgb.comp_node("gpu" + str(dev)))

    if rank == 0:
        res = mgb.config.create_mm_server("0.0.0.0", master_port)
        if res != master_port:
            raise Exception("Failed to start server on port {}".format(master_port))


def is_distributed() -> bool:
    """Return True if the distributed process group has been initialized"""
    return _world_size is not None and _world_size > 1


def get_master_ip() -> str:
    """Get the IP address of the master node"""
    return str(_master_ip)


def get_master_port() -> int:
    """Get the port of the rpc server on the master node"""
    return _master_port


def get_world_size() -> int:
    """Get the total number of processes participating in the job"""
    return _world_size


def get_rank() -> int:
    """Get the rank of the current process"""
    return _rank


def get_backend() -> str:
    """Get the backend str"""
    return str(_backend)


def group_barrier() -> None:
    """Block until all ranks in the group reach this barrier"""
    mgb.config.group_barrier(_master_ip, _master_port, _world_size, _rank)


def synchronized(func: Callable):
    """Decorator. Decorated function will synchronize when finished.
    Specifically, we use this to prevent data race during hub.load"""

    @functools.wraps(func)
    def _(*args, **kwargs):
        if not is_distributed():
            return func(*args, **kwargs)

        ret = func(*args, **kwargs)
        group_barrier()
        return ret

    return _
