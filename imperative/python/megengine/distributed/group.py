# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import List, Optional, Tuple

from ..device import set_default_device
from .server import Client, Server


class StaticData:
    server = None
    client = None
    master_ip = None
    py_server_port = None
    mm_server_port = None
    world_size = None
    proc_rank = None
    device = None
    backend = None
    next_stream = None


_sd = None


class Group:
    def __init__(self, proc_ranks):
        if len(proc_ranks) == 0:  # empty group
            self.proc_ranks = None
            self.stream = None
        else:
            self.reset(proc_ranks)

    def reset(self, proc_ranks):
        self.check(proc_ranks)
        self.proc_ranks = proc_ranks
        self.stream = _sd.next_stream
        _sd.next_stream += 1

    def check(self, proc_ranks):
        assert _sd is not None, "please call init_process_group first"
        for rank in proc_ranks:
            assert isinstance(rank, int)
            assert rank >= 0 and rank < _sd.world_size
        assert _sd.proc_rank in proc_ranks

    @property
    def size(self):
        assert len(self.proc_ranks) > 0, "invalid group"
        return len(self.proc_ranks)

    @property
    def key(self):
        assert len(self.proc_ranks) > 0, "invalid group"
        return ",".join(map(str, self.proc_ranks))

    @property
    def rank(self):
        assert len(self.proc_ranks) > 0, "invalid group"
        return self.proc_ranks.index(_sd.proc_rank)

    @property
    def comp_node(self):
        assert len(self.proc_ranks) > 0, "invalid group"
        return "gpu{}:{}".format(_sd.device, self.stream)


WORLD = Group([])


def init_process_group(
    master_ip: str,
    port: int,
    world_size: int,
    rank: int,
    device: int,
    backend: Optional[str] = "nccl",
) -> None:
    """
    Initialize the distributed process group and specify the device used in the current process

    :param master_ip: ip address of the master node.
    :param port: port available for all processes to communicate.
    :param world_size: total number of processes participating in the job.
    :param rank: rank of the current process.
    :param device: the GPU device id to bind this process to.
    :param backend: communicator backend, currently support 'nccl' and 'ucx'.
    """
    if not isinstance(master_ip, str):
        raise TypeError("Expect type str but got {}".format(type(master_ip)))
    if not isinstance(port, int):
        raise TypeError("Expect type int but got {}".format(type(port)))
    if not isinstance(world_size, int):
        raise TypeError("Expect type int but got {}".format(type(world_size)))
    if not isinstance(rank, int):
        raise TypeError("Expect type int but got {}".format(type(rank)))
    if not isinstance(device, int):
        raise TypeError("Expect type int but got {}".format(type(backend)))
    if not isinstance(backend, str):
        raise TypeError("Expect type str but got {}".format(type(backend)))

    global _sd
    assert _sd is None, "init_process_group should be called only once"
    _sd = StaticData()

    assert world_size > 1
    assert rank >= 0 and rank < world_size
    assert port > 0

    _sd.client = Client(master_ip, port)
    _sd.master_ip = master_ip
    _sd.py_server_port = port
    _sd.mm_server_port = _sd.client.get_mm_server_port()
    _sd.world_size = world_size
    _sd.proc_rank = rank
    _sd.device = device
    _sd.backend = backend
    _sd.next_stream = 1

    WORLD.reset(list(range(world_size)))

    set_default_device("gpu{}".format(device))


def is_distributed() -> bool:
    """Return True if the distributed process group has been initialized."""
    return _sd is not None


def get_rank() -> int:
    """Get the rank of the current process."""
    return _sd.proc_rank if _sd is not None else 0


def get_world_size() -> int:
    """Get the total number of processes participating in the job."""
    return _sd.world_size if _sd is not None else 1


def get_backend() -> str:
    """Get the backend str."""
    assert _sd is not None, "please call init_process_group first"
    return _sd.backend if _sd is not None else None


def get_py_server_addr() -> Tuple[str, int]:
    """Get master_ip and port of python XML RPC server."""
    assert _sd is not None, "please call init_process_group first"
    return _sd.master_ip, _sd.py_server_port


def get_mm_server_addr() -> Tuple[str, int]:
    """Get master_ip and port of C++ mm_server."""
    assert _sd is not None, "please call init_process_group first"
    return _sd.master_ip, _sd.mm_server_port


def get_client() -> Client:
    """Get client of python XML RPC server."""
    assert _sd is not None, "please call init_process_group first"
    return _sd.client


def new_group(proc_ranks: List[int]) -> Group:
    """Build a subgroup containing certain ranks."""
    return Group(proc_ranks)


def group_barrier(group: Optional[Group] = WORLD) -> None:
    """Block until all ranks in the group reach this barrier."""
    # if running with single node, skip it
    if _sd is None:
        return
    assert isinstance(group, Group)
    _sd.client.group_barrier(group.key, group.size)
