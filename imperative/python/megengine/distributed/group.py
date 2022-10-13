# -*- coding: utf-8 -*-
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple

from mprop import mproperty

from ..device import _sh, set_default_device, what_is_xpu
from ..random import seed
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
    device_type = None
    machine_ranks = None


_sd = None


class Group:
    r"""Include ranked nodes running collective communication (See :mod:`~.functional.distributed`).

    By default collectives operate on the default group (also called ``WORLD``)
    and require all processes to enter the distributed function call.

    Args:
        proc_ranks: rank list of the group, the first one is root rank.


    """

    def __init__(self, proc_ranks):
        if len(proc_ranks) == 0:  # empty group
            self.proc_ranks = None
            self.stream = None
        else:
            self.reset(proc_ranks)

    def reset(self, proc_ranks):
        self.check(proc_ranks)
        self.proc_ranks = proc_ranks
        self.is_single_machine_cache = None
        self.stream = _sh.get_next()

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
        return "{}{}:{}".format(_sd.device_type, _sd.device, self.stream)

    @property
    def is_single_machine(self):
        if self.is_single_machine_cache is not None:
            return self.is_single_machine_cache
        assert _sd is not None, "please call init_process_group first"
        for rank in self.proc_ranks:
            if _sd.machine_ranks is None or rank not in _sd.machine_ranks:
                self.is_single_machine_cache = False
                return False
        self.is_single_machine_cache = True
        return True


WORLD = Group([])

_devices = {"gpu", "cuda", "rocm"}
_backends = {"nccl", "rccl", "shm", "auto"}


def init_process_group(
    master_ip: str,
    port: int,
    world_size: int,
    rank: int,
    device: int,
    backend: Optional[str] = "auto",
    device_type: str = "xpu",
) -> None:
    r"""Initialize the distributed process group and specify the device used in the current process

    Args:
        master_ip: ip address of the master node.
        port: port available for all processes to communicate.
        world_size: total number of processes participating in the job.
        rank: rank of the current process.
        device: the GPU device id to bind this process to.
        backend: communicator backend, currently support 'nccl' and 'shm'.
    """
    physical_device_type = what_is_xpu() if device_type == "xpu" else device_type
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
    if backend not in _backends:
        raise ValueError(
            "backend should be one of {} but got {}".format(_backends, backend)
        )
    if physical_device_type not in _devices:
        raise ValueError(
            "{} is not a valid distributed device type".format(device_type)
        )

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
    _sd.device_type = device_type

    WORLD.reset(list(range(world_size)))

    set_default_device("{}{}".format(device_type, device))
    seed(int(time.time()) + rank)

    if backend == "nccl":
        # init nccl env
        from ..core._imperative_rt.common import init_nccl_env

        group_barrier()
        init_nccl_env(master_ip, _sd.mm_server_port, world_size, rank, 0)


def _set_machine_ranks(ranks) -> None:
    global _sd
    assert _sd is not None

    _sd.machine_ranks = ranks


@contextmanager
def override_backend(new_backend: str):
    r"""Override distributed backend

    Args:
        new_backend: communicator backend set in this context.
    """
    global _sd
    assert _sd, "please call init_process_group first"
    old_backend = _sd.backend
    _sd.backend = new_backend
    try:
        yield
    finally:
        _sd.backend = old_backend


def is_distributed() -> bool:
    r"""Return True if the distributed process group has been initialized."""
    return _sd is not None


def get_rank() -> int:
    r"""Get the rank of the current process."""
    return _sd.proc_rank if _sd is not None else 0


def get_world_size() -> int:
    r"""Get the total number of processes participating in the job."""
    return _sd.world_size if _sd is not None else 1


def get_backend() -> str:
    r"""Get the backend str."""
    assert _sd is not None, "please call init_process_group first"
    return _sd.backend if _sd is not None else None


def get_py_server_addr() -> Tuple[str, int]:
    r"""Get master_ip and port of python XML RPC server."""
    assert _sd is not None, "please call init_process_group first"
    return _sd.master_ip, _sd.py_server_port


def get_mm_server_addr() -> Tuple[str, int]:
    r"""Get master_ip and port of C++ mm_server."""
    assert _sd is not None, "please call init_process_group first"
    return _sd.master_ip, _sd.mm_server_port


def get_client() -> Client:
    r"""Get client of python XML RPC server."""
    assert _sd is not None, "please call init_process_group first"
    return _sd.client


def new_group(proc_ranks: List[int]) -> Group:
    r"""Build a subgroup containing certain ranks."""
    return Group(proc_ranks)


def group_barrier(group: Group = WORLD) -> None:
    r"""Block until all ranks in the group reach this barrier."""
    # if running with single node, skip it
    if _sd is None:
        return
    assert isinstance(group, Group)
    _sd.client.group_barrier(group.key, group.size)
