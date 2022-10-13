# -*- coding: utf-8 -*-
from mprop import mproperty

from ..core._imperative_rt.core2 import group_end, group_start
from . import group
from .group import (
    WORLD,
    Group,
    get_backend,
    get_client,
    get_mm_server_addr,
    get_py_server_addr,
    get_rank,
    get_world_size,
    group_barrier,
    init_process_group,
    is_distributed,
    new_group,
    override_backend,
)
from .helper import bcast_list_, make_allreduce_cb, synchronized
from .launcher import launcher
from .server import Client, Server


@mproperty
def backend(mod):
    r"""Get or set backend of collective communication.
    Available backends are ['nccl', 'shm', 'rccl']

    Examples:

        .. code-block::

            import megengine.distributed as dist
            dist.backend = "nccl"
    """
    assert group._sd, "please call init_process_group first"
    return group._sd.backend


@backend.setter
def backend(mod, val):
    assert group._sd, "please call init_process_group first"
    group._sd.backend = val
