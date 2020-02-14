# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Optional, Union

import megengine._internal as mgb
from megengine._internal.opr_param_defs import CollectiveComm as CollParam

from ..core import Buffer, Parameter, Tensor, wrap_io_tensor
from ..core.graph import get_default_graph
from ..functional import add_update
from .util import (
    get_backend,
    get_master_ip,
    get_master_port,
    get_rank,
    get_world_size,
    is_distributed,
)


@wrap_io_tensor
def _collective_comm(
    inp: Union[Tensor, mgb.CompGraph],
    key: str,
    op: CollParam.Mode,
    nr_ranks: Optional[int] = None,
    rank: Optional[int] = None,
    root: Optional[int] = 0,
    dtype: Optional[type] = None,
    device: Optional[mgb.CompNode] = None,
    comp_graph: Optional[mgb.CompGraph] = None,
) -> Tensor:
    """Helper function for creating collective_comm operators

    :param inp: tensor or comp_graph
    :param key: unique identifier for collective communication
    :param op: mode of collective communication
    :param nr_ranks: number of ranks, use util.get_world_size() as default
    :param rank: rank of the current process, use util.get_rank() as default
    :param root: rank of root node, use 0 as default
    :param dtype: output data type, use dtype of inp as default
    :param device: output comp node, use comp node of inp as default
    :param comp_graph: output comp graph, use comp graph of inp as default
    """
    return mgb.opr.collective_comm(
        inp,
        key=str(key),
        nr_devices=nr_ranks if nr_ranks is not None else get_world_size(),
        rank=rank if rank is not None else get_rank(),
        root=root,
        server_addr=get_master_ip(),
        port=get_master_port(),
        param=CollParam(mode=op),
        dtype=dtype,
        backend=get_backend(),
        comp_node=device,
        comp_graph=comp_graph,
    )


def reduce_sum(
    tensor: Tensor,
    key: str,
    nr_ranks: Optional[int] = None,
    rank: Optional[int] = None,
    root: Optional[int] = 0,
) -> Tensor:
    """Create reduce_sum operator for collective communication

    :param tensor: input tensor
    :param key: unique identifier for collective communication
    :param nr_ranks: number of ranks, use util.get_world_size() as default
    :param rank: rank of the current process, use util.get_rank() as default
    :param root: rank of root node, use 0 as default
    """
    return _collective_comm(
        tensor,
        key,
        CollParam.Mode.REDUCE_SUM,
        nr_ranks,
        rank,
        root,
        device=tensor.device,
    )


def broadcast(
    tensor: Tensor,
    key: str,
    nr_ranks: Optional[int] = None,
    rank: Optional[int] = None,
    root: Optional[int] = 0,
) -> Tensor:
    """Create broadcast operator for collective communication

    :param tensor: input tensor
    :param key: unique identifier for collective communication
    :param nr_ranks: number of ranks, use util.get_world_size() as default
    :param rank: rank of the current process, use util.get_rank() as default
    :param root: rank of root node, use 0 as default
    """
    if key is None:
        key = tensor._symvar.name

    if rank is None:
        rank = get_rank()

    if rank == root:
        return _collective_comm(
            tensor,
            key,
            CollParam.Mode.BROADCAST,
            nr_ranks,
            rank,
            root,
            device=tensor.device,
        )
    else:
        return _collective_comm(
            get_default_graph(),
            key,
            CollParam.Mode.BROADCAST,
            nr_ranks,
            rank,
            root,
            dtype=tensor._symvar.dtype,
            device=tensor.device,
        )


def all_gather(
    tensor: Tensor, key: str, nr_ranks: Optional[int] = None, rank: Optional[int] = None
) -> Tensor:
    """Create all_gather operator for collective communication

    :param tensor: input tensor
    :param key: unique identifier for collective communication
    :param nr_ranks: number of ranks, use util.get_world_size() as default
    :param rank: rank of the current process, use util.get_rank() as default
    """
    return _collective_comm(tensor, key, CollParam.Mode.ALL_GATHER, nr_ranks, rank, 0)


def reduce_scatter_sum(
    tensor: Tensor, key: str, nr_ranks: Optional[int] = None, rank: Optional[int] = None
) -> Tensor:
    """Create reduce_scatter_sum operator for collective communication

    :param tensor: input tensor
    :param key: unique identifier for collective communication
    :param nr_ranks: number of ranks, use util.get_world_size() as default
    :param rank: rank of the current process, use util.get_rank() as default
    """
    return _collective_comm(
        tensor, key, CollParam.Mode.REDUCE_SCATTER_SUM, nr_ranks, rank
    )


def all_reduce_sum(
    tensor: Tensor, key: str, nr_ranks: Optional[int] = None, rank: Optional[int] = None
) -> Tensor:
    """Create all_reduce_sum operator for collective communication

    :param tensor: input tensor
    :param key: unique identifier for collective communication
    :param nr_ranks: number of ranks, use util.get_world_size() as default
    :param rank: rank of the current process, use util.get_rank() as default
    """
    return _collective_comm(tensor, key, CollParam.Mode.ALL_REDUCE_SUM, nr_ranks, rank)


def all_reduce_max(
    tensor: Tensor, key: str, nr_ranks: Optional[int] = None, rank: Optional[int] = None
) -> Tensor:
    """Create all_reduce_max operator for collective communication

    :param tensor: input tensor
    :param key: unique identifier for collective communication
    :param nr_ranks: number of ranks, use util.get_world_size() as default
    :param rank: rank of the current process, use util.get_rank() as default
    """
    return _collective_comm(tensor, key, CollParam.Mode.ALL_REDUCE_MAX, nr_ranks, rank)


def all_reduce_min(
    tensor: Tensor, key: str, nr_ranks: Optional[int] = None, rank: Optional[int] = None
) -> Tensor:
    """Create all_reduce_min operator for collective communication

    :param tensor: input tensor
    :param key: unique identifier for collective communication
    :param nr_ranks: number of ranks, use util.get_world_size() as default
    :param rank: rank of the current process, use util.get_rank() as default
    """
    return _collective_comm(tensor, key, CollParam.Mode.ALL_REDUCE_MIN, nr_ranks, rank)


def bcast_param(
    inp: Union[Buffer, Parameter],
    key: str,
    nr_ranks: Optional[int] = None,
    rank: Optional[int] = None,
    root: Optional[int] = 0,
) -> None:
    """Broadcast parameters among devices

    :param inp: input Buffer or Parameter to be synchronized
    :param key: unique identifier for collective communication
    :param nr_ranks: number of ranks, use util.get_world_size() as default
    :param rank: rank of the current process, use util.get_rank() as default
    :param root: rank of root node, use 0 as default
    """
    if not is_distributed():
        return
    assert isinstance(inp, (Buffer, Parameter))
    bcast_res = broadcast(inp, key, nr_ranks, rank, root)
    add_update(inp, bcast_res, alpha=0)
