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
from megengine._internal.opr_param_defs import CollectiveComm as Param

from ..core import Buffer, Parameter, Tensor, wrap_io_tensor
from ..functional import add_update
from .helper import collective_comm_symvar
from .util import get_rank, is_distributed


@wrap_io_tensor
def _collective_comm(*args, **kargs):
    return collective_comm_symvar(*args, **kargs)


def _group_check(*args):
    """Return True when arguments are all None or all not None
    """
    l = [val is None for val in args]
    return len(set(l)) <= 1


def reduce_sum(
    tensor: Tensor,
    key: Optional[str] = None,
    nr_ranks: Optional[int] = None,
    is_root: Optional[bool] = None,
) -> Tensor:
    """Create reduce_sum operator for collective communication

    :param tensor: input tensor
    :param key: unique identifier for collective communication
    :param nr_ranks: number of ranks, use util.get_world_size() as default
    :param is_root: whether this is a root node
    """
    assert _group_check(
        key, nr_ranks, is_root
    ), "key, nr_ranks, is_root should be set at the same time"
    return _collective_comm(
        tensor, key, Param.Mode.REDUCE_SUM, nr_ranks, is_root, device=tensor.device,
    )


def gather(
    tensor: Tensor,
    key: Optional[str] = None,
    nr_ranks: Optional[int] = None,
    is_root: Optional[bool] = None,
    rank: Optional[int] = None,
) -> Tensor:
    """Create gather operator for collective communication

    :param tensor: input tensor
    :param key: unique identifier for collective communication
    :param nr_ranks: number of ranks, use util.get_world_size() as default
    :param is_root: whether this is a root node
    :param rank: rank of this node
    """
    assert _group_check(
        key, nr_ranks, is_root, rank
    ), "key, nr_ranks, is_root, rank should be set at the same time"
    return _collective_comm(
        tensor, key, Param.Mode.GATHER, nr_ranks, is_root, rank, device=tensor.device,
    )


def broadcast(
    tensor: Tensor,
    key: Optional[str] = None,
    nr_ranks: Optional[int] = None,
    is_root: Optional[bool] = None,
) -> Tensor:
    """Create broadcast operator for collective communication

    :param tensor: input tensor
    :param key: unique identifier for collective communication
    :param nr_ranks: number of ranks, use util.get_world_size() as default
    :param is_root: whether this is a root node
    """
    assert _group_check(
        key, nr_ranks, is_root
    ), "key, nr_ranks, is_root should be set at the same time"

    if is_root is None:
        is_root = get_rank() == 0
    if is_root:
        inp = tensor
    else:
        inp = tensor._symvar.owner_graph

    return _collective_comm(
        inp,
        key,
        Param.Mode.BROADCAST,
        nr_ranks,
        is_root,
        dtype=tensor.dtype,
        device=tensor.device,
    )


def scatter(
    tensor: Tensor,
    key: Optional[str] = None,
    nr_ranks: Optional[int] = None,
    is_root: Optional[bool] = None,
    rank: Optional[int] = None,
) -> Tensor:
    """Create scatter operator for collective communication

    :param tensor: input tensor
    :param key: unique identifier for collective communication
    :param nr_ranks: number of ranks, use util.get_world_size() as default
    :param is_root: whether this is a root node
    :param rank: rank of this node
    """
    assert _group_check(
        key, nr_ranks, is_root, rank
    ), "key, nr_ranks, is_root, rank should be set at the same time"
    if key is None:
        key = tensor._symvar.name
    if is_root is None:
        is_root = get_rank() == 0

    if is_root:
        inp = tensor
    else:
        inp = tensor._symvar.owner_graph

    return _collective_comm(
        inp,
        key,
        Param.Mode.SCATTER,
        nr_ranks,
        is_root,
        rank,
        dtype=tensor.dtype,
        device=tensor.device,
    )


def all_to_all(
    tensor: Tensor,
    key: Optional[str] = None,
    nr_ranks: Optional[int] = None,
    rank: Optional[int] = None,
    local_grad: Optional[bool] = False,
) -> Tensor:
    """Create all_to_all operator for collective communication

    :param tensor: input tensor
    :param key: unique identifier for collective communication
    :param nr_ranks: number of ranks, use util.get_world_size() as default
    :param rank: rank of this node
    :param local_grad: whether use local grad
    """
    assert _group_check(
        key, nr_ranks, rank
    ), "key, nr_ranks, rank should be set at the same time"
    return _collective_comm(
        tensor, key, Param.Mode.ALL_TO_ALL, nr_ranks, rank=rank, local_grad=local_grad,
    )


def all_gather(
    tensor: Tensor,
    key: Optional[str] = None,
    nr_ranks: Optional[int] = None,
    rank: Optional[int] = None,
    local_grad: Optional[bool] = False,
) -> Tensor:
    """Create all_gather operator for collective communication

    :param tensor: input tensor
    :param key: unique identifier for collective communication
    :param nr_ranks: number of ranks, use util.get_world_size() as default
    :param rank: rank of this node
    :param local_grad: whether use local grad
    """
    assert _group_check(
        key, nr_ranks, rank
    ), "key, nr_ranks, rank should be set at the same time"
    return _collective_comm(
        tensor, key, Param.Mode.ALL_GATHER, nr_ranks, rank=rank, local_grad=local_grad
    )


def reduce_scatter_sum(
    tensor: Tensor,
    key: Optional[str] = None,
    nr_ranks: Optional[int] = None,
    rank: Optional[int] = None,
    local_grad: Optional[bool] = False,
) -> Tensor:
    """Create reduce_scatter_sum operator for collective communication

    :param tensor: input tensor
    :param key: unique identifier for collective communication
    :param nr_ranks: number of ranks, use util.get_world_size() as default
    :param rank: rank of this node
    :param local_grad: whether use local grad
    """
    assert _group_check(
        key, nr_ranks, rank
    ), "key, nr_ranks, rank should be set at the same time"
    return _collective_comm(
        tensor,
        key,
        Param.Mode.REDUCE_SCATTER_SUM,
        nr_ranks,
        rank=rank,
        local_grad=local_grad,
    )


def all_reduce_sum(
    tensor: Tensor,
    key: Optional[str] = None,
    nr_ranks: Optional[int] = None,
    local_grad: Optional[bool] = False,
) -> Tensor:
    """Create all_reduce_sum operator for collective communication

    :param tensor: input tensor
    :param key: unique identifier for collective communication
    :param nr_ranks: number of ranks, use util.get_world_size() as default
    :param local_grad: whether use local grad
    """
    assert _group_check(key, nr_ranks), "key, nr_ranks should be set at the same time"
    return _collective_comm(
        tensor, key, Param.Mode.ALL_REDUCE_SUM, nr_ranks, local_grad=local_grad
    )


def all_reduce_max(
    tensor: Tensor,
    key: Optional[str] = None,
    nr_ranks: Optional[int] = None,
    local_grad: Optional[bool] = False,
) -> Tensor:
    """Create all_reduce_max operator for collective communication

    :param tensor: input tensor
    :param key: unique identifier for collective communication
    :param nr_ranks: number of ranks, use util.get_world_size() as default
    :param local_grad: whether use local grad
    """
    assert _group_check(key, nr_ranks), "key, nr_ranks should be set at the same time"
    return _collective_comm(
        tensor, key, Param.Mode.ALL_REDUCE_MAX, nr_ranks, local_grad=local_grad
    )


def all_reduce_min(
    tensor: Tensor,
    key: Optional[str] = None,
    nr_ranks: Optional[int] = None,
    local_grad: Optional[bool] = False,
) -> Tensor:
    """Create all_reduce_min operator for collective communication

    :param tensor: input tensor
    :param key: unique identifier for collective communication
    :param nr_ranks: number of ranks, use util.get_world_size() as default
    :param local_grad: whether use local grad
    """
    assert _group_check(key, nr_ranks), "key, nr_ranks should be set at the same time"
    return _collective_comm(
        tensor, key, Param.Mode.ALL_REDUCE_MIN, nr_ranks, local_grad=local_grad
    )


def bcast_param(
    inp: Union[Buffer, Parameter],
    key: Optional[str] = None,
    nr_ranks: Optional[int] = None,
    is_root: Optional[bool] = None,
) -> None:
    """Broadcast parameters among devices

    :param inp: input Buffer or Parameter to be synchronized
    :param key: unique identifier for collective communication
    :param nr_ranks: number of ranks, use util.get_world_size() as default
    :param is_root: whether this is a root node
    """
    if not is_distributed():
        return
    assert _group_check(
        key, nr_ranks, is_root
    ), "key, nr_ranks, is_root should be set at the same time"
    assert isinstance(inp, (Buffer, Parameter))
    bcast_res = broadcast(inp, key, nr_ranks, is_root)
    add_update(inp, bcast_res, alpha=0)
