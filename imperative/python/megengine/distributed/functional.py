# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Optional, Tuple

from ..core._imperative_rt.core2 import apply
from ..core.autodiff.grad import get_grad_managers
from ..core.ops.builtin import CollectiveComm, Copy, RemoteRecv, RemoteSend
from ..device import get_default_device
from ..tensor import Tensor
from .group import WORLD, Group, get_backend, get_client, get_mm_server_addr, get_rank

__all__ = [
    "reduce_sum",
    "broadcast",
    "all_gather",
    "reduce_scatter_sum",
    "all_reduce_sum",
    "all_reduce_max",
    "all_reduce_min",
    "gather",
    "scatter",
    "all_to_all",
    "remote_send",
    "remote_recv",
]


def collective_comm(inp, mode, group, device):
    """Helper function for applying collective communication functions."""
    assert isinstance(group, Group)
    if group is None:
        return inp
    addr, port = get_mm_server_addr()
    op = CollectiveComm(
        key=group.key,
        nr_devices=group.size,
        rank=group.rank,
        is_root=(group.rank == 0),
        local_grad=False,
        addr=addr,
        port=port,
        mode=mode,
        dtype=inp.dtype,
        backend=get_backend(),
        comp_node=device,
    )
    return apply(op, inp)[0]


def reduce_sum(
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = ""
) -> Tensor:
    """
    Create reduce_sum operator for collective communication.

    :param inp: input tensor.
    :param group: communication group.
    :param device: execution device.
    """
    mode = CollectiveComm.Mode.REDUCE_SUM
    return collective_comm(inp, mode, group, device)


def broadcast(
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = ""
) -> Tensor:
    """
    Create broadcast operator for collective communication.

    :param inp: input tensor.
    :param group: communication group.
    :param device: execution device.
    """
    mode = CollectiveComm.Mode.BROADCAST
    return collective_comm(inp, mode, group, device)


def all_gather(
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = ""
) -> Tensor:
    """
    Create all_gather operator for collective communication.

    :param inp: input tensor.
    :param group: communication group.
    :param device: execution device.
    """
    mode = CollectiveComm.Mode.ALL_GATHER
    return collective_comm(inp, mode, group, device)


def reduce_scatter_sum(
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = ""
) -> Tensor:
    """
    Create reduce_scatter_sum operator for collective communication.

    :param inp: input tensor.
    :param group: communication group.
    :param device: execution device.
    """
    mode = CollectiveComm.Mode.REDUCE_SCATTER_SUM
    return collective_comm(inp, mode, group, device)


def all_reduce_sum(
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = ""
) -> Tensor:
    """
    Create all_reduce_sum operator for collective communication.

    :param inp: input tensor.
    :param group: communication group.
    :param device: execution device.
    """
    mode = CollectiveComm.Mode.ALL_REDUCE_SUM
    return collective_comm(inp, mode, group, device)


def all_reduce_max(
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = ""
) -> Tensor:
    """
    Create all_reduce_max operator for collective communication.

    :param inp: input tensor.
    :param group: communication group.
    :param device: execution device.
    """
    mode = CollectiveComm.Mode.ALL_REDUCE_MAX
    return collective_comm(inp, mode, group, device)


def all_reduce_min(
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = ""
) -> Tensor:
    """
    Create all_reduce_min operator for collective communication.

    :param inp: input tensor.
    :param group: communication group.
    :param device: execution device.
    """
    mode = CollectiveComm.Mode.ALL_REDUCE_MIN
    return collective_comm(inp, mode, group, device)


def gather(
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = ""
) -> Tensor:
    """
    Create gather operator for collective communication.

    :param inp: input tensor.
    :param group: communication group.
    :param device: execution device.
    """
    mode = CollectiveComm.Mode.GATHER
    return collective_comm(inp, mode, group, device)


def scatter(
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = ""
) -> Tensor:
    """
    Create scatter operator for collective communication.

    :param inp: input tensor.
    :param group: communication group.
    :param device: execution device.
    """
    mode = CollectiveComm.Mode.SCATTER
    return collective_comm(inp, mode, group, device)


def all_to_all(
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = ""
) -> Tensor:
    """
    Create all_to_all operator for collective communication.

    :param inp: input tensor.
    :param group: communication group.
    :param device: execution device.
    """
    mode = CollectiveComm.Mode.ALL_TO_ALL
    return collective_comm(inp, mode, group, device)


def remote_send(inp: Tensor, dest_rank: int) -> Tensor:
    """
    Send a Tensor to a remote process.

    :param inp: tensor to send.
    :param dest_rank: destination process rank.
    """
    op = RemoteSend()
    op.key = "{}->{}".format(get_rank(), dest_rank)
    op.addr, op.port = get_mm_server_addr()
    op.rank_to = dest_rank
    return apply(op, inp)[0]


def remote_recv(
    src_rank: int,
    shape: Tuple[int],
    dtype: type,
    device: Optional[str] = None,
    inp=None,
) -> Tensor:
    """
    Receive a Tensor from a remote process.

    :param src_rank: source process rank.
    :param shape: the shape of the tensor to receive.
    :param dtype: the data type of the tensor to receive.
    :param device: the device to place the received tensor.
    :param inp: dummy input to determine recved tensor type
    """
    key = "{}->{}".format(src_rank, get_rank())

    if device is None:
        device = get_default_device()
    # dummy input
    if inp == None:
        inp = Tensor([0], device=device)
    tracer_set = get_client().check_remote_tracer(key)
    for grad_manager in get_grad_managers():
        if grad_manager.name in tracer_set:
            grad_manager.wrt(inp)

    op = RemoteRecv()
    op.key = key
    op.cn = device
    op.shape = shape
    op.dtype = dtype
    op.addr, op.port = get_mm_server_addr()
    op.rank_from = src_rank

    return apply(op, inp)[0]
