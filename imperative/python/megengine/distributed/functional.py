# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Optional, Tuple

from ..core._imperative_rt.ops import CollectiveCommMode
from ..core.autodiff.builtin_op_utils import builtin_op_get_backward_fn
from ..core.autodiff.grad import (
    Tracer,
    check_backward_allow_noinput,
    get_grad_managers,
    get_op_has_grad_fn,
    tracer_apply,
)
from ..core.ops.builtin import CollectiveComm, Copy, RemoteRecv, RemoteSend
from ..core.tensor.core import apply
from ..core.tensor.tensor import Tensor, tensor_apply
from ..device import get_default_device
from ..tensor import tensor
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


@apply.register()
def _(op: RemoteSend, *args: Tensor):
    ret = tensor_apply(op, *args)

    # set extra information
    tracer_set = dict()
    for k in set().union(*(i._extra_data for i in args if isinstance(i, Tensor))):
        tracer_set[k.name] = True

    # check tracer_set in remote_recv
    get_client().set_remote_tracer(op.key, tracer_set)
    return ret


@builtin_op_get_backward_fn.register(RemoteSend)
def _(op: RemoteSend, inputs, outputs, input_requires_grad):
    def backward(*args):
        return [
            remote_recv(
                op.rank_to,
                inputs[0].shape,
                inputs[0].dtype,
                device=str(inputs[0].device),
                inp=inputs[0],
            )
        ]

    return backward, [True]


@get_op_has_grad_fn.register(RemoteSend)
def _(op: RemoteSend):
    def has_grad(opnode, reached):
        return get_client().check_is_grad(op.key)

    return has_grad


@check_backward_allow_noinput.register(RemoteSend)
def _(op: RemoteSend):
    return True


@builtin_op_get_backward_fn.register(RemoteRecv)
def _(op: RemoteRecv, inputs, outputs, input_requires_grad):
    def backward(*output_grads):
        return [remote_send(output_grads[0], op.rank_from)]

    return backward, [True]


@get_op_has_grad_fn.register(RemoteRecv)
def _(op: RemoteRecv):
    def has_grad(opnode, reached):
        ret = False
        for v in opnode.outputs:
            if v() in reached:
                ret = True
                break
        get_client().set_is_grad(op.key, ret)
        return ret

    return has_grad


def collective_comm(inp, mode, group, device):
    """Helper function for applying collective communication functions."""
    assert isinstance(group, Group)
    if group is None:
        return inp
    op = CollectiveComm()
    op.key = group.key
    op.nr_devices = group.size
    op.rank = group.rank
    op.is_root = op.rank == 0
    op.local_grad = False
    op.addr, op.port = get_mm_server_addr()
    op.mode = mode
    op.dtype = inp.dtype
    op.backend = get_backend()
    op.comp_node = device
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
    mode = CollectiveCommMode.REDUCE_SUM
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
    mode = CollectiveCommMode.BROADCAST
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
    mode = CollectiveCommMode.ALL_GATHER
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
    mode = CollectiveCommMode.REDUCE_SCATTER_SUM
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
    mode = CollectiveCommMode.ALL_REDUCE_SUM
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
    mode = CollectiveCommMode.ALL_REDUCE_MAX
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
    mode = CollectiveCommMode.ALL_REDUCE_MIN
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
    mode = CollectiveCommMode.GATHER
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
    mode = CollectiveCommMode.SCATTER
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
    mode = CollectiveCommMode.ALL_TO_ALL
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
        inp = tensor([0], device=device)
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
