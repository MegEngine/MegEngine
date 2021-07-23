# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Optional, Tuple

import numpy as np

from ..core._imperative_rt.core2 import apply
from ..core.autodiff.grad import Function, _grad_manager_dict
from ..core.ops.builtin import CollectiveComm, Copy, RemoteRecv, RemoteSend
from ..core.tensor.utils import isscalar, setscalar
from ..device import get_default_device, what_is_xpu
from ..tensor import Tensor
from . import group
from .group import WORLD, Group, get_client, get_mm_server_addr, get_rank

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


_device2backend = {
    "gpu": "nccl",
    "cuda": "nccl",
    "rocm": "rccl",
}


def _backend():
    if group._sd.backend == "auto":
        return _device2backend[what_is_xpu()]
    else:
        return group._sd.backend


def collective_comm(inp, mode, group, device):
    """Helper function for applying collective communication functions."""
    assert isinstance(group, Group)
    if group is None:
        return inp
    if device is None:
        device = ""
    addr, port = get_mm_server_addr()
    op = CollectiveComm(
        key=group.key + _backend(),
        nr_devices=group.size,
        rank=group.rank,
        is_root=(group.rank == 0),
        local_grad=False,
        addr=addr,
        port=port,
        mode=mode,
        dtype=inp.dtype,
        backend=_backend(),
        comp_node=device,
    )
    (result,) = apply(op, inp)
    # assume all workers have homogeneous shape
    if mode in (
        CollectiveComm.Mode.REDUCE_SUM,
        CollectiveComm.Mode.BROADCAST,
        CollectiveComm.Mode.ALL_REDUCE_SUM,
        CollectiveComm.Mode.ALL_REDUCE_MAX,
        CollectiveComm.Mode.ALL_REDUCE_MIN,
    ):
        if isscalar(inp):
            setscalar(result)
    return result


def _save_output_for_autodiff(inp, out):
    for g in _grad_manager_dict.values():
        if g._is_attached_to(inp):
            g._refkeeper.append(out)


def _bcast_has_grad(group, grad):
    if group.rank == 0:
        has_grad = grad is not None
        get_client().bcast_val(has_grad, group.key, group.size)
    else:
        has_grad = get_client().bcast_val(None, group.key, group.size)
    return has_grad


def _bcast_shape_dtype(group, inp):
    if group.rank == 0:
        # FIXME in some cases, shape is not available(output of condtake)
        shape = inp._tuple_shape
        dtype = np.dtype(inp.dtype).name
        get_client().bcast_val({"shape": shape, "dtype": dtype}, group.key, group.size)
    else:
        val = get_client().bcast_val(None, group.key, group.size)
        shape = val["shape"]
        dtype = val["dtype"]

    return shape, dtype


def _bcast_tracer_state(group, inp):
    if group.rank == 0:
        tracer_keys = []
        for n, g in _grad_manager_dict.items():
            if g._is_attached_to(inp):
                tracer_keys.append(n)
        get_client().bcast_val(tracer_keys, group.key, group.size)
    else:
        tracer_keys = get_client().bcast_val(None, group.key, group.size)
        for n in tracer_keys:
            g = _grad_manager_dict.get(n)
            if g is not None:
                g.wrt(inp)
                g._refkeeper.append(inp)


def _dummy_input(shape, dtype, device=None):
    if device is None:
        device = get_default_device()
    inp = Tensor(0, dtype=dtype, device=device)
    if len(shape) > 0:
        inp = inp._broadcast(shape)
    return inp


class _ReduceSum(Function):
    def __init__(self, group=WORLD, device=None):
        self.group = group
        self.out_device = device

    def forward(self, data):
        self.in_device = str(data.device)
        return collective_comm(
            data, CollectiveComm.Mode.REDUCE_SUM, self.group, self.out_device,
        )

    def backward(self, grad):
        has_grad = _bcast_has_grad(self.group, grad)
        if has_grad:
            return broadcast(grad, self.group, self.in_device)


def reduce_sum(
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = None,
) -> Tensor:
    """
    Create reduce_sum operator for collective communication.

    :param inp: input tensor.
    :param group: communication group.
    :param device: execution device.
    """
    op = _ReduceSum(group, device)
    (out,) = apply(op, inp)

    if group.rank == 0:
        return out
    else:
        _save_output_for_autodiff(inp, out)


class _Broadcast(Function):
    def __init__(self, group=WORLD, device=None):
        self.group = group
        self.out_device = device

    def forward(self, data):
        self.in_device = str(data.device)
        return collective_comm(
            data, CollectiveComm.Mode.BROADCAST, self.group, self.out_device,
        )

    def backward(self, grad):
        # TODO backward with a part of grad
        if grad is not None:
            return reduce_sum(grad, self.group, self.in_device)


def broadcast(
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = None,
) -> Tensor:
    """
    Create broadcast operator for collective communication.

    :param inp: input tensor.
    :param group: communication group.
    :param device: execution device.
    """
    shape, dtype = _bcast_shape_dtype(group, inp)
    if group.rank != 0:
        # dummy input to infer shape
        inp = _dummy_input(shape, dtype, device)

    _bcast_tracer_state(group, inp)

    op = _Broadcast(group, device)
    (out,) = apply(op, inp)
    return out


def _bcast_param(
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = None
) -> Tensor:
    mode = CollectiveComm.Mode.BROADCAST
    return collective_comm(inp, mode, group, device)


def all_gather(
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = None,
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
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = None,
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
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = None,
) -> Tensor:
    r"""
    Create all_reduce_sum operator for collective communication.

    This operator sums the tensor data by coordinates across the specified group and returns a tensor with the shape of the input tensor.

    Args:
        inp: The tensor data to apply this operator on.
        group: The communication node list instance of :class:'Group' to apply this operator across. The default group is WORLD which means all processes available.
        Specify a list of process ranks to apply this operator on specific processes, e.g. [1, 3, 5].
        device: The specific device type of :class:'str' to execute this operator. The default device is None which mean the device of inp will be used.
        Specify "cpu" or "gpu" to execute this operator on specific devices.

    Returns:
        opt: The reduce sum tensor of the input tensor data across the specified group.

    Examples:

    .. code-block::

        import megengine as mge
        import megengine.distributed as dist
        import numpy as np
        from warnings import warn


        def func(sum_value):
            # get the rank of this process, the ranks shold be 0, 1, 2, 3 for a 4 gpu task
            rank = dist.get_rank()
            data = mge.tensor(rank)
            # the result should be n * (n - 1) / 2 for all processes
            result = mge.functional.distributed.all_reduce_sum(data).item()
            assert result == sum_value


        def main():
            p_num = mge.device.get_device_count("gpu")
            if p_num < 2:
                warn('This opr only works on group with more than one gpu')
                return
            method = dist.launcher(func)
            method(p_num * (p_num - 1) // 2)


        if __name__ == '__main__':
            main()

    """
    mode = CollectiveComm.Mode.ALL_REDUCE_SUM
    return collective_comm(inp, mode, group, device)


def all_reduce_max(
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = None,
) -> Tensor:
    """
    Create all_reduce_max operator for collective communication.

    :param inp: input tensor.
    :param group: communication group.
    :param device: execution device.
    :returns: reduced tensor. 
    """
    mode = CollectiveComm.Mode.ALL_REDUCE_MAX
    return collective_comm(inp, mode, group, device)


def all_reduce_min(
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = None,
) -> Tensor:
    """
    Create all_reduce_min operator for collective communication.

    :param inp: input tensor.
    :param group: communication group.
    :param device: execution device.
    """
    mode = CollectiveComm.Mode.ALL_REDUCE_MIN
    return collective_comm(inp, mode, group, device)


class _Gather(Function):
    def __init__(self, group=WORLD, device=None):
        self.group = group
        self.out_device = device

    def forward(self, data):
        self.in_device = str(data.device)
        return collective_comm(
            data, CollectiveComm.Mode.GATHER, self.group, self.out_device
        )

    def backward(self, grad):
        has_grad = _bcast_has_grad(self.group, grad)
        if has_grad:
            return scatter(grad, self.group, self.in_device)


def gather(
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = None,
) -> Tensor:
    """
    Create gather operator for collective communication.

    :param inp: input tensor.
    :param group: communication group.
    :param device: execution device.
    """

    op = _Gather(group, device)
    (out,) = apply(op, inp)

    if group.rank == 0:
        return out
    else:
        _save_output_for_autodiff(inp, out)


class _Scatter(Function):
    def __init__(self, group=WORLD, device=None):
        self.group = group
        self.out_device = device

    def forward(self, data):
        self.in_device = str(data.device)
        return collective_comm(
            data, CollectiveComm.Mode.SCATTER, self.group, self.out_device
        )

    def backward(self, grad):
        # TODO backward with a part of grad
        if grad is not None:
            return gather(grad, self.group, self.in_device)


def scatter(
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = None,
) -> Tensor:
    """
    Create scatter operator for collective communication.

    :param inp: input tensor.
    :param group: communication group.
    :param device: execution device.
    """
    shape, dtype = _bcast_shape_dtype(group, inp)
    if group.rank != 0:
        # dummy input to infer shape
        inp = _dummy_input(shape, dtype, device)

    _bcast_tracer_state(group, inp)

    op = _Scatter(group, device)
    (out,) = apply(op, inp)
    return out


def all_to_all(
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = None,
) -> Tensor:
    """
    Create all_to_all operator for collective communication.

    :param inp: input tensor.
    :param group: communication group.
    :param device: execution device.
    """
    mode = CollectiveComm.Mode.ALL_TO_ALL
    return collective_comm(inp, mode, group, device)


class _SendRecvGroup:
    def __init__(self, rank_from, rank_to):
        self.key = "{}->{}".format(rank_from, rank_to)
        self.rank_from = rank_from
        self.rank_to = rank_to
        self.size = 2

    @property
    def rank(self):
        if get_rank() == self.rank_from:
            return 0
        else:
            return 1


class _RemoteSend(Function):
    def __init__(self, op: RemoteSend):
        self.op = op

    def forward(self, data):
        self.device = str(data.device)
        (self.dummy,) = apply(self.op, data)
        return self.dummy

    def backward(self, grad):
        assert grad is None
        has_grad = get_client().bcast_val(None, self.op.key, 2)
        if has_grad:
            return remote_recv(self.op.rank_to, device=self.device, inp=self.dummy,)


class _RemoteRecv(Function):
    def __init__(self, op: RemoteRecv):
        self.op = op

    def forward(self, dummy):
        return apply(self.op, dummy)

    def backward(self, grad):
        get_client().bcast_val(grad is not None, self.op.key, 2)
        if grad is not None:
            remote_send(grad, self.op.rank_from)


def remote_send(inp: Tensor, dest_rank: int):
    """
    Send a Tensor to a remote process.

    :param inp: tensor to send.
    :param dest_rank: destination process rank.
    """
    group = _SendRecvGroup(get_rank(), dest_rank)
    _bcast_shape_dtype(group, inp)

    _bcast_tracer_state(group, inp)

    op = RemoteSend()
    op.key = group.key
    op.addr, op.port = get_mm_server_addr()
    op.rank_to = dest_rank
    op.backend = _backend()
    (out,) = apply(_RemoteSend(op), inp)

    _save_output_for_autodiff(inp, out)


def remote_recv(src_rank: int, device: Optional[str] = None, inp=None) -> Tensor:
    """
    Receive a Tensor from a remote process.

    :param src_rank: source process rank.
    :param device: the device to place the received tensor.
    :param inp: dummy input to determine recved tensor type
    """
    group = _SendRecvGroup(src_rank, get_rank())
    shape, dtype = _bcast_shape_dtype(group, None)

    if device is None:
        device = get_default_device()
    # dummy input
    if inp is None:
        inp = Tensor(0, device=device)
    _bcast_tracer_state(group, inp)

    _isscalar = False
    if len(shape) == 0:
        shape = (1,)
        _isscalar = True

    op = RemoteRecv()
    op.key = group.key
    op.cn = device
    op.shape = shape
    op.dtype = dtype
    op.addr, op.port = get_mm_server_addr()
    op.rank_from = src_rank
    op.backend = _backend()

    (ret,) = apply(_RemoteRecv(op), inp)
    if _isscalar:
        setscalar(ret)
    return ret
