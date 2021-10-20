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
    r"""Helper function for applying collective communication functions."""
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
    r"""Reduce tensor data across the specified group by sum.
    Only root process will receive the final result.

    Args:
        inp: Input tensor.
        group: The process group to work on.
            The default group is WORLD which means all processes available.
            You can use a list of process ranks to create new group to work on it, e.g. [1, 3, 5].
        device: The specific device to execute this operator.
            None default device means the device of inp will be used.
            Specify "gpu0:1" to execute this operator on diffrent cuda stream,
            1 is stream id, and default stream id is 0.

    Returns:
        Reduced tensor if in root process, None in other processes.

    Examples:
        .. code-block::

           input = Tensor([rank])
           # Rank 0 # input: Tensor([0])
           # Rank 1 # input: Tensor([1])
           output = reduce_sum(input)
           # Rank 0 # output: Tensor([1])
           # Rank 1 # output: None

           input = Tensor([rank])
           group = Group([1, 0]) # first rank is root
           output = reduce_sum(input, group)
           # Rank 0 # output: None
           # Rank 1 # output: Tensor([1])
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
    r"""Broadcast tensor data from root process to others.

    Args:
        inp: Input tensor.
        group: The process group to work on.
            The default group is WORLD which means all processes available.
            You can use a list of process ranks to create new group to work on it, e.g. [1, 3, 5].
        device: The specific device to execute this operator.
            None default device means the device of inp will be used.
            Specify "gpu0:1" to execute this operator on diffrent cuda stream,
            1 is stream id, and default stream id is 0.

    Returns:
        Result tensor.

    Examples:

        .. code-block::

           input = Tensor([rank])
           # Rank 0 # input: Tensor([0])
           # Rank 1 # input: Tensor([1])
           output = broadcast(input)
           # Rank 0 # output: Tensor([0])
           # Rank 1 # output: Tensor([0])

           input = Tensor([rank])
           group = Group([1, 0]) # first rank is root
           output = broadcast(input, group)
           # Rank 0 # output: Tensor([1])
           # Rank 1 # output: Tensor([1])
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
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = None, axis=0,
) -> Tensor:
    r"""Gather tensors across the specified group and concat them at first dimension.

    Args:
        inp: Input tensor.
        group: The process group to work on.
            The default group is WORLD which means all processes available.
            You can use a list of process ranks to create new group to work on it, e.g. [1, 3, 5].
        device: The specific device to execute this operator.
            None default device means the device of inp will be used.
            Specify "gpu0:1" to execute this operator on diffrent cuda stream,
            1 is stream id, and default stream id is 0.
        axis: The concat axis for collective_comm result
            The default axis is 0

    Returns:
        Result tensor.

    Examples:

        .. code-block::

           input = Tensor([rank])
           # Rank 0 # input: Tensor([0])
           # Rank 1 # input: Tensor([1])
           output = all_gather(input)
           # Rank 0 # output: Tensor([0 1])
           # Rank 1 # output: Tensor([0 1])

           input = Tensor([rank])
           group = Group([1, 0])
           output = all_gather(input, group)
           # Rank 0 # output: Tensor([1 0])
           # Rank 1 # output: Tensor([1 0])
    """
    mode = CollectiveComm.Mode.ALL_GATHER
    out = collective_comm(inp, mode, group, device)
    if axis == 0:
        return out
    else:
        group_size = group.size if group is not None else 1
        transformed_shape = list(inp._tuple_shape)
        transformed_shape[axis] *= group_size
        n, *shp = out._tuple_shape
        index = (
            [_ for _ in range(1, axis)]
            + [axis, 0]
            + [_ for _ in range(axis + 1, out.ndim + 1)]
        )
        return (
            out.reshape(group_size, n // group_size, *shp)
            .transpose(index)
            .reshape(transformed_shape)
        )


def reduce_scatter_sum(
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = None, axis=0
) -> Tensor:
    r"""Reduce tensors across the specified group by sum and split them at first dimension.

    Args:
        inp: Input tensor.
        group: The process group to work on.
            The default group is WORLD which means all processes available.
            You can use a list of process ranks to create new group to work on it, e.g. [1, 3, 5].
        device: The specific device to execute this operator.
            None default device means the device of inp will be used.
            Specify "gpu0:1" to execute this operator on diffrent cuda stream,
            1 is stream id, and default stream id is 0.
        axis: The split axis for collective_comm result
            The default axis is 0, the data will split in the 0 axis

    Returns:
        Split tensor.

    Examples:

        .. code-block::

           input = Tensor([0 1])
           # Rank 0 # input: Tensor([0 1])
           # Rank 1 # input: Tensor([0 1])
           output = reduce_scatter_sum(input)
           # Rank 0 # output: Tensor([0])
           # Rank 1 # output: Tensor([2])

           input = Tensor([0 1])
           group = Group([1, 0])
           output = reduce_scatter_sum(input, group)
           # Rank 0 # output: Tensor([2])
           # Rank 1 # output: Tensor([0])
    """
    group_size = group.size if group is not None else 1
    assert (
        list(inp._tuple_shape)[axis] % group_size == 0
    ), "current axis: {} can't devided by group size".format(axis)
    if axis != 0:
        k_new_shape = list(inp._tuple_shape)
        k_new_shape[axis] //= group_size
        k_new_shape[0] *= group_size
        new_shape = list(inp._tuple_shape)
        new_shape[axis] //= group_size
        new_shape.insert(axis, group_size)
        index = (
            [axis]
            + [_ for _ in range(0, axis)]
            + [_ for _ in range(axis + 1, inp.ndim + 1)]
        )
        inp = inp.reshape(new_shape).transpose(index).reshape(k_new_shape)
    mode = CollectiveComm.Mode.REDUCE_SCATTER_SUM
    return collective_comm(inp, mode, group, device)


def all_reduce_sum(
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = None,
) -> Tensor:
    r"""Reduce tensors across the specified group by sum.

    Args:
        inp: Input tensor.
        group: The process group to work on.
            The default group is WORLD which means all processes available.
            You can use a list of process ranks to create new group to work on it, e.g. [1, 3, 5].
        device: The specific device to execute this operator.
            None default device means the device of inp will be used.
            Specify "gpu0:1" to execute this operator on diffrent cuda stream,
            1 is stream id, and default stream id is 0.

    Returns:
        Result tensor.

    Examples:

        .. code-block::

           input = Tensor(rank)
           # Rank 0 # input: Tensor(0)
           # Rank 1 # input: Tensor(1)
           output = all_reduce_sum(input)
           # Rank 0 # output: Tensor(1)
           # Rank 1 # output: Tensor(1)
    """
    mode = CollectiveComm.Mode.ALL_REDUCE_SUM
    return collective_comm(inp, mode, group, device)


def all_reduce_max(
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = None,
) -> Tensor:
    r"""Reduce tensors across the specified group by max.

    Args:
        inp: Input tensor.
        group: The process group to work on.
            The default group is WORLD which means all processes available.
            You can use a list of process ranks to create new group to work on it, e.g. [1, 3, 5].
        device: The specific device to execute this operator.
            None default device means the device of inp will be used.
            Specify "gpu0:1" to execute this operator on diffrent cuda stream,
            1 is stream id, and default stream id is 0.

    Returns:
        Result tensor.

    Examples:

        .. code-block::

           input = Tensor(rank)
           # Rank 0 # input: Tensor(0)
           # Rank 1 # input: Tensor(1)
           output = all_reduce_max(input)
           # Rank 0 # output: Tensor(1)
           # Rank 1 # output: Tensor(1)
    """
    mode = CollectiveComm.Mode.ALL_REDUCE_MAX
    return collective_comm(inp, mode, group, device)


def all_reduce_min(
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = None,
) -> Tensor:
    r"""Reduce tensors across the specified group by min.

    Args:
        inp: Input tensor.
        group: The process group to work on.
            The default group is WORLD which means all processes available.
            You can use a list of process ranks to create new group to work on it, e.g. [1, 3, 5].
        device: The specific device to execute this operator.
            None default device means the device of inp will be used.
            Specify "gpu0:1" to execute this operator on diffrent cuda stream,
            1 is stream id, and default stream id is 0.

    Returns:
        Result tensor.

    Examples:

        .. code-block::

           input = Tensor(rank)
           # Rank 0 # input: Tensor(0)
           # Rank 1 # input: Tensor(1)
           output = all_reduce_min(input)
           # Rank 0 # output: Tensor(0)
           # Rank 1 # output: Tensor(0)
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
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = None, axis=0,
) -> Tensor:
    r"""Gather tensors across the specified group.
    Only root process will receive the final result.

    Args:
        inp: Input tensor.
        group: The process group to work on.
            The default group is WORLD which means all processes available.
            You can use a list of process ranks to create new group to work on it, e.g. [1, 3, 5].
        device: The specific device to execute this operator.
            None default device means the device of inp will be used.
            Specify "gpu0:1" to execute this operator on diffrent cuda stream,
            1 is stream id, and default stream id is 0.
        axis: The concat axis for collective_comm result

    Examples:

        .. code-block::

           input = Tensor([rank])
           # Rank 0 # input: Tensor([0])
           # Rank 1 # input: Tensor([1])
           output = gather(input)
           # Rank 0 # output: Tensor([0 1])
           # Rank 1 # output: None

           input = Tensor([rank])
           group = Group([1, 0]) # first rank is root
           output = gather(input, group)
           # Rank 0 # output: None
           # Rank 1 # output: Tensor([1 0])
    """
    assert (
        axis < inp.ndim
    ), "your concat_axis exceeds the dim of the tensor, the tensor shape is {}".format(
        inp.shape
    )

    op = _Gather(group, device)
    (out,) = apply(op, inp)

    if group.rank == 0:
        if axis == 0:
            return out
        else:
            group_size = group.size
            transformed_shape = list(inp._tuple_shape)
            transformed_shape[axis] *= group_size
            n, *shp = out._tuple_shape
            index = (
                [_ for _ in range(1, axis)]
                + [axis, 0]
                + [_ for _ in range(axis + 1, out.ndim + 1)]
            )
            return (
                out.reshape(group_size, n // group_size, *shp)
                .transpose(index)
                .reshape(transformed_shape)
            )
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
    inp: Tensor, group: Optional[Group] = WORLD, device: Optional[str] = None, axis=0,
) -> Tensor:
    r"""Split tensor in root process at first dimension.

    Args:
        inp: Input tensor.
        group: The process group to work on.
            The default group is WORLD which means all processes available.
            You can use a list of process ranks to create new group to work on it, e.g. [1, 3, 5].
        device: The specific device to execute this operator.
            None default device means the device of inp will be used.
            Specify "gpu0:1" to execute this operator on diffrent cuda stream,
            1 is stream id, and default stream id is 0.
        axis: The concat axis for collective_comm result
            The default axis is 0

    Returns:
        Split tensor.

    Examples:

        .. code-block::

           input = Tensor([0 1]) + rank*2
           # Rank 0 # input: Tensor([0 1])
           # Rank 1 # input: Tensor([2 3])
           output = scatter(input)
           # Rank 0 # output: Tensor([0])
           # Rank 1 # output: Tensor([1])

           input = Tensor([0 1]) + rank*2
           group = Group([1, 0]) # first rank is root
           output = scatter(input, group)
           # Rank 0 # output: Tensor([3])
           # Rank 1 # output: Tensor([2])
    """
    shape, dtype = _bcast_shape_dtype(group, inp)
    if group.rank != 0:
        # dummy input to infer shape
        inp = _dummy_input(shape, dtype, device)

    _bcast_tracer_state(group, inp)

    assert (
        list(inp._tuple_shape)[axis] % group.size == 0
    ), "current axis: {} can't devided by group size".format(axis)

    if axis != 0:
        group_size = group.size
        k_new_shape = list(inp._tuple_shape)
        k_new_shape[axis] //= group_size
        k_new_shape[0] *= group_size
        new_shape = list(inp._tuple_shape)
        new_shape[axis] //= group_size
        new_shape.insert(axis, group_size)
        index = (
            [axis]
            + [_ for _ in range(0, axis)]
            + [_ for _ in range(axis + 1, inp.ndim + 1)]
        )
        inp = inp.reshape(new_shape).transpose(index).reshape(k_new_shape)
    op = _Scatter(group, device)
    (out,) = apply(op, inp)
    return out


def all_to_all(
    inp: Tensor,
    group: Optional[Group] = WORLD,
    device: Optional[str] = None,
    split_axis: int = 0,
    concat_axis: int = 0,
) -> Tensor:
    r"""Each process scatter input tensor to all processes and return gathered tensor.

    Args:
        inp: Input tensor.
        group: The process group to work on.
            The default group is WORLD which means all processes available.
            You can use a list of process ranks to create new group to work on it, e.g. [1, 3, 5].
        device: The specific device to execute this operator.
            None default device means the device of inp will be used.
            Specify "gpu0:1" to execute this operator on diffrent cuda stream,
            1 is stream id, and default stream id is 0.
        split_axis: The axis that collectivecomm will split data
            the default axis is 0

    Returns:
        Result tensor.

    Examples:

        .. code-block::

           input = Tensor([0 1]) + rank*2
           # Rank 0 # input: Tensor([0 1])
           # Rank 1 # input: Tensor([2 3])
           output = all_to_all(input)
           # Rank 0 # output: Tensor([0 2])
           # Rank 1 # output: Tensor([1 3])

           input = Tensor([0 1]) + rank*2
           group = Group([1, 0])
           output = all_to_all(input, group)
           # Rank 0 # output: Tensor([0 3])
           # Rank 1 # output: Tensor([2 1])
    """
    group_size = group.size if group is not None else 1
    assert (
        list(inp._tuple_shape)[split_axis] % group_size == 0
    ), "current axis: {} can't devided by group size".format(split_axis)
    origin_shape = inp._tuple_shape
    if split_axis != 0:
        k_new_shape = list(inp._tuple_shape)
        k_new_shape[split_axis] //= group_size
        k_new_shape[0] *= group_size
        new_shape = list(inp._tuple_shape)
        new_shape[split_axis] //= group_size
        new_shape.insert(split_axis, group_size)
        index = (
            [split_axis]
            + [_ for _ in range(0, split_axis)]
            + [_ for _ in range(split_axis + 1, inp.ndim + 1)]
        )
        inp = inp.reshape(new_shape).transpose(index).reshape(k_new_shape)

    mode = CollectiveComm.Mode.ALL_TO_ALL
    out = collective_comm(inp, mode, group, device)

    if concat_axis == 0:
        return out

    transformed_shape = list(origin_shape)
    transformed_shape[concat_axis] *= group_size
    transformed_shape[split_axis] //= group_size

    n, *shp = out._tuple_shape
    index = (
        [_ for _ in range(1, concat_axis)]
        + [concat_axis, 0]
        + [_ for _ in range(concat_axis + 1, out.ndim + 1)]
    )
    return (
        out.reshape(group_size, n // group_size, *shp)
        .transpose(index)
        .reshape(transformed_shape)
    )


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
    r"""Send tensor to another process.

    Args:
        inp: Tensor to send.
        dest_rank: Rank of destination process.

    Returns:
        None.

    Examples:
        .. code-block::

           if rank == 0:
               data = mge.tensor(1)
               # Tensor(1)
               F.distributed.remote_send(data, 1) # return None
           else:
               data = F.distributed.remote_recv(0)
               # Tensor(1)
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
    r"""Receive a tensor from another process.

    Args:
        src_rank: Rank of source process.
        device: The specific device to execute this operator.
            None default device means the device of inp will be used.
            Specify "gpu0:1" to execute this operator on diffrent cuda stream,
            1 is stream id, and default stream id is 0.
        inp: Dummy input to determine received tensor type.

    Returns:
        Received tensor.

    Examples:

    .. code-block::

       if rank == 0:
           data = mge.tensor(1)
           # Tensor(1)
           F.distributed.remote_send(data, 1) # return None
       else:
           data = F.distributed.remote_recv(0)
           # Tensor(1)
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
