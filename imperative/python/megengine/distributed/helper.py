# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import functools
import multiprocessing as mp
from collections import defaultdict
from typing import Callable
from weakref import WeakSet

import numpy as np

from megengine.autodiff.grad_manager import GradManager, get_backwarding_grad_manager
from megengine.device import get_default_device, get_device_count

from ..core._imperative_rt.core2 import apply
from ..core.ops.builtin import ParamPackConcat, ParamPackSplit
from ..functional.tensor import copy
from ..tensor import Tensor
from ..utils.future import Future
from . import group as _group
from .functional import _bcast_param, all_reduce_sum, broadcast
from .group import WORLD, Group, group_barrier, is_distributed, override_backend


def param_pack_split(inp: Tensor, offsets: list, shapes: list):
    r"""
    Returns split tensor to tensor list as offsets and shapes described,
            only used for ``parampack``.

    :param inp: input tensor.
    :param offsets: offsets of outputs, length of `2 * n`,
            while n is tensor nums you want to split,
            format `[begin0, end0, begin1, end1]`.
    :param shapes: tensor shapes of outputs.
    :return: splitted tensors.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        from megengine.distributed.helper import param_pack_split

        a = tensor(np.ones((10,), np.int32))
        b, c = param_pack_split(a, [0, 1, 1, 10], [(1,), (3, 3)])
        print(b.numpy())
        print(c.numpy())

    Outputs:

    .. testoutput::

        [1]
        [[1 1 1]
         [1 1 1]
         [1 1 1]]

    """
    op = ParamPackSplit()
    op.offsets = offsets
    op.shapes = [s or (1,) for s in shapes]
    outputs = apply(op, inp)
    for s, x in zip(shapes, outputs):
        if not s:
            x._setscalar()
    return outputs


def param_pack_concat(inps: list, offsets: Tensor, offsets_val: list):
    r"""
    Returns concated tensor, only used for ``parampack``.

    :param inps: input tensors.
    :param offsets: device value of offsets.
    :param offsets_val: offsets of inputs, length of `2 * n`,
            format `[begin0, end0, begin1, end1]`.
    :return: concated tensor.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        from megengine.distributed.helper import param_pack_concat

        a = tensor(np.ones((1,), np.int32))
        b = tensor(np.ones((3, 3), np.int32))
        offsets_val = [0, 1, 1, 10]
        offsets = tensor(offsets_val, np.int32)
        c = param_pack_concat([a, b], offsets, offsets_val)
        print(c.numpy())

    Outputs:

    .. testoutput::

        [1 1 1 1 1 1 1 1 1 1]

    """
    op = ParamPackConcat()
    op.offsets = offsets_val
    return apply(op, *inps, offsets)[0]


def get_offsets(shapes):
    offsets = []
    offset = 0
    for shape in shapes:
        offsets.append(offset)
        offset += int(np.prod(shape))
        offsets.append(offset)
    return offsets


_enable_p2p_cache = None


def _check_enable_p2p():
    global _enable_p2p_cache
    if _enable_p2p_cache is not None:
        return _enable_p2p_cache
    cmd = ["nvidia-smi", "topo", "-p2p", "w"]
    import subprocess

    output = subprocess.run(cmd, stdout=subprocess.PIPE).stdout
    if output.count(b"OK") > 1:
        _enable_p2p_cache = True
        return True
    else:
        _enable_p2p_cache = False
        return False


def pack_allreduce_split(pack_list, shapes, group, reduce_method):
    offsets_val = get_offsets(shapes)
    offsets = Tensor(offsets_val)
    packed_grads = param_pack_concat(pack_list, offsets, offsets_val)

    packed_grads = all_reduce_sum(packed_grads, group, group.comp_node)
    if reduce_method == "mean":
        packed_grads /= group.size
    grads = param_pack_split(packed_grads, offsets_val, shapes)
    return grads


class TensorFuture(Future):
    def device(self):
        raise "Sorry, this tensor is not ready"

    def numpy(self):
        raise "Sorry, this tensor is not ready"

    def shape(self):
        raise "Sorry, this tensor is not ready"

    def dtype(self):
        raise "Sorry, this tensor is not ready"


def synchronized(func: Callable):
    """
    Decorator. Decorated function will synchronize when finished.
    Specifically, we use this to prevent data race during hub.load"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not is_distributed():
            return func(*args, **kwargs)

        ret = func(*args, **kwargs)
        group_barrier()
        return ret

    return wrapper


def _check_device_initialized(device_type: str, rank: int):
    try:
        test = Tensor(1, device=(device_type + str(rank)))
        inited = False
        del test
    except:
        inited = True
    errmsg = "The cuda env is set before the forked thread starts. Please do not use any cuda function or variable before forking."
    if inited:
        raise RuntimeError(errmsg)


def bcast_list_(inps: list, group: Group = WORLD):
    """
    Broadcast tensors between given group.

    :param inps: input tensors.
    :param group: communication group.
    """
    for inp in inps:
        inp._reset(_bcast_param(inp, group))


class AllreduceCallback:
    """
    Allreduce Callback with tensor fusion optimization.

    :param reduce_method: the method to reduce gradiants.
    :param group: communication group.
    :param backend: override distributed backend in allreduce
    """

    def __init__(self, reduce_method: str, group: Group = WORLD, backend: str = None):
        reduce_method = reduce_method.lower()
        assert reduce_method in ["sum", "mean"], "reduce_method should be sum or mean"
        self._reduce_method = reduce_method
        self._group = group
        self._marked_gm = WeakSet()
        self._param_pack_thd = 10 * 1024 * 1024
        self._reset()
        if backend is None:
            assert _group._sd, "please call init_process_group first"
            backend = _group._sd.backend
        if backend == "auto":
            if group.is_single_machine and not _check_enable_p2p():
                backend = "shm"
            else:
                backend = "nccl"
        self._backend = backend

    def _reset(self):
        self._params = []
        self._gradients_dict = dict()
        self._futures_dict = dict()
        self._packing_list = defaultdict(list)
        self._packing_size = defaultdict(int)
        self._grad_origin_device = dict()

    def _pack(self, dtype):
        if len(self._packing_list[dtype]) == 0:
            return
        grad_list = [self._gradients_dict[p] for p in self._packing_list[dtype]]
        shapes = [p._tuple_shape for p in self._packing_list[dtype]]
        with override_backend(self._backend):
            reduced_grads = pack_allreduce_split(
                grad_list, shapes, self._group, self._reduce_method
            )
        for param, grad in zip(self._packing_list[dtype], reduced_grads):
            self._gradients_dict[param] = grad
        self._packing_list[dtype] = []
        self._packing_size[dtype] = 0

    def __call__(self, param, grad):
        gm = get_backwarding_grad_manager()
        assert isinstance(gm, GradManager)
        if gm not in self._marked_gm:
            gm._register_after_backward_callback(self._flush)
            self._marked_gm.add(gm)
        self._params.append(param)
        self._futures_dict[param] = TensorFuture(ack=False)
        self._gradients_dict[param] = grad
        self._grad_origin_device[param] = str(grad.device)

        dtype_str = str(np.dtype(param.dtype))
        dtype_size = np.dtype(param.dtype).itemsize
        self._packing_list[dtype_str].append(param)
        self._packing_size[dtype_str] += int(np.prod(param._tuple_shape)) * dtype_size
        if self._packing_size[dtype_str] > self._param_pack_thd:
            self._pack(dtype_str)
        return self._futures_dict[param]

    def _flush(self):
        for dtype in sorted(self._packing_list.keys()):
            self._pack(dtype)
        for param in self._params:
            grad = self._gradients_dict[param]
            grad = copy(grad, self._grad_origin_device[param])
            self._futures_dict[param].set(grad)
        self._reset()


make_allreduce_cb = AllreduceCallback
