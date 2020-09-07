# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import functools
import multiprocessing as mp
from collections import defaultdict
from typing import Callable

import numpy as np

from megengine.autodiff.grad_manager import GradManager, get_backwarding_grad_manager
from megengine.device import get_default_device, get_device_count

from ..functional.param_pack import get_offsets, pack_allreduce_split
from ..functional.utils import copy
from .functional import all_reduce_sum, broadcast
from .group import WORLD, group_barrier, is_distributed
from .util import Future


class FakeTensor(Future):
    def device(self):
        raise "Sorry, this tensor is not ready"

    def numpy(self):
        raise "Sorry, this tensor is not ready"

    def shape(self):
        raise "Sorry, this tensor is not ready"

    def dtype(self):
        raise "Sorry, this tensor is not ready"


def synchronized(func: Callable):
    """Decorator. Decorated function will synchronize when finished.
    Specifically, we use this to prevent data race during hub.load"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not is_distributed():
            return func(*args, **kwargs)

        ret = func(*args, **kwargs)
        group_barrier()
        return ret

    return wrapper


def get_device_count_by_fork(device_type: str):
    q = mp.Queue()

    def worker(queue):
        num = get_device_count(device_type)
        queue.put(num)

    p = mp.Process(target=worker, args=(q,))
    p.start()
    p.join()
    return q.get()


def bcast_params_(params, group):
    for p in params:
        p._reset(broadcast(p, group))


class AllreduceCallback:
    def __init__(self, reduce_method, group=WORLD):
        reduce_method = reduce_method.lower()
        assert reduce_method in ["sum", "mean"]
        self._reduce_method = reduce_method
        self._group = group
        self._gm_set = set()
        self._param_pack_thd = 10 * 1024 * 1024
        self._reset()

    def _reset(self):
        self._params = []
        self._gradients_dict = dict()
        self._futures_dict = dict()
        self._packing_list = defaultdict(list)
        self._packing_size = defaultdict(int)

    def _pack(self, dtype):
        grad_list = [self._gradients_dict[p] for p in self._packing_list[dtype]]
        shapes = [p.shape for p in self._packing_list[dtype]]
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
        if gm not in self._gm_set:
            gm.register_after_backward_callback(self._flush)
            self._gm_set.add(gm)
        self._params.append(param)
        self._futures_dict[param] = FakeTensor(ack=False)
        self._gradients_dict[param] = grad

        self._packing_list[param.dtype].append(param)
        self._packing_size[param.dtype] += (
            int(np.prod(list(param.shape))) * np.dtype(param.dtype).itemsize
        )
        if self._packing_size[param.dtype] > self._param_pack_thd:
            self._pack(param.dtype)
        return self._futures_dict[param]

    def _flush(self):
        for dtype in self._packing_list.keys():
            self._pack(dtype)
        for param in self._params:
            grad = self._gradients_dict[param]
            grad = copy(grad, get_default_device())
            self._futures_dict[param].set(grad)
        self._reset()


make_allreduce_cb = AllreduceCallback
