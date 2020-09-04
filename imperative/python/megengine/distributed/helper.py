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
from typing import Callable

from megengine.device import get_device_count

from .functional import all_reduce_sum, broadcast
from .group import WORLD, group_barrier, is_distributed


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
        self._reduce_method = reduce_method
        self._group = group

    def __call__(self, param, grad):
        ret = all_reduce_sum(grad, self._group)
        if self._reduce_method == "MEAN":
            ret = ret / self._group.size
        return ret


make_allreduce_cb = AllreduceCallback
