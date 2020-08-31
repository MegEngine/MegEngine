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

from .group import group_barrier, is_distributed


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
