# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import functools
from typing import Callable

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
