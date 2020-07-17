# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from enum import Enum
from functools import partial, update_wrapper, wraps


def register_method_to_class(cls):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        if isinstance(func, partial):
            update_wrapper(func, func.func)
        setattr(cls, func.__name__, wrapper)
        return func

    return decorator


class QuantMode(Enum):
    SYMMERTIC = 1
    ASYMMERTIC = 2
    TQT = 3


qparam_dict = {
    QuantMode.SYMMERTIC: {"mode": QuantMode.SYMMERTIC, "scale": None,},
    QuantMode.ASYMMERTIC: {
        "mode": QuantMode.ASYMMERTIC,
        "scale": None,
        "zero_point": None,
    },
    QuantMode.TQT: {"mode": QuantMode.TQT, "scale": None,},
}


def get_qparam_dict(mode):
    return qparam_dict.get(mode, None)
