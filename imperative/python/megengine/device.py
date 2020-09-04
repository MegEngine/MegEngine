# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import os

from .core._imperative_rt.common import CompNode, DeviceType

__all__ = [
    "is_cuda_available",
    "get_device_count",
    "get_default_device",
    "set_default_device",
]


def _valid_device(inp):
    if isinstance(inp, str) and len(inp) == 4:
        if inp[0] in {"x", "c", "g"} and inp[1:3] == "pu":
            if inp[3] == "x" or inp[3].isdigit():
                return True
    return False


def _str2device_type(type_str: str, allow_unspec: bool = True):
    type_str = type_str.upper()
    if type_str == "CPU":
        return DeviceType.CPU
    elif type_str == "GPU" or type_str == "CUDA":
        return DeviceType.CUDA
    else:
        assert allow_unspec and str == "XPU", "bad device type"
        return DeviceType.UNSPEC


def get_device_count(device_type: str) -> int:
    """Gets number of devices installed on this system.

    :param device_type: device type, one of 'gpu' or 'cpu'
    """

    device_type_set = ("cpu", "gpu")
    assert device_type in device_type_set, "device must be one of {}".format(
        device_type_set
    )
    device_type = _str2device_type(device_type)
    return CompNode._get_device_count(device_type, False)


def is_cuda_available() -> bool:
    """Returns whether cuda device is available on this system.

    """
    t = _str2device_type("gpu")
    return CompNode._get_device_count(t, False) > 0


def set_default_device(device: str = "xpux"):
    r"""Sets default computing node.

    :param device: default device type. The type can be 'cpu0', 'cpu1', etc.,
        or 'gpu0', 'gpu1', etc., to specify the particular cpu or gpu to use.
        'cpux' and  'gpux' can also be used to specify any number of cpu or gpu devices.

        'multithread' device type is avaliable when inference, which implements
        multi-threading parallelism at the operator level. For example,
        'multithread4' will compute with 4 threads. which implements

        The default value is 'xpux' to specify any device available.

        It can also be set by environmental variable `MGE_DEFAULT_DEVICE`.
    """
    assert _valid_device(device), "Invalid device name {}".format(device)
    CompNode._set_default_device(device)


def get_default_device() -> str:
    r"""Gets default computing node.

    It returns the value set by :func:`~.set_default_device`.
    """
    return CompNode._get_default_device()


set_default_device(os.getenv("MGE_DEFAULT_DEVICE", "xpux"))
