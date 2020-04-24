# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import os

import megengine._internal as mgb

_default_device = os.getenv("MGE_DEFAULT_DEVICE", "xpux")


def get_device_count(device_type: str) -> int:
    """Gets number of devices installed on this system.

    :param device_type: device type, one of 'gpu' or 'cpu'
    """

    device_type_set = ("cpu", "gpu")
    assert device_type in device_type_set, "device must be one of {}".format(
        device_type_set
    )
    return mgb.config.get_device_count(device_type)


def is_cuda_available() -> bool:
    """Returns whether cuda device is available on this system.

    """
    return mgb.config.get_device_count("gpu", warn=False) > 0


def set_default_device(device: str = "xpux"):
    r"""Sets default computing node.

    :param device: default device type. The type can be 'cpu0', 'cpu1', etc.,
        or 'gpu0', 'gpu1', etc., to specify the particular cpu or gpu to use.
        'cpux' and  'gupx' can also be used to specify any number of cpu or gpu devices.

        'multithread' device type is avaliable when inference, which implements
        multi-threading parallelism at the operator level. For example,
        'multithread4' will compute with 4 threads. which implements

        The default value is 'xpux' to specify any device available.

        It can also be set by environmental variable `MGE_DEFAULT_DEVICE`.
    """
    global _default_device  # pylint: disable=global-statement
    _default_device = device


def get_default_device() -> str:
    r"""Gets default computing node.

    It returns the value set by :func:`~.set_default_device`.
    """
    return _default_device
