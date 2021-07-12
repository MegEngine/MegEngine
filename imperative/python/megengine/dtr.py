# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import re
from typing import Union

from mprop import mproperty

from .core._imperative_rt.core2 import set_option as _set_option
from .core._imperative_rt.utils import _set_defrag

_eviction_threshold = 0
_evictee_minimum_size = 1024 ** 2


def _str2bytes(text: str) -> int:
    regex = re.compile(r"(\d+(?:\.\d+)?)\s*([kmg]?b)", re.IGNORECASE)
    order = ["b", "kb", "mb", "gb"]
    result = regex.findall(text)
    if len(result) != 1:
        raise ValueError(
            "Formatting of `value` only supports bytes(B), kilobyte(KB), megabyte(MB) and gigabyte(GB) units"
        )
    return int(float(result[0][0]) * 1024 ** order.index(result[0][1].lower()))


@mproperty
def eviction_threshold(mod):
    r"""
    Get or set the eviction threshold in bytes. It can also be set to a string,
    whose formatting supports byte(B), kilobyte(KB), megabyte(MB) and
    gigabyte(GB) units.

    .. note::

       When GPU memory usage exceeds this value, DTR will heuristically select
       and evict resident tensors until the amount of used memory falls below
       this threshold.

    Examples:

    .. code-block::

        import megengine as mge
        mge.dtr.eviction_threshold = "2GB"

    """
    return mod._eviction_threshold


@eviction_threshold.setter
def eviction_threshold(mod, value: Union[int, str]):
    if isinstance(value, str):
        mod._eviction_threshold = mod._str2bytes(value)
    elif isinstance(value, int):
        mod._eviction_threshold = value
    else:
        raise TypeError("`value` should be a str or an int")
    _set_option("dtr_eviction_threshold", mod._eviction_threshold)


@mproperty
def evictee_minimum_size(mod):
    r"""
    Get or set the memory threshold of tensors in bytes. It can also be set to a
    string, whose formatting supports byte(B), kilobyte(KB), megabyte(MB) and
    gigabyte(GB) units.

    .. note::

       Only tensors whose size exceeds this threshold will be added to the
       candidate set. A tensor that is not added to the candidate set will
       never be evicted during its lifetime.

    Examples:

    .. code-block::

        import megengine as mge
        mge.dtr.evictee_minimum_size = "2MB"

    """
    return mod._evictee_minimum_size


@evictee_minimum_size.setter
def evictee_minimum_size(mod, value: Union[int, str]):
    if isinstance(value, str):
        mod._evictee_minimum_size = mod._str2bytes(value)
    elif isinstance(value, int):
        mod._evictee_minimum_size = value
    else:
        raise TypeError("`value` should be a str or an int")
    _set_option("dtr_evictee_minimum_size", mod._evictee_minimum_size)


def enable():
    r"""
    Enable to record computing path of tensors and to perform DTR policy.
    """
    _set_defrag(True)
    _set_option("enable_dtr_auto_drop", 1)
    _set_option("enable_drop", 1)
    _set_option("buffer_length", 0)
    _set_option("record_computing_path", 1)


def disable():
    r"""
    Stop recording computing path of tensors and performing DTR policy.
    """
    _set_defrag(False)
    _set_option("enable_dtr_auto_drop", 0)
    _set_option("enable_drop", 0)
    _set_option("record_computing_path", 0)
