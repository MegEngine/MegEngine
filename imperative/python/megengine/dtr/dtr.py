# -*- coding: utf-8 -*-
import re
from typing import Union

from ..core import set_option as _set_option
from ..core._imperative_rt.core2 import clear_candidates as _clear_candidates

_eviction_threshold = 0
_evictee_minimum_size = 1024 ** 2
_enable_sqrt_sampling = False


def _str2bytes(text: str) -> int:
    regex = re.compile(r"(\d+(?:\.\d+)?)\s*([kmg]?b)", re.IGNORECASE)
    order = ["b", "kb", "mb", "gb"]
    result = regex.findall(text)
    if len(result) != 1:
        raise ValueError(
            "Formatting of `value` only supports bytes(B), kilobyte(KB), megabyte(MB) and gigabyte(GB) units"
        )
    return int(float(result[0][0]) * 1024 ** order.index(result[0][1].lower()))


@property
def eviction_threshold(mod):
    r"""Get or set the eviction threshold in bytes. It can also be set to a string,
    whose formatting supports byte(B), kilobyte(KB), megabyte(MB) and
    gigabyte(GB) units.
    
    Note: 
       When GPU memory usage exceeds this value, DTR will heuristically select
       and evict resident tensors until the amount of used memory falls below
       this threshold.
    
    Examples:
        .. code-block::

           import megengine as mge
           mge.dtr.eviction_threshold = "2GB"
    """
    return _eviction_threshold


@eviction_threshold.setter
def eviction_threshold(mod, value: Union[int, str]):
    global _eviction_threshold
    if isinstance(value, str):
        _eviction_threshold = _str2bytes(value)
    elif isinstance(value, int):
        _eviction_threshold = value
    else:
        raise TypeError("`value` should be a str or an int")
    _set_option("dtr_eviction_threshold", _eviction_threshold)


@property
def evictee_minimum_size(mod):
    r"""Get or set the memory threshold of tensors in bytes. It can also be set to a
    string, whose formatting supports byte(B), kilobyte(KB), megabyte(MB) and
    gigabyte(GB) units.
    
    Note:
       Only tensors whose size exceeds this threshold will be added to the
       candidate set. A tensor that is not added to the candidate set will
       never be evicted during its lifetime.
    
    Examples:
    
        .. code-block::

           import megengine as mge
           mge.dtr.evictee_minimum_size = "2MB"
    """
    return _evictee_minimum_size


@evictee_minimum_size.setter
def evictee_minimum_size(mod, value: Union[int, str]):
    global _evictee_minimum_size
    if isinstance(value, str):
        _evictee_minimum_size = _str2bytes(value)
    elif isinstance(value, int):
        _evictee_minimum_size = value
    else:
        raise TypeError("`value` should be a str or an int")
    _set_option("dtr_evictee_minimum_size", _evictee_minimum_size)


@property
def enable_sqrt_sampling(mod):
    r"""Get or set whether sqrt sampling is allowed. Sqrt sampling means that given
    the size of the candidate set is N, only enumerate sqrt(N) tensors. When
    the number of tensors is very high, enabling this optimization will speed
    up the training.
    
    Examples:    
        .. code-block::

           import megengine as mge
           mge.dtr.enable_sqrt_sampling = True
    """
    return _enable_sqrt_sampling


@enable_sqrt_sampling.setter
def enable_sqrt_sampling(mod, value: bool):
    global _enable_sqrt_sampling
    _enable_sqrt_sampling = value
    _set_option("enable_dtr_sqrt_sampling", _enable_sqrt_sampling)


def enable():
    r"""Enable to record computing path of tensors and to perform DTR policy."""
    _set_option("enable_dtr_auto_drop", 1)
    _set_option("enable_drop", 1)
    _set_option("record_computing_path", 1)


def disable():
    r"""Stop recording computing path of tensors and performing DTR policy."""
    _set_option("enable_dtr_auto_drop", 0)
    _set_option("enable_drop", 0)
    _set_option("record_computing_path", 0)
    _clear_candidates()
