# -*- coding: utf-8 -*-
import os
from contextlib import contextmanager

from ._imperative_rt.core2 import (
    _clear_algorithm_cache,
    get_auto_format_convert,
    get_option,
    set_auto_format_convert,
    set_option,
)

__compute_mode = "default"
__conv_format = "default"
_benchmark_kernel = False
_deterministic_kernel = False

__all__ = [
    "benchmark_kernel",
    "deterministic_kernel",
    "async_level",
    "disable_memory_forwarding",
    "_compute_mode",
    "_conv_format",
    "_override",
]


@property
def benchmark_kernel(mod):
    r"""Whether or not run possible algorithms on real device to find the best one. The default option is false,
    which means use heuristic to choose the fastest algorithm.

    Examples:
        .. code-block::

           import megengine as mge
           mge.config.benchmark_kernel = True
    """
    return _benchmark_kernel


@benchmark_kernel.setter
def benchmark_kernel(mod, option: bool):
    global _benchmark_kernel
    # try different strategy, then clear algorithm cache
    if option != _benchmark_kernel:
        _clear_algorithm_cache()
    _benchmark_kernel = option


@property
def deterministic_kernel(mod):
    r"""Whether or not the fastest algorithm choosed is reproducible. The default option is false,
    which means the algorithm is not reproducible.

    Examples:
        .. code-block::

           import megengine as mge
           mge.config.deterministic_kernel = True
    """
    return _deterministic_kernel


@deterministic_kernel.setter
def deterministic_kernel(mod, option: bool):
    global _deterministic_kernel
    _deterministic_kernel = option


@property
def async_level(mod) -> int:
    r"""Get or set config whether raise error exactly when invoking op. The default level is 2,
    which means both device and user side errors are async.

    Examples:
        .. code-block::

           import megengine as mge
           mge.config.async_level = 2
    """
    return get_option("async_level")


@async_level.setter
def async_level(mod, level: int):
    assert level >= 0 and level <= 2, "async_level should be 0, 1 or 2"
    set_option("async_level", level)


@property
def disable_memory_forwarding(mod) -> bool:
    r"""Get or set config whether to disable memory forwarding. The default option is false, 
    which means storage may be shared among tensors.
    
    Examples:    
        .. code-block::

           import megengine as mge
           mge.config.disable_memory_forwarding = False
    """
    return bool(get_option("disable_memory_forwarding"))


@disable_memory_forwarding.setter
def disable_memory_forwarding(mod, disable: bool):
    set_option("disable_memory_forwarding", disable)


@property
def _compute_mode(mod):
    r"""Get or set the precision of intermediate results. The default option is "default",
    which means that no special requirements will be placed on.  When set to 'float32', it
    would be used for accumulator and intermediate result, but only effective when input and 
    output are of float16 dtype.

    Examples:
        .. code-block::

           import megengine as mge
           mge.config._compute_mode = "default"
    """
    return __compute_mode


@_compute_mode.setter
def _compute_mode(mod, _compute_mode: str):
    global __compute_mode
    __compute_mode = _compute_mode


@property
def _conv_format(mod):
    r"""Get or set convolution data/filter/output layout format. The default option is "default",
    which means that no special format will be placed on. There are all layout definitions

    ``NCHW`` layout: ``{N, C, H, W}``
    ``NHWC`` layout: ``{N, H, W, C}``
    ``NHWCD4`` layout: ``{N, H, (C + 3) / 4, W, 4}``
    ``NHWCD4I`` layout: with ``align_axis = 2``
    ``NCHW4`` layout: ``{N, C/4, H, W, 4}``
    ``NCHW88`` layout: ``{N, C/8, H, W, 8}``
    ``CHWN4`` layout: ``{C/4, H, W, N, 4}``
    ``NCHW64`` layout: ``{N, C/64, H, W, 64}``

    Examples:
        .. code-block::

           import megengine as mge
           mge.config._conv_format = "NHWC"
    """
    return __conv_format


@_conv_format.setter
def _conv_format(mod, format: str):
    global __conv_format
    __conv_format = format


@property
def _auto_format_convert(mod):
    r"""Automatically convert indexing params' order for NCHW Tensor to NHWC order.
    The default value is False, which means no convert.

    Examples:
        .. code-block::

           import megengine as mge
           mge.config._auto_format_convert = True
    """
    return get_auto_format_convert()


@_auto_format_convert.setter
def _auto_format_convert(mod, option: bool):
    set_auto_format_convert(option)


def _reset_execution_config(
    benchmark_kernel=None,
    deterministic_kernel=None,
    async_level=None,
    compute_mode=None,
    conv_format=None,
    auto_format_convert=None,
):
    global _benchmark_kernel, _deterministic_kernel, __compute_mode, __conv_format
    orig_flags = (
        _benchmark_kernel,
        _deterministic_kernel,
        get_option("async_level"),
        __compute_mode,
        __conv_format,
        get_auto_format_convert(),
    )
    if benchmark_kernel is not None:
        _benchmark_kernel = benchmark_kernel
    if deterministic_kernel is not None:
        _deterministic_kernel = deterministic_kernel
    if async_level is not None:
        set_option("async_level", async_level)
    if compute_mode is not None:
        __compute_mode = compute_mode
    if conv_format is not None:
        __conv_format = conv_format
    if auto_format_convert is not None:
        set_auto_format_convert(auto_format_convert)

    return orig_flags


@contextmanager
def _override(
    benchmark_kernel=None,
    deterministic_kernel=None,
    async_level=None,
    compute_mode=None,
    conv_format=None,
    auto_format_convert=None,
):
    r"""A context manager that users can opt in by attaching the decorator to set 
    the config of the global variable.

    Examples:
        .. code-block::

           import megengine as mge

           @mge.config._override(
                benchmark_kernel = True,
                deterministic_kernel = Fasle,
                async_level=2,
                compute_mode="float32",
                conv_format="NHWC",
                auto_format_convert=True,
            )
           def train():
    """
    orig_flags = _reset_execution_config(
        benchmark_kernel,
        deterministic_kernel,
        async_level,
        compute_mode,
        conv_format,
        auto_format_convert,
    )
    try:
        yield
    finally:
        # recover the previous values
        _reset_execution_config(*orig_flags)


def _get_actual_op_param(function_param, config_param):
    return function_param if config_param == "default" else config_param
