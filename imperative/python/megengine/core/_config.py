# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import os
from contextlib import contextmanager

__compute_mode = "default"
__conv_format = "default"
_benchmark_kernel = False
_deterministic_kernel = False
_async_level = os.getenv("MEGENGINE_INTERP_ASYNC_LEVEL", 2)


__all__ = [
    "benchmark_kernel",
    "deterministic_kernel",
    "async_level",
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
    return _async_level


@async_level.setter
def async_level(mod, level: int):
    global _async_level
    _async_level = level


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


def _reset_execution_config(
    benchmark_kernel=None,
    deterministic_kernel=None,
    async_level=None,
    compute_mode=None,
    conv_format=None,
):
    global _benchmark_kernel, _deterministic_kernel, _async_level, __compute_mode, __conv_format
    orig_flags = (
        _benchmark_kernel,
        _deterministic_kernel,
        _async_level,
        __compute_mode,
        __conv_format,
    )
    if benchmark_kernel is not None:
        _benchmark_kernel = benchmark_kernel
    if deterministic_kernel is not None:
        _deterministic_kernel = deterministic_kernel
    if async_level is not None:
        _async_level = async_level
    if compute_mode is not None:
        __compute_mode = compute_mode
    if conv_format is not None:
        __conv_format = conv_format

    return orig_flags


@contextmanager
def _override(
    benchmark_kernel=None,
    deterministic_kernel=None,
    async_level=None,
    compute_mode=None,
    conv_format=None,
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
            )
           def train():
    """
    orig_flags = _reset_execution_config(
        benchmark_kernel, deterministic_kernel, async_level, compute_mode, conv_format,
    )
    try:
        yield
    finally:
        # recover the previous values
        _reset_execution_config(*orig_flags)


def _get_actual_op_param(function_param, config_param):
    return function_param if config_param == "default" else config_param
