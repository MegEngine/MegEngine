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

# use "default" to distinguish it from None in _reset_execution_config
__compute_mode = "default"
_benchmark_kernel = False
_deterministic_kernel = False
_benchmark_with_subprocess = False

__all__ = [
    "benchmark_kernel",
    "benchmark_with_subprocess",
    "deterministic_kernel",
    "async_level",
    "disable_memory_forwarding",
    "_compute_mode",
    "_auto_format_convert",
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
def benchmark_with_subprocess(mod):
    r"""Whether or not run possible algorithms on real device to find the best one. The default option is false,
    which means use heuristic to choose the fastest algorithm.
    
    Examples:    
        .. code-block::

           import megengine as mge
           mge.config.benchmark_with_subprocess = True
    """
    return _benchmark_with_subprocess


@benchmark_with_subprocess.setter
def benchmark_with_subprocess(mod, option: bool):
    if option:
        import sys
        from ._imperative_rt.utils import _set_fork_exec_path_for_timed_func

        _set_fork_exec_path_for_timed_func(
            sys.executable,
            os.path.join(
                os.path.dirname(__file__), "../utils", "_timed_func_fork_exec_entry.py"
            ),
        )


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
    r"""Get or set the precision of intermediate results for conv, matmul. The default
    option is None and will fallback to "default". When set to "float32", it will
    trigger mixed precision computation on TensorCore, but only effective when input and
    output are of float16 dtype.

    Examples:

        .. code-block::

           import megengine as mge
           mge.config._compute_mode = "float32"
    """
    return __compute_mode


@_compute_mode.setter
def _compute_mode(mod, _compute_mode: str):
    global __compute_mode
    __compute_mode = _compute_mode


@property
def _bn_format(mod):
    r"""Get or set batchnorm param layout format. The default option is None and will
    fallback to "dim_1c11" which corresponds to NCHW format. When set to "dim_111c",
    param format of batchnorm will be changed to NHWC.

    Examples:

        .. code-block::

           import megengine as mge
           mge.config._bn_format = "dim_111c"
    """
    return __bn_format


@_bn_format.setter
def _bn_format(mod, format: str):
    global __bn_format
    __bn_format = format


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
):
    global _benchmark_kernel, _deterministic_kernel, __compute_mode
    orig_flags = (
        _benchmark_kernel,
        _deterministic_kernel,
        get_option("async_level"),
        __compute_mode,
    )
    if benchmark_kernel is not None:
        _benchmark_kernel = benchmark_kernel
    if deterministic_kernel is not None:
        _deterministic_kernel = deterministic_kernel
    if async_level is not None:
        set_option("async_level", async_level)
    if compute_mode is not None:
        __compute_mode = compute_mode

    return orig_flags


@contextmanager
def _override(
    benchmark_kernel=None,
    deterministic_kernel=None,
    async_level=None,
    compute_mode=None,
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
            )
           def train():
    """
    orig_flags = _reset_execution_config(
        benchmark_kernel=benchmark_kernel,
        deterministic_kernel=deterministic_kernel,
        async_level=async_level,
        compute_mode=compute_mode,
    )
    try:
        yield
    finally:
        # recover the previous values
        _reset_execution_config(*orig_flags)


def _get_actual_op_param(function_param, config_param):
    return function_param if config_param == "default" else config_param
