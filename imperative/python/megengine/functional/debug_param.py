# -*- coding: utf-8 -*-
import os

from ..core import _config
from ..core._imperative_rt.core2 import _clear_algorithm_cache
from ..core.ops import builtin
from ..logger import get_logger
from ..utils.deprecation import deprecated

Strategy = builtin.Convolution.Strategy

if os.getenv("MEGENGINE_CONV_EXECUTION_STRATEGY") != None:
    get_logger().warning(
        "Environment variable `MEGENGINE_CONV_EXECUTION_STRATEGY` is deprecated, please use `MEGENGINE_EXECUTION_STRATEGY`"
    )

_valid_string_option = {
    "REPRODUCIBLE": Strategy.REPRODUCIBLE,
    "HEURISTIC": Strategy.HEURISTIC,
    "PROFILE": Strategy.PROFILE,
}


def get_execution_strategy() -> Strategy:
    r"""Returns the execution strategy of :class:`~.module.Conv2d` and :func:`~.matmul`

    See :func:`~.set_execution_strategy` for possible return values
    """
    strategy = Strategy(0)
    if _config._benchmark_kernel:
        strategy |= Strategy.PROFILE
    else:
        strategy |= Strategy.HEURISTIC
    if _config._deterministic_kernel:
        strategy |= Strategy.REPRODUCIBLE
    return strategy


def set_execution_strategy(option):
    r"""Sets the execution strategy of :class:`~.module.Conv2d` and :func:`~.matmul`

    Args:
        option: Decides how :class:`~.module.Conv2d` and :func:`~.matmul` algorithms are chosen.
            Available strategy values:

            * "HEURISTIC": uses heuristic to choose the fastest algorithm.
            * "PROFILE": runs possible algorithms on a real device to find the best one.
            * "REPRODUCIBLE": uses algorithms that are reproducible.

    The default strategy is "HEURISTIC", these options can be combined to
    form a combination option, e.g. PROFILE_REPRODUCIBLE is a combination
    of "PROFILE" and "REPRODUCIBLE", which means using the fastest profiling
    result that is also reproducible.

    Available values string:

    * "HEURISTIC" uses heuristic to choose the fastest algorithm.
    * "PROFILE" runs possible algorithms on a real device to find the best one.
    * "PROFILE_REPRODUCIBLE" uses the fastest profiling result that is also reproducible.
    * "HEURISTIC_REPRODUCIBLE" uses heuristic to choose the fastest algorithm that is also reproducible.

    The default strategy is "HEURISTIC".

    It can also be set through the environment variable ``MEGENGINE_EXECUTION_STRATEGY``.
    """
    _benchmark_kernel = False
    _deterministic_kernel = False
    if isinstance(option, Strategy):
        _benchmark_kernel = (
            True if option & _valid_string_option["PROFILE"] != Strategy(0) else False
        )
        _deterministic_kernel = (
            True
            if option & _valid_string_option["REPRODUCIBLE"] != Strategy(0)
            else False
        )
        if _benchmark_kernel != _config._benchmark_kernel:
            _clear_algorithm_cache()
        _config._benchmark_kernel = _benchmark_kernel
        _config._deterministic_kernel = _deterministic_kernel
        return

    assert isinstance(option, str)

    for opt in option.split("_"):
        if not opt in _valid_string_option:
            raise ValueError(
                "Valid option can only be one of {}, or combine them with '_'.".format(
                    _valid_string_option.keys()
                )
            )
        _benchmark_kernel |= _valid_string_option[opt] == Strategy.PROFILE
        _deterministic_kernel |= _valid_string_option[opt] == Strategy.REPRODUCIBLE
    if _benchmark_kernel != _config._benchmark_kernel:
        _clear_algorithm_cache()
    _config._benchmark_kernel = _benchmark_kernel
    _config._deterministic_kernel = _deterministic_kernel


@deprecated(version="1.3", reason="use get_execution_strategy() instead")
def get_conv_execution_strategy() -> str:
    return get_execution_strategy()


@deprecated(version="1.3", reason="use set_execution_strategy() instead")
def set_conv_execution_strategy(option: str):
    return set_execution_strategy(option)
