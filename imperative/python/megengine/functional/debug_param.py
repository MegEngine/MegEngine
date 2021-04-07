# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import os

from ..core.ops import builtin
from ..logger import get_logger
from ..utils.deprecation import deprecated

Strategy = builtin.ops.Convolution.Strategy

_execution_strategy = os.getenv("MEGENGINE_EXECUTION_STRATEGY", "HEURISTIC")

if os.getenv("MEGENGINE_CONV_EXECUTION_STRATEGY") != None:
    get_logger().warning(
        "Environment variable `MEGENGINE_CONV_EXECUTION_STRATEGY` is deprecated, please use `MEGENGINE_EXECUTION_STRATEGY`"
    )


def get_execution_strategy() -> Strategy:
    """
    Returns the execution strategy of :class:`~.Conv2d` and :func:'~.matmul'

    See :func:`~.set_execution_strategy` for possible return values
    """
    return _execution_strategy


def set_execution_strategy(option):
    """
    Sets the execution strategy of :class:`~.Conv2d` and :func:'~.matmul'

    :param option: Decides how :class:`~.Conv2d`and :func:'~.matmul' algorithms are chosen.
        Available value Strategy
        * HEURISTIC uses heuristic to choose the fastest algorithm.
        * PROFILE runs possible algorithms on real device to find the best one.
        * REPRODUCIBLE uses the algorithms that is reproducible.
        * OPTIMIZED uses the algorithms that is optimized.

        The default strategy is HEURISTIC, this options can be combined to
        form a combination option, e.g. PROFILE | REPRODUCIBLE
        can combined a option that uses the fastest of profiling result that is also reproducible.

        Available values string:

        * 'HEURISTIC' uses heuristic to choose the fastest algorithm.
        * 'PROFILE' runs possible algorithms on real device to find the best one.
        * 'PROFILE_HEURISTIC' uses profiling result and heuristic to choose the fastest algorithm.
        * 'PROFILE_REPRODUCIBLE' uses the fastest of profiling result that is also reproducible.
        * 'HEURISTIC_REPRODUCIBLE' uses heuristic to choose the fastest algorithm that is also reproducible.

        The default strategy is 'HEURISTIC'.

        It can also be set through the environment variable 'MEGENGINE_EXECUTION_STRATEGY'.
    """
    valid_string_option = {
        "REPRODUCIBLE": Strategy.REPRODUCIBLE,
        "HEURISTIC": Strategy.HEURISTIC,
        "PROFILE": Strategy.PROFILE,
    }

    global _execution_strategy  # pylint: disable=global-statement
    if isinstance(option, Strategy):
        _execution_strategy = option
        return

    assert isinstance(option, str)

    strategy_tmp = Strategy(0)
    for opt in option.split("_"):
        if not opt in valid_string_option:
            raise ValueError(
                "Valid option can only be one of {}, or combine them with '_'.".format(
                    valid_string_option.keys()
                )
            )
        strategy_tmp = strategy_tmp | valid_string_option[opt]
    _execution_strategy = strategy_tmp


@deprecated(version="1.3", reason="use get_execution_strategy() instead")
def get_conv_execution_strategy() -> str:
    return get_execution_strategy()


@deprecated(version="1.3", reason="use set_execution_strategy() instead")
def set_conv_execution_strategy(option: str):
    return set_execution_strategy(option)
