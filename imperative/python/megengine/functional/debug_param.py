# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import os

_conv_execution_strategy = os.getenv("MEGENGINE_CONV_EXECUTION_STRATEGY", "HEURISTIC")


def get_conv_execution_strategy() -> str:
    """
    Returns the execuation strategy of :class:`~.Conv2d`.

    See :func:`~.set_conv_execution_strategy` for possible return values
    """
    return _conv_execution_strategy


def set_conv_execution_strategy(option: str):
    """
    Sets the execuation strategy of :class:`~.Conv2d`.

    :param option: Decides how :class:`~.Conv2d` algorithm is chosen.
        Available values:

        * 'HEURISTIC' uses heuristic to choose the fastest algorithm.
        * 'PROFILE' runs possible algorithms on real device to find the best one.
        * 'PROFILE_HEURISTIC' uses profiling result and heuristic to choose the fastest algorithm.
        * 'PROFILE_REPRODUCIBLE' uses the fastest of profiling result that is also reproducible.
        * 'HEURISTIC_REPRODUCIBLE' uses heuristic to choose the fastest algorithm that is also reproducible.

        The default strategy is 'HEURISTIC'.

        It can also be set through the environment variable 'MEGENGINE_CONV_EXECUTION_STRATEGY'.
    """
    valid_option = (
        "HEURISTIC",
        "PROFILE",
        "PROFILE_HEURISTIC",
        "PROFILE_REPRODUCIBLE",
        "HEURISTIC_REPRODUCIBLE",
    )
    if not option in valid_option:
        raise ValueError("Valid option can only be one of {}".format(valid_option))

    global _conv_execution_strategy  # pylint: disable=global-statement
    _conv_execution_strategy = option
