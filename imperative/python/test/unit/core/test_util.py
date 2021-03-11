# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import logging

from megengine.core._imperative_rt import Logger
from megengine.logger import _imperative_rt_logger, set_mgb_log_level


def test_logger():
    orig_level = Logger().set_log_level(Logger.LogLevel.Debug)
    assert Logger().set_log_level(Logger.LogLevel.Debug) == Logger.LogLevel.Debug
    Logger().set_log_level(orig_level)
    orig_level = set_mgb_log_level(logging.DEBUG)
    assert (
        _imperative_rt_logger.set_log_level(Logger.LogLevel.Debug)
        == Logger.LogLevel.Debug
    )
    _imperative_rt_logger.set_log_level(orig_level)
