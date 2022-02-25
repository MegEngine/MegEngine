# -*- coding: utf-8 -*-
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
