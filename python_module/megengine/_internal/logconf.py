# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import logging
import os

_replaced_logger = None


def get_logger():
    global _replaced_logger
    if _replaced_logger is not None:
        return _replaced_logger
    logger = logging.getLogger("megbrain")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(MgbLogFormatter(datefmt="%d %H:%M:%S"))
    handler.setLevel(0)
    del logger.handlers[:]
    logger.addHandler(handler)
    _replaced_logger = logger
    return logger


class MgbLogFormatter(logging.Formatter):
    def format(self, record):
        date = "\x1b[32m[%(asctime)s %(lineno)d@%(filename)s:%(name)s]\x1b[0m"
        msg = "%(message)s"
        if record.levelno == logging.DEBUG:
            fmt = "{} \x1b[32mDBG\x1b[0m {}".format(date, msg)
        elif record.levelno == logging.WARNING:
            fmt = "{} \x1b[1;31mWRN\x1b[0m {}".format(date, msg)
        elif record.levelno == logging.ERROR:
            fmt = "{} \x1b[1;4;31mERR\x1b[0m {}".format(date, msg)
        else:
            fmt = date + " " + msg
        self._style._fmt = fmt
        return super().format(record)


def set_logger(logger):
    """replace the logger"""
    global _replaced_logger
    _replaced_logger = logger
    from .mgb import _register_logger

    _register_logger(logger)
