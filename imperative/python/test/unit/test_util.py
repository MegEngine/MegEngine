# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from megengine.core._imperative_rt import Logger


def test_logger():
    orig_level = Logger().set_log_level(Logger.LogLevel.Info)
    assert Logger().set_log_level(Logger.LogLevel.Info) == Logger.LogLevel.Info
    Logger().set_log_level(orig_level)
