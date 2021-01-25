# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import os
import sys
from contextlib import contextmanager

from ._imperative_rt.core2 import get_option, set_option
from .tensor.megbrain_graph import Graph


@contextmanager
def option(key, value):
    value = int(value)
    old = get_option(key)
    set_option(key, value)
    yield
    assert get_option(key) == value
    set_option(key, old)
