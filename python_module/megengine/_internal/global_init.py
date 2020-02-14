# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""global initialization work; classes/functions defined in this module should
not be used by user code"""

import atexit
import os
import sys
import traceback

from . import mgb
from .logconf import get_logger
from .persistent_cache import PersistentCacheOnServer


class PyStackExtracterImpl(mgb._PyStackExtracter):
    def extract(self):
        return "".join(traceback.format_stack()[:-1])


mgb._register_logger(get_logger())
assert sys.executable
mgb._timed_func_set_fork_exec_path(
    sys.executable,
    os.path.join(os.path.dirname(__file__), "_timed_func_fork_exec_entry.py"),
)

persistent_cache_impl_ins = PersistentCacheOnServer()
mgb._PersistentCache.reg(persistent_cache_impl_ins)

PyStackExtracterImplIns = PyStackExtracterImpl()
PyStackExtracterImpl.reg(PyStackExtracterImplIns)

atexit.register(mgb._mgb_global_finalize)
