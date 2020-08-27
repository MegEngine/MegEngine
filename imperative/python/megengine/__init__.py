# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import os
import sys

from .core._imperative_rt.utils import _set_fork_exec_path_for_timed_func
from .device import *
from .logger import enable_debug_log, get_logger, set_log_file, set_log_level
from .serialization import load, save
from .tensor import Tensor, tensor
from .tensor_nn import Buffer, Parameter
from .version import __version__

_set_fork_exec_path_for_timed_func(
    sys.executable,
    os.path.join(os.path.dirname(__file__), "utils", "_timed_func_fork_exec_entry.py"),
)

del _set_fork_exec_path_for_timed_func
