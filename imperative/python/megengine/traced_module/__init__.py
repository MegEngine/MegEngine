# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from ..core._imperative_rt.core2 import set_cpp_apply_module_trace
from . import compat
from .traced_module import (
    TracedModule,
    _register_all_builtin_module,
    cpp_apply_module_trace,
    register_as_builtin,
    trace_module,
    wrap,
)

_register_all_builtin_module()
set_cpp_apply_module_trace(cpp_apply_module_trace)
