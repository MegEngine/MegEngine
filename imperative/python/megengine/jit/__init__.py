# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ..core._imperative_rt.core2 import (
    set_cpp_apply_compiled_mode,
    set_cpp_apply_const_compiled_mode,
    set_cpp_apply_const_with_tracing,
    set_cpp_apply_with_tracing,
)
from .sublinear_memory_config import SublinearMemoryConfig
from .tracing import (
    apply_compiled_mode,
    apply_const_compiled_mode,
    apply_const_with_tracing,
    apply_with_tracing,
    exclude_from_trace,
    trace,
)

set_cpp_apply_with_tracing(apply_with_tracing)
set_cpp_apply_const_with_tracing(apply_const_with_tracing)
set_cpp_apply_compiled_mode(apply_compiled_mode)
set_cpp_apply_const_compiled_mode(apply_const_compiled_mode)
