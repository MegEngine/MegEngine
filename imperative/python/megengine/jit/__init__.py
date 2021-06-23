# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ..core._imperative_rt.core2 import (
    set_cpp_apply_const_with_tracing,
    set_cpp_apply_with_tracing,
)
from .dtr_config import DTRConfig
from .graph_opt_config import GraphOptimizationConfig
from .sublinear_memory_config import SublinearMemoryConfig
from .tracing import (
    apply_const_with_tracing,
    apply_with_tracing,
    exclude_from_trace,
    trace,
)

set_cpp_apply_with_tracing(apply_with_tracing)
set_cpp_apply_const_with_tracing(apply_const_with_tracing)
