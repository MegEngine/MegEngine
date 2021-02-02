# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ..core._imperative_rt.core2 import apply
from ..core.ops import builtin
from ..core.ops.builtin import InplaceAdd


def _inplace_add_(dest, delta, alpha, beta):
    isscalar = dest.isscalar()
    dest._reset(apply(InplaceAdd(), dest, delta, alpha, beta)[0])
    if isscalar:
        dest.setscalar()
    return dest
