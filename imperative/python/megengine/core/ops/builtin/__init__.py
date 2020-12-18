# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import warnings
from typing import Union

from ..._imperative_rt import OpDef, ops

__all__ = ["OpDef"]

for k, v in ops.__dict__.items():
    if isinstance(v, type) and issubclass(v, OpDef):
        globals()[k] = v
        __all__.append(k)
