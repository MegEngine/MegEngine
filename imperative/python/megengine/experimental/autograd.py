# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from ..core._imperative_rt.core2 import (
    set_allow_higher_order_directive as _set_allow_higher_order_directive,
)

__all__ = [
    "enable_higher_order_directive",
    "disable_higher_order_directive",
]


def enable_higher_order_directive():
    _set_allow_higher_order_directive(True)


def disable_higher_order_directive():
    _set_allow_higher_order_directive(False)
