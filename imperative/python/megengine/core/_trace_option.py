# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import os

_use_symbolic_shape = False
if os.environ.get("MEGENGINE_USE_SYMBOLIC_SHAPE"):
    _use_symbolic_shape = True


def use_symbolic_shape() -> bool:
    """
    Returns whether tensor.shape returns a tensor instead of a tuple

    """
    return _use_symbolic_shape


def set_symbolic_shape(option: bool):
    """ Sets whether tensor.shape returns a tensor instead of a tuple
    """
    global _use_symbolic_shape
    _org = _use_symbolic_shape
    _use_symbolic_shape = option
    return _org
