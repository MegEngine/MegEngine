# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import os

_use_tensor_shape = False
if os.environ.get("MEGENGINE_USE_TENSOR_SHAPE"):
    _use_tensor_shape = True


def use_tensor_shape() -> bool:
    """Returns whether tensor.shape returns a tensor instead of a tuple

    """
    return _use_tensor_shape


def set_tensor_shape(option: bool):
    """ Sets whether tensor.shape returns a tensor instead of a tuple
    """
    global _use_tensor_shape
    _use_tensor_shape = option
