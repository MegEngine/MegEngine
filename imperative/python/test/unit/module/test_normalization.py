# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

import megengine.module.normalization as norm
from megengine import tensor


def shape_to_tuple(shape):
    if isinstance(shape, tensor):
        shape = tuple(shape.tolist())
    return shape


def test_group_norm():
    input_shape = (2, 100, 128, 128)
    channels = input_shape[1]
    groups = [2, 5, 10, 50]
    x = tensor(np.random.rand(*input_shape))
    for group in groups:
        gn = norm.GroupNorm(group, channels)
        out = gn(x)
        assert shape_to_tuple(out.shape) == input_shape


def test_layer_norm():
    input_shape = (2, 100, 128, 128)
    channels = input_shape[1]
    x = tensor(np.random.rand(*input_shape))
    ln = norm.LayerNorm(channels)
    out = ln(x)
    assert shape_to_tuple(out.shape) == input_shape


def test_instance_norm():
    input_shape = (2, 100, 128, 128)
    channels = input_shape[1]
    x = tensor(np.random.rand(*input_shape))
    inst_norm = norm.InstanceNorm(channels)
    out = inst_norm(x)
    assert shape_to_tuple(out.shape) == input_shape
