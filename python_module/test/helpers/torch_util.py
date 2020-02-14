# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np
import torch

from megengine.core import tensor
from megengine.utils import prod


def _uniform(shape):
    return np.random.random(shape).astype(np.float32)


def init_with_same_value(mge_param, torch_param, initializer=_uniform):
    mge_shape = mge_param.shape
    torch_shape = torch_param.shape
    assert prod(mge_shape) == prod(torch_shape)
    weight = initializer(mge_shape)
    mge_param.set_value(weight)
    torch_param.data = torch.Tensor(weight.reshape(torch_shape))


def gen_same_input(shape, initializer=_uniform):
    data = initializer(shape)
    mge_input = tensor(data)
    torch_input = torch.Tensor(data)
    return mge_input, torch_input
