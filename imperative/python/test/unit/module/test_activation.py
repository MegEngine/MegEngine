# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

import megengine as mge
from megengine.module import LeakyReLU, ELU


def test_leaky_relu():
    data = np.array([-8, -12, 6, 10]).astype(np.float32)
    negative_slope = 0.1

    leaky_relu = LeakyReLU(negative_slope)
    output = leaky_relu(mge.tensor(data))

    np_output = np.maximum(0, data) + negative_slope * np.minimum(0, data)
    np.testing.assert_equal(output.numpy(), np_output)

def test_elu():
    data = np.array([-2,-1,0,1,2]).astype(np.float32)
    alpha = 1.0

    elu = ELU(alpha)
    output = elu(mge.tensor(data))

    np_output = np.maximum(0, data) + np.minimum(0, alpha * (np.exp(data) - 1))
    np.testing.assert_equal(output.numpy(), np_output)
