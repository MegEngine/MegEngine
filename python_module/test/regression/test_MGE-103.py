# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np
import pytest

import megengine as mge
from megengine.core import tensor
from megengine.jit import trace
from megengine.module import BatchNorm2d
from megengine.test import assertTensorClose


@pytest.mark.regression
def test_batchnorm_change_batchsize():
    data_shape = (2, 3, 8, 8)
    real_shape = (4, 3, 8, 8)
    data = np.random.random(data_shape).astype(np.float32)
    d = np.random.random(real_shape).astype(np.float32)

    bn = BatchNorm2d(3)
    f = trace(bn)
    f(data)

    y1 = f(d)

    y0 = bn(tensor(d))

    assertTensorClose(y0.numpy(), y1.numpy())
