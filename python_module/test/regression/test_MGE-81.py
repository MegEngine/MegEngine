# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

import megengine as mge
import megengine.functional as F
import megengine.module as M
from megengine.core import tensor


def test_mge_81():
    np.random.seed(0)
    N, D = 3, 4
    x = mge.Parameter(value=np.random.normal(size=(N, D)).astype(np.float32))
    y = mge.Parameter(value=np.random.normal(size=(N, D)).astype(np.float32))
    z = mge.Parameter(value=np.random.normal(size=(N, D)).astype(np.float32))
    a = x * y
    b = a + z
    c = F.sum(b)
    grad_x = F.grad(c, x, use_virtual_grad=False)
    grad_y = F.grad(c, y, use_virtual_grad=False)
    grad_z = F.grad(c, z, use_virtual_grad=False)
    print(grad_x.numpy())
    print(grad_y.numpy())
    print(grad_z.numpy())
    m = M.BatchNorm2d(4)
    input = tensor(np.zeros((64, 4, 32, 32), dtype=np.float32))
    _ = m(input)
    m = M.BatchNorm2d(4, affine=False)
    _ = m(input)
