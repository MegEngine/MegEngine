# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

import megengine.functional as F
from megengine import tensor
from megengine.module import Elemwise


def test_module_elemwise():
    def test_func(method, *inps):
        elemwise = Elemwise(method)
        outputs = elemwise(*inps)
        return outputs.numpy()

    x = np.random.rand(100).astype("float32")
    y = np.random.rand(100).astype("float32")
    x, y = tensor(x), tensor(y)
    np.testing.assert_almost_equal(
        test_func("H_SWISH", x), F.hswish(x).numpy(), decimal=6
    )
    np.testing.assert_almost_equal(
        test_func("ADD", x, y), F.add(x, y).numpy(), decimal=6
    )
