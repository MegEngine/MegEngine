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


def test_mge_323():
    # Regression: set_value does not update eager_val
    x = mge.tensor([0])
    _ = x * 2
    x.set_value([1, 1])
    np.testing.assert_array_equal(x.numpy(), [1, 1])
    assert x.shape == (2,)
    np.testing.assert_array_equal(x * 2, [2, 2])
