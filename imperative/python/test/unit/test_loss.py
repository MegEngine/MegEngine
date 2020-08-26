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


# XXX need to test label_smooth
def test_cross_entropy_with_softmax():
    data = tensor([1, 100]).astype(np.float32).reshape((1, 2))
    label = tensor([1]).astype(np.int32)
    loss = F.cross_entropy_with_softmax(data, label)
    np.testing.assert_allclose(loss.numpy(), 0.0)
    label = tensor([0]).astype(np.int32)
    loss = F.cross_entropy_with_softmax(data, label)
    np.testing.assert_allclose(loss.numpy(), 100 - 1)

    label = np.array([1])
    loss = F.cross_entropy_with_softmax(data, label)
    np.testing.assert_allclose(loss.numpy(), 0.0)
