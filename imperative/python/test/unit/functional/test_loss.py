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


def test_cross_entropy_with_logits():
    data = tensor([1, 100]).astype(np.float32).reshape((1, 2))
    label = tensor([1]).astype(np.int32)
    loss = F.nn.cross_entropy(data, label)
    np.testing.assert_allclose(loss.numpy(), 0.0)
    label = tensor([0]).astype(np.int32)
    loss = F.nn.cross_entropy(data, label)
    np.testing.assert_allclose(loss.numpy(), 100 - 1)

    label = np.array([1])
    loss = F.nn.cross_entropy(data, label)
    np.testing.assert_allclose(loss.numpy(), 0.0)


def test_cross_entropy():
    def softmax(x):
        x = np.exp(x)
        x /= x.sum(1, keepdims=True)
        return x

    def ref(x, y):
        return np.mean([-np.log(x[i, y[i]]) for i in range(len(y))])

    x = (np.random.rand(5, 10) - 0.5) * 4
    y = np.random.randint(10, size=(5,))
    for i in range(len(x)):
        x[i, y[i]] += np.random.rand() * 2
    x = softmax(x)
    l_ref = ref(x, y)
    l = F.nn.cross_entropy(tensor(x, "float32"), tensor(y, "int32"), with_logits=False)
    np.testing.assert_allclose(l.numpy(), l_ref)
