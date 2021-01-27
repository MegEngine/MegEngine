# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np
import pytest

import megengine.functional as F
from megengine import tensor


def test_cross_entropy_with_logits():
    data = tensor([[0, 50], [0, -150]]).astype(np.float32)
    label = tensor([1, 0]).astype(np.int32)
    loss = F.nn.cross_entropy(data, label)
    np.testing.assert_allclose(loss.numpy(), 0.0)
    label = tensor([0, 1]).astype(np.int32)
    loss = F.nn.cross_entropy(data, label)
    np.testing.assert_allclose(loss.numpy(), 100)

    label = np.array([1, 0])
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


def test_cross_entropy_reduction():
    logits = np.random.randn(16, 10)
    label = np.random.randint(10, size=[16])
    logits = tensor(logits, dtype="float32")
    label = tensor(label, dtype="int32")

    perm = np.random.permutation(16)
    logits_perm = tensor(logits[perm], dtype="float32")
    label_perm = tensor(label[perm], dtype="int32")

    loss = F.nn.cross_entropy(logits, label, reduction="none")
    loss_perm = F.nn.cross_entropy(logits_perm, label_perm, reduction="none")
    np.testing.assert_allclose(loss.numpy()[perm], loss_perm.numpy())

    loss_sum = F.nn.cross_entropy(logits, label, reduction="sum")
    np.testing.assert_allclose(loss.numpy().sum(), loss_sum.numpy(), rtol=2e-7)

    loss_mean = F.nn.cross_entropy(logits, label, reduction="mean")
    np.testing.assert_allclose(loss_mean.numpy(), loss_sum.numpy() / 16)

    loss_ls = F.nn.cross_entropy(logits, label, reduction="mean", label_smooth=0.1)
    loss_ls_none_reduce = F.nn.cross_entropy(
        logits, label, reduction="none", label_smooth=0.1
    )
    np.testing.assert_allclose(
        loss_ls.numpy(), loss_ls_none_reduce.numpy().mean(), rtol=2e-7
    )

    with pytest.raises(ValueError):
        F.nn.cross_entropy(logits, label, reduction="MEAN")

    with pytest.raises(ValueError):
        F.nn.cross_entropy(logits, label, reduction="max")
