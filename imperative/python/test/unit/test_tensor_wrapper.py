# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

from megengine.core.tensor.tensor_wrapper import TensorWrapper


def test_basic():
    x_np = np.random.rand(10).astype("float32")
    x = TensorWrapper(x_np)
    y = x * x
    y_np = y.numpy()
    np.testing.assert_almost_equal(y_np, x_np * x_np)


def test_literal_arith():
    x_np = np.random.rand(10).astype("float32")
    x = TensorWrapper(x_np)
    y = x * 2
    y_np = y.numpy()
    np.testing.assert_almost_equal(y_np, x_np * 2)


def test_matmul():
    A = TensorWrapper(np.random.rand(5, 7).astype("float32"))
    B = TensorWrapper(np.random.rand(7, 10).astype("float32"))
    C = A @ B
    np.testing.assert_almost_equal(C.numpy(), A.numpy() @ B.numpy(), decimal=6)


def test_reduce():
    for m in ["sum", "prod", "min", "max", "mean"]:
        x_np = np.random.rand(10).astype("float32")
        x = TensorWrapper(x_np)
        y = getattr(x, m)(-1)
        np.testing.assert_almost_equal(y.numpy(), getattr(x_np, m)(-1), decimal=6)


def test_set_subtensor():
    x = TensorWrapper([1, 2, 3])
    x[:] = [1, 1, 1]
    np.testing.assert_almost_equal(x.numpy(), [1, 1, 1], decimal=6)
    x[[0, 2]] = [3, 2]
    np.testing.assert_almost_equal(x.numpy(), [3, 1, 2], decimal=6)
    x[1:3] = [4, 5]
    np.testing.assert_almost_equal(x.numpy(), [3, 4, 5], decimal=6)
