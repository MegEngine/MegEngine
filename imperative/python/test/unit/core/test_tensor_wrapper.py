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
    def test_x(x_np):
        for m in ["sum", "prod", "min", "max", "mean"]:
            x = TensorWrapper(x_np)
            y = getattr(x, m)(axis=-1, keepdims=True)
            np.testing.assert_almost_equal(y.numpy(), getattr(x_np, m)(-1), decimal=6)

    test_x((10 * np.random.rand(10) + 1).astype("int32"))
    test_x(np.random.rand(10).astype("float32"))
    test_x(np.array([True, True, True]))
    test_x(np.array([True, False, True]))


def test_set_subtensor():
    x = TensorWrapper([1, 2, 3])
    x[:] = [1, 1, 1]
    np.testing.assert_almost_equal(x.numpy(), [1, 1, 1], decimal=6)
    x[[0, 2]] = [3, 2]
    np.testing.assert_almost_equal(x.numpy(), [3, 1, 2], decimal=6)
    x[1:3] = [4, 5]
    np.testing.assert_almost_equal(x.numpy(), [3, 4, 5], decimal=6)


def test_computing_with_numpy_array():
    x = np.array([1, 2, 3], dtype=np.int32)
    xx = TensorWrapper(x, device="cpu0")
    y = np.array([1, 0, 3], dtype=np.int32)
    assert np.add(xx, y).device == xx.device
    np.testing.assert_equal(np.add(xx, y).numpy(), np.add(x, y))
    np.testing.assert_equal(np.equal(xx, y).numpy(), np.equal(x, y))
    np.testing.assert_equal(np.equal(xx, xx).numpy(), np.equal(x, x))


def test_transpose():
    x = np.random.rand(2, 5).astype("float32")
    xx = TensorWrapper(x)
    np.testing.assert_almost_equal(xx.T.numpy(), x.T)
