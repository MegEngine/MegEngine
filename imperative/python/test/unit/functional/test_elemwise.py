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


def test_abs():
    np.testing.assert_allclose(
        F.abs(tensor([-3.0, -4.0, -5.0])).numpy(),
        np.abs(np.array([-3.0, -4.0, -5.0], dtype=np.float32)),
    )

    np.testing.assert_allclose(F.abs(-3.0).numpy(), np.abs(np.float32(-3.0)))


def test_multiply():
    np.testing.assert_allclose(
        F.mul(-3.0, -4.0).numpy(), np.multiply(np.float32(-3.0), np.float32(-4.0))
    )

    np.testing.assert_allclose(
        F.mul(tensor([3.0, 4.0]), 4.0).numpy(),
        np.multiply(np.array([3.0, 4.0], dtype=np.float32), 4.0),
    )

    np.testing.assert_allclose(
        F.mul(4.0, tensor([3.0, 4.0])).numpy(),
        np.multiply(4.0, np.array([3.0, 4.0], dtype=np.float32)),
    )

    np.testing.assert_allclose(
        F.mul(tensor([3.0, 4.0]), tensor([3.0, 4.0])).numpy(),
        np.multiply(
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        ),
    )


def test_clamp():
    """Fix an issue when `lower` or `upper` is 0, it will be recognized as `False` and
    `F.clip` will fall into wrong conditions unexpectedly.
    """
    x = np.linspace(-6, 6, dtype="float32")
    np.testing.assert_allclose(
        F.clip(tensor(x) + 3, 0, 6).numpy(), np.clip(x + 3, 0, 6)
    )
    np.testing.assert_allclose(
        F.clip(tensor(x) - 3, -6, 0).numpy(), np.clip(x - 3, -6, 0)
    )


def test_isnan():
    for case in [[1, float("nan"), 0]]:
        np.testing.assert_allclose(F.isnan(tensor(case)).numpy(), np.isnan(case))


def test_isinf():
    for case in [[1, float("inf"), 0]]:
        np.testing.assert_allclose(F.isinf(tensor(case)).numpy(), np.isinf(case))


def test_sign():
    for case in [[1, -1, 0]]:
        x = tensor(case)
        np.testing.assert_allclose(F.sign(x).numpy(), np.sign(case).astype(x.dtype))


def test_cosh():
    np.random.seed(42)
    x = np.random.randn(100).astype("float32")
    y_np = np.cosh(x)
    y_mge = F.cosh(tensor(x)).numpy()
    np.testing.assert_allclose(y_np, y_mge, rtol=1e-5)


def test_sinh():
    np.random.seed(42)
    x = np.random.randn(100).astype("float32")
    y_np = np.sinh(x)
    y_mge = F.sinh(tensor(x)).numpy()
    np.testing.assert_allclose(y_np, y_mge, rtol=1e-5)


def test_asinh():
    np.random.seed(42)
    x = np.random.randn(100).astype("float32")
    y_np = np.arcsinh(x)
    y_mge = F.asinh(tensor(x)).numpy()
    np.testing.assert_almost_equal(y_np, y_mge, decimal=5)


def test_acosh():
    x = np.arange(0, 10000).astype("float32") / 100 + 1
    y_np = np.arccosh(x)
    y_mge = F.acosh(tensor(x)).numpy()
    np.testing.assert_almost_equal(y_np, y_mge, decimal=6)


def test_atanh():
    np.random.seed(42)
    x = np.random.rand(100).astype("float32") * 2 - 1
    y_np = np.arctanh(x)
    y_mge = F.atanh(tensor(x)).numpy()
    np.testing.assert_almost_equal(y_np, y_mge, decimal=5)


def test_hswish():
    np.random.seed(42)
    x = np.random.randn(100).astype("float32")
    y_np = x * np.minimum(np.maximum(x + 3, 0), 6) / 6
    y_mge = F.hswish(tensor(x)).numpy()
    np.testing.assert_almost_equal(y_np, y_mge, decimal=6)


def test_hsigmoid():
    np.random.seed(42)
    x = np.random.randn(100).astype("float32")
    y_np = np.minimum(np.maximum(x + 3, 0), 6) / 6
    y_mge = F.hsigmoid(tensor(x)).numpy()
    np.testing.assert_equal(y_np, y_mge)


def test_logical_oprs():
    x = np.array([[True, False], [False, True]])
    y = np.array([[True, True], [False, False]])
    xx = tensor(x)
    yy = tensor(y)
    np.testing.assert_equal(~x, (F.logical_not(xx)).numpy())
    np.testing.assert_equal(x & y, F.logical_and(xx, yy).numpy())
    np.testing.assert_equal(x | y, F.logical_or(xx, yy).numpy())
    np.testing.assert_equal(x ^ y, F.logical_xor(xx, yy).numpy())
