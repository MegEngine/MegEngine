# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import copy

import numpy as np
import pytest
from utils import make_tensor

from megengine.core.tensor.dtype import get_scale, get_zero_point, qint8, quint8
from megengine.tensor import Parameter, Tensor
from megengine.utils.network import Network


@pytest.mark.parametrize("is_varnode", [True, False])
def test_basic(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    x_np = np.random.rand(10).astype("float32")
    x = make_tensor(x_np, network)
    y = x * x
    y_np = y.numpy()
    np.testing.assert_almost_equal(y_np, x_np * x_np)


@pytest.mark.parametrize("is_varnode", [True, False])
def test_literal_arith(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    x_np = np.random.rand(10).astype("float32")
    x = make_tensor(x_np, network)
    y = x * 2
    y_np = y.numpy()
    np.testing.assert_almost_equal(y_np, x_np * 2)


@pytest.mark.parametrize("is_varnode", [True, False])
def test_matmul(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    A = make_tensor(np.random.rand(5, 7).astype("float32"), network)
    B = make_tensor(np.random.rand(7, 10).astype("float32"), network)
    C = A @ B
    np.testing.assert_almost_equal(C.numpy(), A.numpy() @ B.numpy(), decimal=6)


@pytest.mark.parametrize("is_varnode", [True, False])
def test_inplace_add(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    x_np = np.random.rand(10).astype("float32")
    y_np = np.random.rand(10).astype("float32")
    x = make_tensor(x_np, network)
    y = make_tensor(y_np, network)
    y += x
    out_np = y.numpy()
    np.testing.assert_almost_equal(out_np, x_np + y_np)


@pytest.mark.parametrize("is_varnode", [True, False])
def test_reduce(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    def test_x(x_np):
        for m in ["sum", "prod", "min", "max", "mean"]:
            x = make_tensor(x_np, network)
            y = getattr(x, m)(axis=-1, keepdims=True)
            np.testing.assert_almost_equal(y.numpy(), getattr(x_np, m)(-1), decimal=6)

    test_x((10 * np.random.rand(10) + 1).astype("int32"))
    test_x(np.random.rand(10).astype("float32"))
    test_x(np.array([True, True, True]))
    test_x(np.array([True, False, True]))


@pytest.mark.parametrize("is_varnode", [True, False])
def test_set_value(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    v0 = np.random.random((2, 3)).astype(np.float32)
    param = make_tensor(v0, network)
    v1 = np.random.random((2, 3)).astype(np.float32)
    param[...] = v1
    np.testing.assert_allclose(param.numpy(), v1, atol=5e-6)


@pytest.mark.parametrize("is_varnode", [True, False])
def test_set_subtensor(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    x = make_tensor([1, 2, 3], network)
    x[:] = [1, 1, 1]
    np.testing.assert_almost_equal(x.numpy(), [1, 1, 1], decimal=6)
    x[[0, 2]] = [3, 2]
    np.testing.assert_almost_equal(x.numpy(), [3, 1, 2], decimal=6)
    x[1:3] = [4, 5]
    np.testing.assert_almost_equal(x.numpy(), [3, 4, 5], decimal=6)


def test_computing_with_numpy_array():
    x = np.array([1, 2, 3], dtype=np.int32)
    xx = Tensor(x, device="cpu0")
    y = np.array([1, 0, 3], dtype=np.int32)
    assert np.add(xx, y).device == xx.device
    np.testing.assert_equal(np.add(xx, y).numpy(), np.add(x, y))
    np.testing.assert_equal(np.equal(xx, y).numpy(), np.equal(x, y))
    np.testing.assert_equal(np.equal(xx, xx).numpy(), np.equal(x, x))


@pytest.mark.parametrize("is_varnode", [True, False])
def test_transpose(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    x = np.random.rand(2, 5).astype("float32")
    xx = make_tensor(x, network)
    np.testing.assert_almost_equal(xx.T.numpy(), x.T)


@pytest.mark.parametrize("is_varnode", [True, False])
def test_as_type(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    x_np = np.array([1, 2, 3], dtype=np.float32)
    x = make_tensor(x_np, network)
    y = x.astype(qint8(0.1))
    np.testing.assert_almost_equal(get_scale(y.dtype), 0.1)
    z = y.astype(qint8(0.2))
    np.testing.assert_almost_equal(get_scale(z.dtype), 0.2)
    a = z.astype(quint8(0.3, 127))
    np.testing.assert_almost_equal(get_scale(a.dtype), 0.3)
    np.testing.assert_equal(get_zero_point(a.dtype), 127)
    b = a.astype(quint8(0.3, 128))
    np.testing.assert_almost_equal(get_scale(b.dtype), 0.3)
    np.testing.assert_equal(get_zero_point(b.dtype), 128)


def test_serialization():
    x = Tensor([1, 2, 3], dtype=np.float32)
    newargs = x.__getnewargs__()
    states = x.__getstate__()
    assert np.all(newargs[0] == x.numpy())
    assert newargs[1] == x.dtype
    assert newargs[2] == x.device.logical_name
    assert not states
    x.qparams
    states = x.__getstate__()
    assert len(states.keys()) == 1
    assert states["qparams"] == x.qparams


def test_qparams():
    x = Tensor(1)
    assert x.qparams.scale is None
    x.qparams.scale = Tensor(1.0)
    assert x.qparams.scale.numpy() == 1.0
    x2 = copy.copy(x)
    assert x.qparams is x2.qparams and x2.qparams.scale.numpy() == 1.0
    x3 = copy.deepcopy(x)
    assert x.qparams is not x3.qparams and x3.qparams.scale.numpy() == 1.0


def test_name():
    x = Tensor(0)
    assert x.name == ""
    x.name = "x"
    assert x.name == "x"
    x = Tensor(0, name="x")
    assert x.name == "x"


def test_tensor_type():
    x1 = Parameter(1)
    x2 = Tensor(2)
    y1 = x1 + x2
    y2 = x2 + x1
    assert type(y1) == type(y2)
