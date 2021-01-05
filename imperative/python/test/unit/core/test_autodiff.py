# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import gc
import platform
import weakref

import numpy as np
import pytest

import megengine as mge
import megengine.distributed as dist
import megengine.functional as F
from megengine.core._imperative_rt import CompNode, TensorAttr, imperative
from megengine.core._imperative_rt.core2 import TensorWeakRef, apply, sync
from megengine.core.autodiff.grad import Grad
from megengine.core.ops.builtin import Elemwise
from megengine.distributed.helper import get_device_count_by_fork
from megengine.functional.distributed import remote_recv, remote_send


def _elwise(mode):
    op = Elemwise(mode)

    def f(*args):
        (result,) = apply(op, *args)
        return result

    return f


add = _elwise(Elemwise.Mode.ADD)
mul = _elwise(Elemwise.Mode.MUL)
cos = _elwise(Elemwise.Mode.COS)
relu = _elwise(Elemwise.Mode.RELU)


def as_tensor(x):
    return mge.Tensor(x)


def save_to(self, name="grad"):
    def callback(grad):
        setattr(self, name, grad)

    return callback


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="windows disable MGB_ENABLE_OPR_MM"
)
@pytest.mark.skipif(get_device_count_by_fork("gpu") < 2, reason="need more gpu device")
@pytest.mark.isolated_distributed
def test_dist_grad():
    world_size = 2
    x_np = np.random.rand(10).astype("float32")

    @dist.launcher
    def worker():
        rank = dist.get_rank()
        if rank == 0:
            grad = Grad()

            x = as_tensor(x_np)
            grad.wrt(x, callback=save_to(x))
            # need a placeholder to trace operator
            remote_send(x, 1)
            recv_x = remote_recv(1, x_np.shape, x_np.dtype)
            y = recv_x * recv_x

            grad([y], [as_tensor(np.ones_like(x_np))])
            np.testing.assert_almost_equal(x.grad.numpy(), x.numpy() * 2)
        elif rank == 1:
            grad = Grad()

            recv_x = remote_recv(0, x_np.shape, x_np.dtype)
            remote_send(recv_x, 0)

            grad([], [])

    worker()


def test_grad():
    x_np = np.random.rand(10).astype("float32")
    x = as_tensor(x_np)

    grad = Grad().wrt(x, callback=save_to(x))

    y = cos(x)

    grad(y, as_tensor(np.ones_like(x_np)))
    np.testing.assert_almost_equal(x.grad.numpy(), -np.sin(x_np))


def test_grad_2():
    x_np = np.random.rand(10).astype("float32")
    x = as_tensor(x_np)

    grad = Grad().wrt(x, callback=save_to(x))

    y = mul(x, x)
    y = mul(y, y)

    grad(y, as_tensor(np.ones_like(x_np)))
    np.testing.assert_almost_equal(x.grad.numpy(), 4 * x_np ** 3, decimal=6)


@pytest.mark.skip(reason="high order gradient was not implemented yet")
def test_2nd_grad():
    x_np = np.random.rand(10).astype("float32")
    x = as_tensor(x_np)
    ones = as_tensor(np.ones_like(x_np))

    grad = Grad().wrt(x, callback=save_to(x))
    grad2 = Grad().wrt(x, callback=save_to(x))

    y = cos(x)

    grad(y, ones)
    np.testing.assert_almost_equal(x.grad.numpy(), -np.sin(x_np), decimal=5)

    grad2(x.grad, ones)
    np.testing.assert_almost_equal(x.grad.numpy(), -np.cos(x_np))


def test_grad_with_tensor_wrapper():
    x_np = np.random.rand(10).astype("float32")
    x = mge.Tensor(x_np)

    grad = Grad().wrt(x, callback=save_to(x))

    y = mul(x, x)
    y = mul(y, y)

    grad(y, mge.Tensor(np.ones_like(x_np)))
    np.testing.assert_almost_equal(x.grad.numpy(), 4 * x_np ** 3, decimal=6)


def test_release():
    def check(f):
        n = 0
        d = None
        gc.disable()
        try:
            for i in range(3):
                f()
                m = len(gc.get_objects())
                d = m - n
                n = m
            assert d == 0
        finally:
            gc.enable()

    x = mge.Tensor([0.0])
    dy = mge.Tensor(np.ones_like(x.numpy()))

    @check
    def _():
        g = Grad().wrt(x)
        y = x * x
        g(y, dy)

    @check
    def _():
        with Grad().wrt(x):
            pass

    @check
    def _():
        with Grad().wrt(x):
            y = x * x


def test_grad_inplace():
    x_np = np.random.rand(10).astype("float32")
    x = mge.Tensor(x_np)

    grad = Grad().wrt(x, callback=save_to(x))

    y = mul(x, x)
    y *= y

    grad(y, mge.Tensor(np.ones_like(x_np)))
    np.testing.assert_almost_equal(x.grad.numpy(), 4 * x_np ** 3, decimal=6)


def test_elemwise_add():
    x_np = np.random.rand(10).astype("float32")
    y_np = np.random.rand(10, 10).astype("float32")
    dz_np = np.random.rand(10, 10).astype("float32")
    x = mge.Tensor(x_np)
    y = mge.Tensor(y_np)
    dz = mge.Tensor(dz_np)

    refs = {}

    def f(x, y):
        x = x * 2
        refs["x"] = TensorWeakRef(x)
        refs["y"] = TensorWeakRef(y)
        return x + y

    grad = Grad().wrt(x, callback=save_to(x))

    z = f(x, y)
    del y

    for k, r in refs.items():
        assert r() is None

    grad(z, dz)
    np.testing.assert_almost_equal(x.grad.numpy(), dz_np.sum(0) * 2, decimal=5)


def test_elemwise_relu():
    x_np = [1.0, -1.0]
    dz_np = [1.0]
    x = mge.Tensor(x_np)
    dz = mge.Tensor(dz_np)

    refs = {}

    def f(x):
        x = x * 2
        refs["x"] = TensorWeakRef(x)
        return relu(x)

    grad = Grad().wrt(x, callback=save_to(x))

    z = f(x)

    assert refs["x"]() is None

    grad(z, dz)
    np.testing.assert_almost_equal(x.grad.numpy(), [2.0, 0])


def test_elemwise_relu_backward_fn():
    op = Elemwise(Elemwise.Mode.RELU)
    attr = TensorAttr()
    attr.dtype = "float32"
    attr.comp_node = "xpux"
    result = imperative.make_backward_graph(op, [attr], [True], [True])
    backward_graph, save_for_backward_mask, input_has_grad = result
    assert save_for_backward_mask == [False, True, True], save_for_backward_mask


def test_reshape():
    x_np = np.random.rand(2, 5).astype("float32")
    x = mge.Tensor(x_np)

    grad = Grad().wrt(x, callback=save_to(x))

    refs = {}

    def f(x):
        x = x * 1
        y = x.reshape(5, 2)
        refs["x"] = TensorWeakRef(x)
        return y

    y = f(x)
    for _, r in refs.items():
        assert r() is None

    grad(y, F.ones_like(y))
    np.testing.assert_equal(np.ones((2, 5), dtype=np.float32), x.grad.numpy())


def test_subtensor():
    x_np = np.random.rand(3, 3).astype("float32")
    x = mge.Tensor(x_np)

    grad = Grad().wrt(x, callback=save_to(x))

    refs = {}

    def f(x):
        x = x * 1
        y = x[1:-1, :2]
        refs["x"] = TensorWeakRef(x)
        return y

    y = f(x)
    for _, r in refs.items():
        assert r() is None

    grad(y, F.ones_like(y))
    np.testing.assert_equal(
        np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32), x.grad.numpy()
    )


def test_IndexingMultiAxisVec():
    x_np = np.random.rand(3, 3).astype("float32")
    x = mge.Tensor(x_np)

    grad = Grad().wrt(x, callback=save_to(x))

    refs = {}

    def f(x):
        x = x * 1
        y = x[[0, 2], [0, 2]]
        refs["x"] = TensorWeakRef(x)
        return y

    y = f(x)
    for _, r in refs.items():
        assert r() is None

    grad(y, F.ones_like(y))
    np.testing.assert_equal(
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.float32), x.grad.numpy()
    )


def test_AxisAddRemove():
    x_np = np.random.rand(1, 5).astype("float32")
    x = mge.Tensor(x_np)

    grad = Grad().wrt(x, callback=save_to(x))

    refs = {}

    def f(x):
        x = x * 1
        y = F.squeeze(F.expand_dims(x, 2), 0)
        refs["x"] = TensorWeakRef(x)
        return y

    y = f(x)
    for _, r in refs.items():
        assert r() is None

    grad(y, F.ones_like(y))
    np.testing.assert_equal(
        np.array([[1, 1, 1, 1, 1]], dtype=np.float32), x.grad.numpy()
    )


def test_Broadcast():
    x_np = np.random.rand(3, 3, 1).astype("float32")
    x = mge.Tensor(x_np)

    grad = Grad().wrt(x, callback=save_to(x))
    y = F.broadcast_to(x, (3, 3, 10))

    grad(y, F.ones_like(y))
    np.testing.assert_equal(np.ones((3, 3, 1), dtype=np.float32) * 10, x.grad.numpy())


def test_Reduce_sum():
    x_np = np.random.rand(3, 3).astype("float32")
    x = mge.Tensor(x_np)

    grad = Grad().wrt(x, callback=save_to(x))
    y = x.sum(axis=0)

    grad(y, F.ones_like(y))
    np.testing.assert_equal(np.ones((3, 3), dtype=np.float32), x.grad.numpy())


def test_Reduce_mean():
    x_np = np.random.rand(3, 3).astype("float32")
    x = mge.Tensor(x_np)

    grad = Grad().wrt(x, callback=save_to(x))
    y = x.mean(axis=0)

    grad(y, F.ones_like(y))
    np.testing.assert_equal(np.ones((3, 3), dtype=np.float32) / 3, x.grad.numpy())


def test_addAxis():
    x_np = np.random.rand(3, 3).astype("float32")
    x = mge.Tensor(x_np)

    grad = Grad().wrt(x, callback=save_to(x))

    refs = {}

    def f(x):
        x = x * 1
        y = F.expand_dims(x, [2, 3])
        refs["x"] = TensorWeakRef(x)
        return y

    y = f(x)
    for _, r in refs.items():
        assert r() is None

    grad(y, F.ones_like(y))
    np.testing.assert_equal(np.ones((3, 3), dtype=np.float32), x.grad.numpy())


def test_removeAxis():
    x_np = np.random.rand(3, 3, 1, 1).astype("float32")
    x = mge.Tensor(x_np)

    grad = Grad().wrt(x, callback=save_to(x))

    refs = {}

    def f(x):
        x = x * 1
        y = F.squeeze(x, [2, 3])
        refs["x"] = TensorWeakRef(x)
        return y

    y = f(x)
    for _, r in refs.items():
        assert r() is None

    grad(y, F.ones_like(y))
    np.testing.assert_equal(np.ones((3, 3, 1, 1), dtype=np.float32), x.grad.numpy())
