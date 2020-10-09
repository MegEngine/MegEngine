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
from megengine.core._imperative_rt import TensorAttr, imperative
from megengine.core._imperative_rt.imperative import sync
from megengine.core.autodiff.grad import Grad
from megengine.core.ops.builtin import Elemwise
from megengine.core.tensor.raw_tensor import as_raw_tensor
from megengine.core.tensor.tensor import Tensor, apply
from megengine.core.tensor.tensor_wrapper import TensorWrapper
from megengine.functional.distributed import remote_recv, remote_send


def _elwise(mode):
    op = Elemwise(mode=mode)

    def f(*args):
        (result,) = apply(op, *args)
        return result

    return f


add = _elwise("add")
mul = _elwise("mul")
cos = _elwise("cos")
relu = _elwise("relu")


def as_tensor(x):
    return Tensor(as_raw_tensor(x, device=mge.device.get_default_device()))


def save_to(self, name="grad"):
    def callback(tensor, grad):
        setattr(self, name, grad)

    return callback


@pytest.mark.isolated_distributed
@pytest.mark.skipif(
    platform.system() == "Windows", reason="windows disable MGB_ENABLE_OPR_MM"
)
def test_dist_grad():
    world_size = 2
    x_np = np.random.rand(10).astype("float32")
    port = dist.get_free_ports(1)[0]
    server = dist.Server(port)

    def worker0():
        dist.init_process_group("localhost", port, world_size, 0, 0)
        mge.device.set_default_device("gpu0")
        grad = Grad()

        x = as_tensor(x_np)
        grad.wrt(x, callback=save_to(x))
        # need a placeholder to trace operator
        send_x = remote_send(x, 1)
        recv_x = remote_recv(1, x_np.shape, x_np.dtype, "gpu0")
        y = recv_x * recv_x

        grad([y], [as_tensor(np.ones_like(x_np))])
        np.testing.assert_almost_equal(x.grad.numpy(), x.numpy() * 2)

    def worker1():
        dist.init_process_group("localhost", port, world_size, 1, 1)
        mge.device.set_default_device("gpu1")
        grad = Grad()

        recv_x = remote_recv(0, x_np.shape, x_np.dtype, "gpu1")
        send_x = remote_send(recv_x, 0)

        grad([], [])

        # sync because grad has a send operator
        sync()
        send_x.device._cn._sync_all()

    import multiprocessing as mp

    p0 = mp.Process(target=worker0)
    p1 = mp.Process(target=worker1)
    p0.start()
    p1.start()
    p0.join(10)
    p1.join(10)
    assert p0.exitcode == 0 and p1.exitcode == 0


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
    x = TensorWrapper(x_np)

    grad = Grad().wrt(x, callback=save_to(x))

    y = mul(x, x)
    y = mul(y, y)

    grad(y, TensorWrapper(np.ones_like(x_np)))
    np.testing.assert_almost_equal(x.grad.numpy(), 4 * x_np ** 3, decimal=6)


def test_release():
    def check(f):
        n = 0
        d = None
        for i in range(3):
            f()
            m = len(gc.get_objects())
            d = m - n
            n = m
        assert d == 0

    x = TensorWrapper([0.0])
    dy = TensorWrapper(np.ones_like(x.numpy()))

    @check
    def _():
        g = Grad().wrt(x)
        y = x * x
        g(y, dy)

    @check
    def _():
        with Grad().wrt(x) as g:
            pass

    @check
    def _():
        with Grad().wrt(x) as g:
            y = x * x


def test_grad_inplace():
    x_np = np.random.rand(10).astype("float32")
    x = TensorWrapper(x_np)

    grad = Grad().wrt(x, callback=save_to(x))

    y = mul(x, x)
    y *= y

    grad(y, TensorWrapper(np.ones_like(x_np)))
    np.testing.assert_almost_equal(x.grad.numpy(), 4 * x_np ** 3, decimal=6)


def test_elemwise_add():
    x_np = np.random.rand(10).astype("float32")
    y_np = np.random.rand(10, 10).astype("float32")
    dz_np = np.random.rand(10, 10).astype("float32")
    x = TensorWrapper(x_np)
    y = TensorWrapper(y_np)
    dz = TensorWrapper(dz_np)

    refs = {}

    def f(x, y):
        x = x * 2
        refs["x"] = weakref.ref(x.__wrapped__)
        refs["y"] = weakref.ref(y.__wrapped__)
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
    x = TensorWrapper(x_np)
    dz = TensorWrapper(dz_np)

    refs = {}

    def f(x):
        x = x * 2
        refs["x"] = weakref.ref(x.__wrapped__)
        return relu(x)

    grad = Grad().wrt(x, callback=save_to(x))

    z = f(x)

    assert refs["x"]() is None

    grad(z, dz)
    np.testing.assert_almost_equal(x.grad.numpy(), [2.0, 0])


def test_elemwise_relu_backward_fn():
    op = Elemwise(mode="relu").to_c()
    attr = TensorAttr()
    attr.dtype = "float32"
    attr.comp_node = "xpux"
    result = imperative.make_backward_graph(op, [attr], [True], [True])
    backward_graph, save_for_backward_mask, input_has_grad = result
    assert save_for_backward_mask == [False, True, True], save_for_backward_mask


def test_reshape():
    x_np = np.random.rand(2, 5).astype("float32")
    x = TensorWrapper(x_np)

    grad = Grad().wrt(x, callback=save_to(x))
    y = x.reshape(5, 2)

    grad(y, F.ones_like(y))
    np.testing.assert_equal(np.ones((2, 5), dtype=np.float32), x.grad.numpy())


def test_subtensor():
    x_np = np.random.rand(3, 3).astype("float32")
    x = TensorWrapper(x_np)

    grad = Grad().wrt(x, callback=save_to(x))
    y = x[1:-1, :2]

    grad(y, F.ones_like(y))
    np.testing.assert_equal(
        np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32), x.grad.numpy()
    )


def test_IndexingMultiAxisVec():
    x_np = np.random.rand(3, 3).astype("float32")
    x = TensorWrapper(x_np)

    grad = Grad().wrt(x, callback=save_to(x))
    y = x[[0, 2], [0, 2]]

    grad(y, F.ones_like(y))
    np.testing.assert_equal(
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.float32), x.grad.numpy()
    )


def test_AxisAddRemove():
    x_np = np.random.rand(1, 5).astype("float32")
    x = TensorWrapper(x_np)

    grad = Grad().wrt(x, callback=save_to(x))
    y = F.squeeze(F.expand_dims(x, 2), 0)

    grad(y, F.ones_like(y))
    np.testing.assert_equal(
        np.array([[1, 1, 1, 1, 1]], dtype=np.float32), x.grad.numpy()
    )


def test_Broadcast():
    x_np = np.random.rand(3, 3, 1).astype("float32")
    x = TensorWrapper(x_np)

    grad = Grad().wrt(x, callback=save_to(x))
    y = F.broadcast(x, (3, 3, 10))

    grad(y, F.ones_like(y))
    np.testing.assert_equal(np.ones((3, 3, 1), dtype=np.float32) * 10, x.grad.numpy())


def test_Reduce_sum():
    x_np = np.random.rand(3, 3).astype("float32")
    x = TensorWrapper(x_np)

    grad = Grad().wrt(x, callback=save_to(x))
    y = x.sum(axis=0)

    grad(y, F.ones_like(y))
    np.testing.assert_equal(np.ones((3, 3), dtype=np.float32), x.grad.numpy())


def test_Reduce_mean():
    x_np = np.random.rand(3, 3).astype("float32")
    x = TensorWrapper(x_np)

    grad = Grad().wrt(x, callback=save_to(x))
    y = x.mean(axis=0)

    grad(y, F.ones_like(y))
    np.testing.assert_equal(np.ones((3, 3), dtype=np.float32) / 3, x.grad.numpy())
