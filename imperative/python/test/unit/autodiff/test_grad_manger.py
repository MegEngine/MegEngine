# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import os
import platform
import weakref

import numpy as np
import pytest

import megengine as mge
import megengine.distributed as dist
import megengine.functional as F
import megengine.module as M
import megengine.optimizer as optim
from megengine.autodiff import GradManager
from megengine.distributed.helper import get_device_count_by_fork
from megengine.jit import trace


def test_basic():
    x = mge.tensor([1.0, 3.0, 5.0]).reshape(1, 3)
    w = mge.tensor([2.0, 4.0, 6.0]).reshape(3, 1)
    b = mge.tensor(-1.0)

    gm = GradManager().attach([w, b])
    gm.record()

    p = F.matmul(x, w)
    y = p + b

    gm.backward(y)
    gm.release()  # is not necessary
    np.testing.assert_equal(w.grad.numpy(), [[1], [3], [5]])
    np.testing.assert_equal(b.grad.numpy(), [1])

    w.grad = None
    b.grad = None
    with gm:
        p = F.matmul(x, w)
        y = p + b
        gm.backward(y)

    np.testing.assert_equal(w.grad.numpy(), [[1], [3], [5]])
    np.testing.assert_equal(b.grad.numpy(), [1])


def test_attach_in_with_block():
    a = mge.Parameter([1.0])
    gm = GradManager()
    with gm:
        b = a * 3
        gm.attach(b)
        c = b + 1
        gm.backward(c)
    assert int(b.grad.numpy()) == 1


def test_attach_temporary():
    w = mge.Parameter(2.0)
    gm = GradManager()
    gm.attach(w)

    def cb(x, g):
        assert x is ref()
        cb.called = True

    for i in range(3):
        with gm:
            cb.called = False
            x = mge.Tensor(i, dtype="float32")
            gm.attach(x, callbacks=cb)
            ref = weakref.ref(x)
            y = x * w
            gm.backward(y)
            assert cb.called
        del x
        assert ref() is None

    # NOTE: does not guarantee timely release when recording
    # for i in range(3):
    #     with gm:
    #         x = mge.Tensor(i, dtype='float32')
    #         gm.attach(x)
    #         ref = weakref.ref(x)
    #         y = x * w
    #         del x
    #         assert ref() is None
    #         gm.backward(y)


def test_no_dependency():
    x = mge.tensor(3)

    w = mge.Parameter(1.0)
    w_no_dep = mge.Parameter(1.0)
    gm = GradManager()
    gm.attach(w)
    gm.attach(w_no_dep)

    with gm:
        out1 = x * w
        out2 = w_no_dep * out1
        gm.backward(out1.sum())

    assert w.grad is not None
    assert w_no_dep.grad is None


def test_regression_1762():
    x = F.ones((10, 10, 3, 3))

    conv = M.Conv2d(10, 10, kernel_size=3, padding=1)

    t_shape = (1, 10, 1, 1)
    weight = mge.Parameter(np.ones(t_shape, dtype=np.float32))
    bias = mge.Parameter(np.zeros(t_shape, dtype=np.float32))

    gm = GradManager()
    gm.attach(list(conv.parameters()) + [weight, bias])

    with gm:
        out1 = conv(x)

        out2 = F.batch_norm(out1, None, None, weight, bias, training=True,)

        # Weird error only occur when this action is placed after BN
        # Op type is not relevant
        loss = out1 + 1
        gm.backward(loss)


@pytest.mark.require_ngpu(2)
@pytest.mark.isolated_distributed
@pytest.mark.parametrize(
    "trace_mode", [True, False, None], ids=["symbolic", "trace", "no_trace"]
)
def test_remote_grad(trace_mode):
    @dist.launcher
    def worker():
        rank = dist.get_rank()
        size = dist.get_world_size()
        x = mge.tensor(np.random.randn(1, rank * 2 + 2), dtype=np.float32)
        m = M.Linear(rank * 2 + 2, rank * 2 + 4)
        gm = GradManager().attach(m.parameters())
        opt = optim.SGD(m.parameters(), 1e-3, momentum=0.9)

        def train_func(x):
            with gm:
                if rank != 0:
                    x = dist.functional.remote_recv(rank - 1)
                y = m(x)
                if rank != size - 1:
                    dist.functional.remote_send(y, dest_rank=rank + 1)
                    gm.backward()
                else:
                    y = y.mean()
                    gm.backward(y)
                opt.step().clear_grad()

        if trace_mode is not None:
            train_func = trace(symbolic=trace_mode)(train_func)

        for i in range(3):
            train_func(x)

    worker()


@pytest.mark.require_ngpu(3)
@pytest.mark.isolated_distributed
@pytest.mark.parametrize(
    "trace_mode", [True, False, None], ids=["symbolic", "trace", "no_trace"]
)
def test_gather_grad(trace_mode):
    @dist.launcher(n_gpus=3)
    def worker():
        m = M.Linear(10, 10)
        x = F.ones([3, 10], dtype="float32")

        def func():
            with GradManager().attach(m.parameters()) as gm:
                y = m(x)
                y = F.distributed.gather(y)
                if dist.get_rank() == 0:
                    loss = (2 * y + 1).mean()
                    gm.backward(loss)
                else:
                    gm.backward()

        if trace_mode is not None:
            func = trace(symbolic=trace_mode)(func)
        func()

    worker()


@pytest.mark.require_ngpu(3)
@pytest.mark.isolated_distributed
@pytest.mark.parametrize(
    "trace_mode", [True, False, None], ids=["symbolic", "trace", "no_trace"]
)
def test_scatter_grad(trace_mode):
    @dist.launcher(n_gpus=3)
    def worker():
        x = F.ones([3, 10], dtype="float32")
        m = M.Linear(10, 10)

        def func():
            with GradManager().attach(m.parameters()) as gm:
                if dist.get_rank() == 0:
                    y = m(x)
                else:
                    y = x
                y = F.distributed.scatter(y)
                gm.backward(y)

        if trace_mode is not None:
            func = trace(symbolic=trace_mode)(func)
        func()

    worker()


@pytest.mark.require_ngpu(3)
@pytest.mark.isolated_distributed
@pytest.mark.parametrize(
    "trace_mode", [True, False, None], ids=["symbolic", "trace", "no_trace"]
)
def test_reduce_grad(trace_mode):
    @dist.launcher(n_gpus=3)
    def worker():
        m = M.Linear(10, 10)
        x = F.ones([3, 10], dtype="float32")

        def func():
            with GradManager().attach(m.parameters()) as gm:
                y = m(x)
                y = F.distributed.reduce_sum(y)
                if dist.get_rank() == 0:
                    loss = (2 * y + 1).mean()
                    gm.backward(loss)
                else:
                    gm.backward()

        if trace_mode is not None:
            func = trace(symbolic=trace_mode)(func)
        func()

    worker()


@pytest.mark.require_ngpu(3)
@pytest.mark.isolated_distributed
@pytest.mark.parametrize(
    "trace_mode", [True, False, None], ids=["symbolic", "trace", "no_trace"]
)
def test_broadcast_grad(trace_mode):
    @dist.launcher(n_gpus=3)
    def worker():
        x = F.ones([3, 10], dtype="float32")
        m = M.Linear(10, 10)

        def func():
            with GradManager().attach(m.parameters()) as gm:
                if dist.get_rank() == 0:
                    y = m(x)
                else:
                    y = x
                y = F.distributed.broadcast(y)
                gm.backward(y)

        if trace_mode is not None:
            func = trace(symbolic=trace_mode)(func)
        func()

    worker()


@pytest.mark.require_higher_order_directive()
def test_2nd_grad_with_manager():
    x_np = np.random.rand(10).astype("float32")
    x = mge.tensor(x_np)

    gm = GradManager().attach([x])
    gm2 = GradManager().attach([x])

    with gm:
        with gm2:
            y = F.cos(x)
            gm2.backward(y)
        np.testing.assert_almost_equal(x.grad.numpy(), -np.sin(x_np), decimal=5)
        gm.backward(x.grad)
    np.testing.assert_almost_equal(
        x.grad.numpy(), -np.sin(x_np) - np.cos(x_np), decimal=5
    )


@pytest.mark.require_higher_order_directive()
def test_grad_manager_group():
    x_np = np.random.rand(10).astype("float32")
    x = mge.tensor(x_np)

    gm = GradManager().attach([x])
    gm2 = GradManager().attach([x])

    with gm | gm2:
        y = F.cos(x)
        gm.backward(y)
        gm2.backward(y)
    np.testing.assert_almost_equal(x.grad.numpy(), -2 * np.sin(x_np), decimal=5)

    x.grad = None


@pytest.mark.require_higher_order_directive()
def test_grad_manager_group_visibility():
    x_np = np.random.rand(10).astype("float32")
    x = mge.tensor(x_np)

    gm = GradManager().attach([x])
    gm2 = GradManager().attach([x])

    with gm | gm2:
        y = F.cos(x)
        gm2.backward(y)
        np.testing.assert_almost_equal(x.grad.numpy(), -np.sin(x_np), decimal=5)
        gm.backward(x.grad)
        np.testing.assert_almost_equal(x.grad.numpy(), -np.sin(x_np), decimal=5)


@pytest.mark.require_higher_order_directive()
def test_grad_manager_visibility_by_order():
    x_np = np.random.rand(10).astype("float32")
    x = mge.tensor(x_np)

    gm = GradManager().attach([x])
    gm2 = GradManager().attach([x])

    with gm2:
        with gm:
            y = F.cos(x)
            gm2.backward(y)
            np.testing.assert_almost_equal(x.grad.numpy(), -np.sin(x_np), decimal=5)
            gm.backward(x.grad)

    np.testing.assert_almost_equal(x.grad.numpy(), -np.sin(x_np), decimal=5)
