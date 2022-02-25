import os
import platform
import weakref

import numpy as np
import pytest

import megengine as mge
import megengine.core.tensor.dtype as dtype
import megengine.distributed as dist
import megengine.functional as F
import megengine.module as M
import megengine.optimizer as optim
from megengine.autodiff import Function, GradManager
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


def test_dy():
    x = mge.tensor([1.0, 3.0, 5.0]).reshape(1, 3)
    w = mge.tensor([2.0, 4.0, 6.0]).reshape(3, 1)
    b = mge.tensor(-1.0)

    gm = GradManager().attach([w, b])

    def get_grad(grad, dy, idx):
        if isinstance(dy, (list, tuple)):
            return np.array(grad) * dy[idx]
        else:
            return np.array(grad) * dy

    # dy's shape should be the same as y's
    dy = mge.tensor(2.5).reshape(1, 1)
    w.grad = None
    b.grad = None
    with gm:
        p = F.matmul(x, w)
        y = p + b
        gm.backward(y, dy=dy)

    np.testing.assert_equal(w.grad.numpy(), [[1], [3], [5]] * dy.numpy())
    np.testing.assert_equal(b.grad.numpy(), [1] * dy.numpy())


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


def test_attached_tensors():
    w1 = mge.Parameter(2.0)
    w2 = mge.Parameter(2.0)
    gm = GradManager()

    def check(expected):
        actual = gm.attached_tensors()
        assert len(expected) == len(actual)
        for exp, act in zip(expected, actual):
            assert exp is act

    gm.attach(w1)
    check([w1])
    gm.attach(w2)
    check([w1, w2])
    gm.attach(w1)
    check([w1, w2])


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


def test_empty_grad_in_backward():
    x = mge.Parameter(F.full(100, 0.5))
    y = mge.Parameter(F.ones(100))

    gm = GradManager()
    gm.attach([x, y])

    with gm:
        z = F.where(x > 0.7, x, y)
        loss = z.sum()
        gm.backward(loss)
        assert np.all(x.grad.numpy() == 0)
        assert np.all(y.grad.numpy() == 1)


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
                    x = dist.functional.remote_send(y, dest_rank=rank + 1)
                    gm.backward()
                else:
                    y = y.mean()
                    gm.backward(y)
                opt.step().clear_grad()

        if trace_mode is not None:
            train_func = trace(symbolic=trace_mode)(train_func)

        for i in range(1):
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


@pytest.mark.parametrize("target", [F.cos, F.sin, lambda x: x * 2 + 1])
def test_emulate_forward_mode_with_reverse_mode(target):
    def jvp(inp, expr):
        with GradManager() as gm:
            with GradManager().attach([inp]) as gm2:
                oup = expr(inp)
                oup_grad = F.zeros_like(oup)
                gm.attach(oup_grad)
                gm2.backward(oup, oup_grad)
            gm.backward(inp.grad)
        return oup, oup_grad.grad

    def fake_jvp(inp, expr):
        delta = 0.001
        return expr(inp), (expr(inp + delta) - expr(inp - delta)) / (2 * delta)

    x_np = np.random.rand(10).astype("float32")
    x = mge.tensor(x_np)
    y, dy = jvp(x, target)
    y1, dy1 = fake_jvp(x, target)

    np.testing.assert_almost_equal(y.numpy(), y1.numpy(), decimal=5)
    np.testing.assert_almost_equal(dy.numpy(), dy1.numpy(), decimal=3)


def test_2nd_grad_with_custom_gradient():
    class MySin(Function):
        def forward(self, x):
            self.inp = x
            x = mge.Tensor(x.numpy())
            y = F.sin(x)
            return y

        def backward(self, dy):
            dx = F.cos(self.inp) * dy
            return dx

    class MyCos(Function):
        def forward(self, x):
            self.inp = x
            x = mge.Tensor(x.numpy())
            y = F.cos(x)
            return y

        def backward(self, dy):
            if dy is None:
                return None
            dx = -MySin()(self.inp) * dy
            return dx

    x_np = np.random.rand(10).astype("float32")
    x = mge.tensor(x_np)

    gm = GradManager().attach([x])
    gm2 = GradManager().attach([x])

    with gm:
        with gm2:
            y = MyCos()(x)
            gm2.backward(y)
        np.testing.assert_almost_equal(x.grad.numpy(), -np.sin(x_np), decimal=5)
        gm.backward(x.grad)
    np.testing.assert_almost_equal(
        x.grad.numpy(), -np.sin(x_np) - np.cos(x_np), decimal=5
    )


@pytest.mark.parametrize("invalid_dtype", [np.uint8, np.int8, np.int32])
def test_attach_invalid_tensor_dtype(invalid_dtype):
    gm = GradManager()
    x = mge.tensor([1], dtype=invalid_dtype)
    with pytest.raises(AssertionError):
        gm.attach([x])


@pytest.mark.parametrize("differentible_dtype", [np.float32, np.float16])
def test_attach_differentible_tensor_dtype(differentible_dtype):
    gm = GradManager()
    x = mge.tensor([1], dtype=differentible_dtype)
    gm.attach([x])
