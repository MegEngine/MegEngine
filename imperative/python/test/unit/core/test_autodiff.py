# -*- coding: utf-8 -*-
import gc

import numpy as np
import pytest

import megengine as mge
import megengine.distributed as dist
import megengine.functional as F
import megengine.module as M
from megengine import Tensor
from megengine.core import _imperative_rt
from megengine.core._imperative_rt import CompNode, TensorAttr, imperative
from megengine.core._imperative_rt.core2 import TensorWeakRef, apply, sync
from megengine.core.autodiff.grad import Grad
from megengine.core.ops import builtin
from megengine.core.ops.builtin import Elemwise, Identity
from megengine.functional.distributed import remote_recv, remote_send
from megengine.functional.tensor import ones, zeros


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


@pytest.mark.require_ngpu(2)
@pytest.mark.isolated_distributed
def test_dist_grad():
    world_size = 2
    x_np = np.random.rand(10).astype("float32")

    @dist.launcher
    def worker():
        rank = dist.get_rank()
        if rank == 0:
            with Grad() as grad:
                x = as_tensor(x_np)
                grad.wrt(x, callback=save_to(x))
                # need a placeholder to trace operator
                remote_send(x, 1)
                recv_x = remote_recv(1)
                y = recv_x * recv_x
                grad([y], [as_tensor(np.ones_like(x_np))])
            np.testing.assert_almost_equal(x.grad.numpy(), x.numpy() * 2)
        elif rank == 1:
            with Grad() as grad:
                recv_x = remote_recv(0)
                remote_send(recv_x, 0)
                grad([], [])

    worker()


def test_grad():
    x_np = np.random.rand(10).astype("float32")
    x = as_tensor(x_np)

    with Grad() as grad:
        grad.wrt(x, callback=save_to(x))
        y = cos(x)
        grad(y, as_tensor(np.ones_like(x_np)))

    np.testing.assert_almost_equal(x.grad.numpy(), -np.sin(x_np))


def test_grad_2():
    x_np = np.random.rand(10).astype("float32")
    x = as_tensor(x_np)

    with Grad() as grad:
        grad.wrt(x, callback=save_to(x))
        y = mul(x, x)
        y = mul(y, y)
        grad(y, as_tensor(np.ones_like(x_np)))

    np.testing.assert_almost_equal(x.grad.numpy(), 4 * x_np ** 3, decimal=6)


@pytest.mark.require_higher_order_directive()
def test_2nd_grad():
    x_np = np.random.rand(10).astype("float32")
    x = as_tensor(x_np)
    ones = as_tensor(np.ones_like(x_np))

    with Grad("grad2") as grad2:
        with Grad("grad") as grad:
            grad2.wrt(x, callback=save_to(x))
            grad.wrt(x, callback=save_to(x))
            y = cos(x)
            grad(y, ones)
            z = x.grad
            np.testing.assert_almost_equal(x.grad.numpy(), -np.sin(x_np), decimal=5)

        x.grad = None
        grad2(z, ones)

        np.testing.assert_almost_equal(x.grad.numpy(), -np.cos(x_np), decimal=5)


def test_grad_with_tensor_wrapper():
    x_np = np.random.rand(10).astype("float32")
    x = mge.Tensor(x_np)

    with Grad() as grad:
        grad.wrt(x, callback=save_to(x))
        y = mul(x, x)
        y = mul(y, y)
        grad(y, mge.Tensor(np.ones_like(x_np)))

    np.testing.assert_almost_equal(x.grad.numpy(), 4 * x_np ** 3, decimal=6)


def test_wrt_intermediate_var():
    x_np = np.random.rand(10).astype("float32")
    x = mge.Tensor(x_np)

    result = {}

    with Grad() as grad:
        grad.wrt(x, callback=lambda dx: result.update(dx=dx))
        y = mul(x, x)
        grad.wrt(y, callback=lambda dy: result.update(dy=dy))
        z = mul(y, y)
        grad(z, mge.Tensor(np.ones_like(x_np)))

    np.testing.assert_almost_equal(result["dx"].numpy(), 4 * x_np ** 3, decimal=6)
    np.testing.assert_almost_equal(result["dy"].numpy(), 2 * (x_np ** 2), decimal=6)


@pytest.mark.parametrize("in_path", [False, True])
def test_wrt_visibility(in_path):
    x_np = np.random.rand(10).astype("float32")
    x = mge.Tensor(x_np)

    def copy(x):
        xx = mge.Tensor(x)
        xx._reset(x)
        return xx

    result = {}

    with Grad() as grad:
        if in_path:
            grad.wrt(x, callback=lambda _: None)
        y = mul(x, x)
        grad.wrt(copy(y), callback=lambda dy: result.update(dy=dy))
        z = mul(y, y)
        grad(z, mge.Tensor(np.ones_like(x_np)))

    assert not result


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
        with Grad() as g:
            g.wrt(x)
            y = x * x
            g(y, dy)

    @check
    def _():
        with Grad() as g:
            g.wrt(x)
            pass

    @check
    def _():
        with Grad() as g:
            g.wrt(x)
            y = x * x


def test_grad_inplace():
    x_np = np.random.rand(10).astype("float32")
    x = mge.Tensor(x_np)

    with Grad() as grad:
        grad.wrt(x, callback=save_to(x))
        y = mul(x, x)
        y *= y
        grad(y, mge.Tensor(np.ones_like(x_np)))

    np.testing.assert_almost_equal(x.grad.numpy(), 4 * x_np ** 3, decimal=6)


def test_identity():
    x_np = np.random.rand(10).astype("float32")
    x = mge.Tensor(x_np)
    dy_np = np.random.rand(*x.shape).astype("float32")
    dy = mge.Tensor(dy_np)

    with Grad() as grad:
        grad.wrt(x, callback=save_to(x))
        (y,) = apply(Identity(), x)
        grad(y, dy)

    np.testing.assert_array_equal(x.grad.numpy(), dy_np)


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

    with Grad() as grad:
        grad.wrt(x, callback=save_to(x))
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

    with Grad() as grad:
        grad.wrt(x, callback=save_to(x))
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

    with Grad() as grad:
        grad.wrt(x, callback=save_to(x))
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

    with Grad() as grad:
        grad.wrt(x, callback=save_to(x))
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

    with Grad() as grad:
        grad.wrt(x, callback=save_to(x))
        refs = {}

        def f(x):
            x = x * 1
            y = x[[0, 0, 2, 1], [2, 2, 1, 0]]
            refs["x"] = TensorWeakRef(x)
            return y

        y = f(x)
        for _, r in refs.items():
            assert r() is None
        grad(y, F.ones_like(y))

    np.testing.assert_equal(
        np.array([[0, 0, 2], [1, 0, 0], [0, 1, 0]], dtype=np.float32), x.grad.numpy()
    )


def test_AxisAddRemove():
    x_np = np.random.rand(1, 5).astype("float32")
    x = mge.Tensor(x_np)

    with Grad() as grad:
        grad.wrt(x, callback=save_to(x))
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

    with Grad() as grad:
        grad.wrt(x, callback=save_to(x))
        y = F.broadcast_to(x, (3, 3, 10))
        grad(y, F.ones_like(y))

    np.testing.assert_equal(np.ones((3, 3, 1), dtype=np.float32) * 10, x.grad.numpy())


def test_interpolate_fastpath():
    x_np = np.random.rand(3, 3, 32, 32).astype("float32")
    x = mge.Tensor(x_np)

    with Grad() as grad:
        grad.wrt(x, callback=save_to(x))
        y = F.vision.interpolate(x, size=(16, 16), mode="bilinear")
        grad(y, F.ones_like(y))

    np.testing.assert_equal(np.ones(x_np.shape, dtype=np.float32) / 4, x.grad.numpy())


def test_Reduce_sum():
    x_np = np.random.rand(3, 3).astype("float32")
    x = mge.Tensor(x_np)

    with Grad() as grad:
        grad.wrt(x, callback=save_to(x))
        y = x.sum(axis=0)
        grad(y, F.ones_like(y))

    np.testing.assert_equal(np.ones((3, 3), dtype=np.float32), x.grad.numpy())


def test_Reduce_mean():
    x_np = np.random.rand(3, 3).astype("float32")
    x = mge.Tensor(x_np)

    with Grad() as grad:
        grad.wrt(x, callback=save_to(x))
        y = x.mean(axis=0)
        grad(y, F.ones_like(y))

    np.testing.assert_equal(np.ones((3, 3), dtype=np.float32) / 3, x.grad.numpy())


def test_addAxis():
    x_np = np.random.rand(3, 3).astype("float32")
    x = mge.Tensor(x_np)

    with Grad() as grad:
        grad.wrt(x, callback=save_to(x))
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

    with Grad() as grad:
        grad.wrt(x, callback=save_to(x))
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


def test_dot():
    x = np.random.rand(2, 2).astype("float32")
    x = mge.Tensor(x)
    u = F.ones((2,))
    v = F.ones((2,))

    with Grad() as grad:
        grad.wrt(x, callback=save_to(x))

        def f(x):
            return F.dot(u, F.matmul(x, v))

        y = f(x)
        grad(y, F.ones_like(y))

    np.testing.assert_equal(np.ones((2, 2), dtype=np.float32), x.grad.numpy())


def test_pixel_shuffle():

    x = np.random.rand(2, 3, 16, 3, 4).astype("float32")
    x = mge.Tensor(x)
    with Grad() as grad:
        grad.wrt(x, callback=save_to(x))

        def f(x):
            p = F.pixel_shuffle(x, 2)
            return p * p

        y = f(x)
        grad(y, F.ones_like(y))
    np.testing.assert_equal(2 * x.numpy(), x.grad.numpy())


def test_matmul():
    def test_one(xdim, ydim, transposeA, transposeB):
        xshape = (1, 4) if xdim == 1 else (2,) * (xdim - 2) + (3, 4)
        yshape = (4, 1) if ydim == 1 else (2,) * (ydim - 2) + (4, 5)
        x = np.random.rand(*xshape).astype("float32")
        y = np.random.rand(*yshape).astype("float32")
        gshape = (x @ y).shape
        g = np.random.rand(*gshape).astype("float32")
        dx = g @ np.swapaxes(y, -1, -2)
        dy = np.swapaxes(x, -1, -2) @ g
        while dx.shape != x.shape:
            dx = dx.sum(0)
        while dy.shape != y.shape:
            dy = dy.sum(0)
        if transposeA:
            x = np.swapaxes(x, -1, -2)
            dx = np.swapaxes(dx, -1, -2)
        if transposeB:
            y = np.swapaxes(y, -1, -2)
            dy = np.swapaxes(dy, -1, -2)
        x = mge.Tensor(x.squeeze())
        y = mge.Tensor(y.squeeze())
        g = mge.Tensor(g.squeeze())
        with Grad() as grad:
            grad.wrt(x, callback=save_to(x))
            grad.wrt(y, callback=save_to(y))
            z = F.matmul(x, y, transpose_a=transposeA, transpose_b=transposeB)
            grad(z, g)
        np.testing.assert_almost_equal(dx.squeeze(), x.grad.numpy(), decimal=5)
        np.testing.assert_almost_equal(dy.squeeze(), y.grad.numpy(), decimal=5)

    for xdim in [1, 2, 3, 4]:
        for ydim in [1, 2, 3, 4]:
            for transposeA in [False, True]:
                if xdim == 1 and transposeA == True:
                    continue
                for transposeB in [False, True]:
                    if ydim == 1 and transposeB == True:
                        continue
                    test_one(xdim, ydim, transposeA, transposeB)


def test_indexing():
    x = np.array([[1.0, 2.0]]).astype("float32")
    x = mge.Tensor(x)
    index = mge.Tensor([0])

    with Grad() as grad:
        grad.wrt(x, callback=save_to(x))

        def f(x):
            return F.indexing_one_hot(x, index, -1)

        y = f(x)
        grad(y, F.ones_like(y))

    np.testing.assert_equal(np.array([[1, 0]], dtype=np.float32), x.grad.numpy())


def test_indexing_set_one_hot():
    x = mge.tensor(np.arange(1, 4, dtype=np.int32))

    with Grad() as grad:
        zeros_tensor = zeros((3, 4), dtype=x.dtype, device=x.device)
        ones_tensor = ones((3, 1), dtype=x.dtype, device=x.device)

        grad.wrt(zeros_tensor, callback=save_to(zeros_tensor))
        grad.wrt(ones_tensor, callback=save_to(ones_tensor))

        def f(x):
            op = builtin.IndexingSetOneHot(axis=x.ndim, ndim=x.ndim)
            (result,) = apply(op, zeros_tensor, x, ones_tensor)
            return result

        y = f(x)
        grad(y, F.ones_like(y))
        np.testing.assert_equal(
            np.array([[1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]], dtype=np.int32),
            zeros_tensor.grad.numpy(),
        )
        np.testing.assert_equal(
            np.array([[1], [1], [1]], dtype=np.int32), ones_tensor.grad.numpy(),
        )
