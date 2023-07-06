import platform

import numpy as np
import pytest

import megengine as mge
import megengine.functional as F
import megengine.jit as jit
import megengine.tensor as tensor
from megengine import autodiff, is_cuda_available
from megengine.autodiff.grad_manager import GradManager


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_conv2d():
    np.random.seed(123)
    mge.random.seed(123)

    def tester(x_shape, w_shape, b_shape, stride, padding, groups, dtype=None):
        dtype = dtype or np.float32
        x = tensor(0.1 * np.random.rand(*x_shape), dtype=dtype)
        w = tensor(0.1 * np.random.rand(*w_shape), dtype=dtype)
        b = tensor(0.1 * np.random.rand(*b_shape), dtype=dtype) if b_shape else None
        y = F.conv2d(x, w, b, stride=stride, padding=padding, groups=groups)
        dy = tensor(0.1 * np.random.rand(*y.shape), dtype=dtype)

        gm = GradManager()

        if b is not None:

            @jit.xla_trace(without_host=True)
            def func(x, w, b, dy):
                gm.attach([x, w, b])
                with gm:
                    y = F.conv2d(x, w, b, stride=stride, padding=padding, groups=groups)
                    gm.backward(y, dy)
                return [y, x.grad, w.grad, b.grad]

            mge_rsts = func(x, w, b, dy)
            xla_rsts = func(x, w, b, dy)
        else:

            @jit.xla_trace(without_host=True)
            def func(x, w, dy):
                gm.attach([x, w])
                with gm:
                    y = F.conv2d(x, w, stride=stride, padding=padding, groups=groups)
                    gm.backward(y, dy)
                return [y, x.grad, w.grad]

            mge_rsts = func(x, w, dy)
            xla_rsts = func(x, w, dy)

        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

    tester(
        (4, 16, 24, 24), (32, 16, 3, 3), (1, 32, 1, 1), stride=1, padding=1, groups=1
    )
    tester(
        (4, 16, 24, 24),
        (32, 16, 3, 3),
        (1, 32, 1, 1),
        stride=(2, 3),
        padding=(2, 1),
        groups=1,
    )
    tester(
        (4, 16, 24, 24),
        (16, 1, 1, 3, 3),
        None,
        stride=(2, 3),
        padding=(2, 1),
        groups=16,
    )

    tester((4, 16, 24, 24), (32, 16, 1, 1), None, stride=1, padding=1, groups=1)
    tester(
        (4, 16, 1, 1),
        (32, 16, 1, 1),
        (1, 32, 1, 1),
        stride=(2, 3),
        padding=(2, 1),
        groups=1,
    )
    tester(
        (4, 16, 24, 24),
        (16, 1, 1, 1, 1),
        (1, 16, 1, 1),
        stride=(2, 3),
        padding=(2, 1),
        groups=16,
    )
    tester(
        (4, 16, 24, 24),
        (4, 4, 4, 1, 1),
        (1, 16, 1, 1),
        stride=(2, 3),
        padding=(2, 1),
        groups=4,
    )


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_conv_transpose2d():
    np.random.seed(123)
    mge.random.seed(123)

    def tester(x_shape, w_shape, b_shape, stride, padding, groups, dtype=None):
        dtype = dtype or np.float32
        x = tensor(0.1 * np.random.rand(*x_shape), dtype=dtype)
        w = tensor(0.1 * np.random.rand(*w_shape), dtype=dtype)
        b = tensor(0.1 * np.random.rand(*b_shape), dtype=dtype) if b_shape else None
        y = F.conv_transpose2d(x, w, b, stride=stride, padding=padding, groups=groups)
        dy = tensor(0.1 * np.random.rand(*y.shape), dtype=dtype)

        gm = GradManager()

        if b is not None:

            @jit.xla_trace(without_host=True)
            def func(x, w, b, dy):
                gm.attach([x, w, b])
                with gm:
                    y = F.conv_transpose2d(
                        x, w, b, stride=stride, padding=padding, groups=groups
                    )
                    gm.backward(y, dy)
                return [y, x.grad, w.grad, b.grad]

            mge_rsts = func(x, w, b, dy)
            xla_rsts = func(x, w, b, dy)
        else:

            @jit.xla_trace(without_host=True)
            def func(x, w, dy):
                gm.attach([x, w])
                with gm:
                    y = F.conv2d(x, w, stride=stride, padding=padding, groups=groups)
                    gm.backward(y, dy)
                return [y, x.grad, w.grad]

            mge_rsts = func(x, w, dy)
            xla_rsts = func(x, w, dy)

        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-4)

    tester(
        (4, 16, 24, 24), (16, 32, 3, 3), (1, 32, 1, 1), stride=1, padding=1, groups=1
    )
    tester(
        (4, 16, 24, 24),
        (16, 32, 3, 3),
        (1, 32, 1, 1),
        stride=(2, 3),
        padding=(2, 1),
        groups=1,
    )


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_adaptive_pooling():
    def tester(fpool, ishape, oshape, dtype=None):
        oshape = (oshape, oshape) if isinstance(oshape, int) else oshape
        dtype = dtype or np.float32

        x = tensor(np.random.randn(*ishape), dtype=dtype)
        dy = tensor(np.random.randn(*ishape[:-2], *oshape), dtype=dtype)
        gm = autodiff.GradManager()

        @jit.xla_trace(without_host=True)
        def func(x, dy):
            gm.attach([x])
            with gm:
                y = fpool(x, oshape)
                gm.backward(y, dy)
            return y, x.grad

        mge_rsts = func(x, dy)
        xla_rsts = func(x, dy)

        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

    for fpool in [F.adaptive_avg_pool2d, F.adaptive_max_pool2d]:
        for oshape in [(1, 1), (2, 2), 3, (4, 4), (2, 4), (5, 5), (5, 7)]:
            tester(fpool, (32, 16, 24, 24), oshape)
            tester(fpool, (32, 16, 17, 13), oshape)


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_pooling():
    def tester(fpool, ishape, kernel, stride, padding, dtype=None, **kwargs):
        oshape = fpool(
            tensor(np.random.randn(*ishape).astype("float32")), kernel, stride, padding
        ).shape
        x = tensor(np.random.randn(*ishape).astype("float32"))
        dy = tensor(np.random.randn(*oshape).astype("float32"))

        gm = autodiff.GradManager()

        @jit.xla_trace(without_host=True)
        def func(x, dy):
            gm.attach([x])
            with gm:
                y = fpool(x, kernel, stride, padding, **kwargs)
                gm.backward(y, dy)
            return y, x.grad

        mge_rsts = func(x, dy)
        xla_rsts = func(x, dy)

        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

    tester(F.max_pool2d, [32, 16, 8, 13], (3, 3), 2, 1)
    tester(F.avg_pool2d, [32, 16, 8, 13], (3, 1), (2, 1), (1, 0), mode="average")
    tester(F.avg_pool2d, [32, 16, 8, 2], (3, 3), 2, 1)


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_softmax():
    def tester(ishape, axis, dtype=None):
        dtype = dtype or np.float32
        x = tensor(np.random.randn(*ishape), dtype=dtype)
        dy = tensor(np.random.randn(*ishape), dtype=dtype)

        gm = autodiff.GradManager()

        @jit.xla_trace(without_host=True)
        def func(x, dy):
            gm.attach([x])
            with gm:
                y = F.softmax(x, axis=axis)
                gm.backward(y, dy)
            return y, x.grad

        mge_rsts = func(x, dy)
        xla_rsts = func(x, dy)

        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

    tester((32, 16, 8, 8), 1)
    tester((1, 16, 17, 128), [0, 2])
    tester((32, 16, 5), -2)
    tester((32, 16, 5), 0)
    tester((1, 16, 5), -1)
    tester((14, 1, 13, 5), 1)


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_loss():
    def tester(
        loss_fn,
        pred_shape,
        label_shape,
        label_type="default",
        atol=1e-5,
        dtype=None,
        **kwargs
    ):
        dtype = dtype or np.float32
        pred = tensor(np.random.randn(*pred_shape), dtype=dtype)
        if label_type == "default":
            label = tensor(np.random.randn(*label_shape), dtype=dtype)
        elif label_type == "classes":
            label = tensor(np.random.randint(0, 10, size=label_shape), dtype=dtype)
        dout = tensor(np.random.randn(1,), dtype=dtype)

        gm = autodiff.GradManager()

        @jit.xla_trace(without_host=True)
        def func(pred, label, dout):
            gm.attach([pred])
            with gm:
                out = loss_fn(pred, label, **kwargs)
                gm.backward(out, dout)
            return out, pred.grad

        mge_rsts = func(pred, label, dout)
        xla_rsts = func(pred, label, dout)

        for idx, (mge_rst, xla_rst) in enumerate(zip(mge_rsts, xla_rsts)):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=atol)

    from megengine.functional import loss

    tester(loss.l1_loss, (32, 16, 8, 8), (32, 16, 8, 8))
    tester(loss.l1_loss, (1, 16), (1, 16))
    tester(loss.square_loss, (32, 16, 8, 8), (32, 16, 8, 8))
    tester(loss.square_loss, (16, 1), (16, 1))
    tester(
        loss.cross_entropy,
        (16, 32),
        (16,),
        label_type="classes",
        axis=1,
        with_logits=True,
        label_smooth=0.0,
    )
    tester(
        loss.cross_entropy,
        (16, 32),
        (32,),
        label_type="classes",
        axis=0,
        with_logits=False,
        label_smooth=0.5,
    )
    tester(loss.binary_cross_entropy, (16, 32, 4, 8), (16, 32, 4, 8), with_logits=True)
    tester(loss.binary_cross_entropy, (1, 32, 1), (1, 32, 1), with_logits=False)
    tester(loss.hinge_loss, (32, 16, 8, 8), (32, 16, 8, 8), norm="L1")
    tester(loss.hinge_loss, (1, 16, 1, 1), (1, 16, 1, 1), norm="L2")
