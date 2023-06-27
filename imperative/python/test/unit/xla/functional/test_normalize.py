import numpy as np

import megengine.functional as F
import megengine.jit as jit
import megengine.tensor as tensor
from megengine import autodiff
from megengine.autodiff.grad_manager import GradManager


def test_layer_norm():
    def tester(x_shape, normalized_shape, w_shape, b_shape, eps=1e-5, dtype=None):
        dtype = dtype or np.float32
        x = tensor(0.1 * np.random.rand(*x_shape), dtype=dtype)
        w = tensor(0.1 * np.random.rand(*w_shape), dtype=dtype) if w_shape else None
        b = tensor(0.1 * np.random.rand(*b_shape), dtype=dtype) if b_shape else None
        y = F.layer_norm(
            x,
            normalized_shape=normalized_shape,
            affine=b is not None,
            weight=w,
            bias=b,
            eps=eps,
        )
        dy = tensor(0.1 * np.random.rand(*y.shape), dtype=dtype)

        gm = GradManager()

        if b is not None:

            @jit.trace(without_host=True, use_xla=True)
            def func(x, w, b, dy):
                gm.attach([x, w, b])
                with gm:
                    y = F.layer_norm(
                        x,
                        normalized_shape=normalized_shape,
                        affine=True,
                        weight=w,
                        bias=b,
                        eps=eps,
                    )
                    gm.backward(y, dy)
                return [y, x.grad, w.grad, b.grad]

            mge_rsts = func(x, w, b, dy)
            xla_rsts = func(x, w, b, dy)
        else:

            @jit.trace(without_host=True, use_xla=True)
            def func(x, dy):
                gm.attach([x])
                with gm:
                    y = F.layer_norm(
                        x, normalized_shape=normalized_shape, affine=False, eps=eps
                    )
                    gm.backward(y, dy)
                return [y, x.grad]

            mge_rsts = func(x, dy)
            xla_rsts = func(x, dy)

        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

    tester((4, 16, 24, 24), (24, 24), (24, 24), (24, 24))
    tester((4, 16, 24, 24), (24, 24), None, None)
    tester((4, 16, 24, 28), (28,), (28,), (28,))
    tester((4, 16, 24, 28), (28,), None, None)


def test_batch_norm():
    def tester(ishape, training, momentum, eps, inplace, dtype=None):
        dtype = dtype or np.float32
        x = tensor(np.random.randn(*ishape), dtype=dtype)
        rmean = tensor(np.random.randn(1, ishape[1], 1, 1), dtype=dtype)
        rvar = tensor(np.abs(np.random.randn(1, ishape[1], 1, 1)), dtype=dtype)
        weight = tensor(np.random.randn(1, ishape[1], 1, 1), dtype=dtype)
        bias = tensor(np.random.randn(1, ishape[1], 1, 1), dtype=dtype)
        dy = tensor(np.random.randn(*ishape), dtype=dtype)

        gm = autodiff.GradManager()

        @jit.trace(without_host=True, use_xla=True)
        def func(x, rmean, rvar, weight, bias, dy):
            gm.attach([x, weight, bias])
            with gm:
                outs = F.batch_norm(
                    x,
                    rmean,
                    rvar,
                    weight,
                    bias,
                    training=training,
                    momentum=momentum,
                    eps=eps,
                    inplace=inplace,
                )
                if inplace:
                    y = outs
                else:
                    y, rmean, rvar = outs
                if training:
                    gm.backward(y, dy)
                    return y, rmean, rvar, x.grad, weight.grad, bias.grad
                else:
                    return [y]

        mge_rsts = func(x, rmean, rvar, weight, bias, dy)
        xla_rsts = func(x, rmean, rvar, weight, bias, dy)

        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=5e-4)

    tester((32, 16, 8, 8), True, 0.9, 1e-5, True)
    tester((1, 16, 17, 128), True, 0.7, 1e-5, False)
    tester((32, 16, 64, 5), False, 0.8, 1e-5, True)


if __name__ == "__main__":
    test_layer_norm()
    test_batch_norm()
