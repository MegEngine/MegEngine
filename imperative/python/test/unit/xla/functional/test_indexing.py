import numpy as np

import megengine.functional as F
import megengine.jit as jit
import megengine.tensor as tensor
from megengine.autodiff.grad_manager import GradManager


def test_subtensor():
    def tester(ishape, index, dtype=None):
        dtype = dtype or np.float32
        inp = tensor(np.random.randn(*ishape), dtype=dtype)
        oshape = inp[index].shape
        dout = tensor(np.random.randn(*oshape), dtype=dtype)

        gm = GradManager()

        @jit.trace(without_host=True, capture_as_const=True, use_xla=True)
        def func(inp, dout):
            gm.attach([inp])
            with gm:
                out = inp[index]
                gm.backward(out, dout)
            return out, inp.grad

        mge_rsts = func(inp, dout)
        xla_rsts = func(inp, dout)

        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

    tester(
        (16, 32, 64, 128), (10, slice(3, 13, 1), slice(-12, -3, 2), slice(None, 13, 3),)
    )
    tester(
        (16, 32, 64, 128),
        (slice(3, None, 1), slice(5, None, 3), slice(None, 13, 1), slice(None, 18, 4),),
    )
    tester(
        (16, 32, 64, 128),
        (slice(None, None, 1), None, slice(None, None, 5), slice(-12, -3, 1),),
    )
    tester(
        (16, 32, 1, 128),
        (slice(-12, -3, 2), slice(-13, None, 1), 0, slice(-12, None, 3),),
    )
    tester(
        (16, 32, 64, 128),
        (slice(None, -4, 1), 18, slice(None, -3, 4), slice(None, -3, 1),),
    )
    tester((16, 32, 64, 128), 10)
    tester((16, 32, 64, 128), None)
    tester((16, 32, 64, 128), (slice(3, None, 1), None, slice(-12, -3, 2),))


def test_setsubtensor():
    def tester(x_shape, y_shape, indices, dtype=None):
        dtype = dtype or np.float32
        x = tensor(np.random.randn(*x_shape), dtype=dtype)
        y = tensor(np.random.randn(*y_shape), dtype=dtype)

        @jit.trace(without_host=True, use_xla=True)
        def func(x, y):
            x.__setitem__(indices, y)
            return x

        mge_rsts = func(x, y)
        xla_rsts = func(x, y)

        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

    tester((32, 16, 8), (16, 8), (11,))
    tester((32, 16, 8), (16, 8), (11,))
    tester((32, 16, 8), (1, 8), (11,))
    tester((32, 16, 8), (8,), (11,))
    tester((32, 16, 8), (1,), (11,))
    tester((32, 16, 8), (14, 16, 8), (slice(2, 16, 1),))
    tester((32, 16, 8), (7, 16, 8), (slice(2, 16, 2),))
    tester((32, 16, 8), (16, 8), (slice(2, 16, 1),))
    tester((32, 16, 8), (16, 8), (slice(2, 16, 2),))
    tester((32, 16, 8), (1, 8), (slice(2, 16, 1),))
    tester((32, 16, 8), (1, 8), (slice(2, 16, 2),))
    tester((32, 16, 8), (8,), (slice(2, 16, 1),))
    tester((32, 16, 8), (8,), (slice(2, 16, 2),))
    tester((32, 16, 8), (1,), (slice(2, 16, 1),))
    tester((32, 16, 8), (1,), (slice(2, 16, 2),))
    tester((32, 16, 8), (8, 10, 8), (slice(4, 26, 3), slice(2, 12, 1),))
    tester((32, 16, 8), (1, 10, 8), (slice(4, 26, 3), slice(2, 12, 1),))
    tester((32, 16, 8), (10, 8), (slice(4, 26, 3), slice(2, 12, 1),))
    tester((32, 16, 8), (1, 8), (slice(4, 26, 3), slice(2, 12, 1),))
    tester((32, 16, 8), (8,), (slice(4, 26, 3), slice(2, 12, 1),))
    tester((32, 16, 8), (1,), (slice(4, 26, 3), slice(2, 12, 1),))
    tester((32, 16, 8), (10, 8), (10, slice(2, 12, 1),))
    tester((32, 16, 8), (1, 8), (10, slice(2, 12, 1),))
    tester((32, 16, 8), (8,), (10, slice(2, 12, 1),))
    tester((32, 16, 8), (1,), (10, slice(2, 12, 1),))
    tester((32, 16, 8), (1, 10, 16, 8), (None, slice(2, 12, 1),))
    tester((32, 16, 8), (1, 1, 16, 8), (None, slice(2, 12, 1),))
    tester((32, 16, 8), (10, 16, 8), (None, slice(2, 12, 1),))
    tester((32, 16, 8), (1, 16, 8), (None, slice(2, 12, 1),))
    tester((32, 16, 8), (8,), (None, slice(2, 12, 1),))


def test_indexing_one_hot():
    def tester(ishape, axis, keepdims, dtype=None):
        dtype = dtype or np.float32
        x = tensor(np.random.randn(*ishape), dtype=dtype)
        nr_class = ishape[axis]
        idx_shape = list(ishape)
        del idx_shape[axis]
        index = tensor(np.random.randint(0, nr_class, idx_shape), dtype="int32")
        oshape = F.nn.indexing_one_hot(x, index, axis, keepdims=keepdims).shape
        dy = tensor(np.random.randn(*oshape), dtype=dtype)

        gm = GradManager()

        # only capture_as_const is True, this function can be trace successfully
        @jit.trace(without_host=True, capture_as_const=True, use_xla=True)
        def func(x, index, dy):
            gm.attach([x])
            with gm:
                y = F.nn.indexing_one_hot(x, index, axis, keepdims=keepdims)
                gm.backward(y, dy)
            return y, x.grad

        mge_rsts = func(x, index, dy)
        xla_rsts = func(x, index, dy)

        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

    tester((4, 8, 16), 0, True)
    tester((4, 8, 16), 0, False)
    tester((4, 8, 16), 1, True)
    tester((4, 8, 16), 1, False)
    tester((4, 8, 16), -1, True)
    tester((4, 8, 16), -1, False)
    tester((4, 1, 16), -2, True)
    tester((4, 1, 16), -2, False)


if __name__ == "__main__":
    test_subtensor()
    test_setsubtensor()
    test_indexing_one_hot()
