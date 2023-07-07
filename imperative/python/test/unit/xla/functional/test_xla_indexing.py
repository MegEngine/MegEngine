import platform

import numpy as np
import pytest

import megengine.functional as F
import megengine.jit as jit
import megengine.tensor as tensor
from megengine import is_cuda_available
from megengine.autodiff.grad_manager import GradManager


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_subtensor():
    def tester(ishape, index, dtype=None):
        dtype = dtype or np.float32
        inp = tensor(np.random.randn(*ishape), dtype=dtype)
        oshape = inp[index].shape
        dout = tensor(np.random.randn(*oshape), dtype=dtype)

        gm = GradManager()

        @jit.xla_trace(without_host=True, capture_as_const=True)
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


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_setsubtensor():
    def tester(x_shape, y_shape, indices, dtype=None):
        dtype = dtype or np.float32
        x = tensor(np.random.randn(*x_shape), dtype=dtype)
        y = tensor(np.random.randn(*y_shape), dtype=dtype)

        @jit.xla_trace(without_host=True)
        def func(x, y):
            x.__setitem__(indices, y)
            return x

        mge_rst = func(x, y)
        xla_rst = func(x, y)
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


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
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
        @jit.xla_trace(without_host=True, capture_as_const=True)
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


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_index_multi_vec():
    def tester(x_shape, index_type, dtype):
        dtype = dtype or np.float32
        x = tensor(np.random.randn(*x_shape), dtype=dtype)
        max_val = x.shape[0]
        ind = tensor(np.random.randint(-max_val + 1, max_val, 24).astype("int32"))
        gm = GradManager()
        rand_num = tensor(np.random.random(x[ind].shape).astype(dtype))

        @jit.xla_trace(without_host=True, capture_as_const=True)
        def func(inp, ind):
            gm.attach([inp])
            with gm:
                x = inp
                if index_type == "set":
                    x[ind] = tensor(rand_num)
                else:
                    x = x[ind]
                gm.backward((x * x).sum())
            return x, inp.grad

        mge_rsts = func(x, ind)
        xla_rsts = func(x, ind)
        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

    tester((3, 4, 5, 6), "get", np.float32)
    tester((3, 4, 5, 6), "get", np.float16)

    # tester((2,2,2,2), "set", np.float32)
    # tester((3,4,5,6), "set", np.float16)
    # tester((3,4,5,6), "set", np.float16)
