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
def test_broadcast_to():
    def tester(ishape, tgtshape):
        dtype = None
        dtype = dtype or np.float32
        inp = tensor(np.random.randn(*ishape), dtype=dtype)
        dout = tensor(np.random.randn(*tgtshape), dtype=dtype)

        gm = GradManager()

        @jit.xla_trace(without_host=True)
        def func(inp, dout):
            gm.attach([inp])
            with gm:
                out = F.broadcast_to(inp, tgtshape)
                gm.backward(out, dout)
            return [out, inp.grad]

        mge_rsts = func(inp, dout)
        xla_rsts = func(inp, dout)
        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

    tester((1, 1, 1), (1, 1, 1, 1))
    tester((1, 1, 1, 1), (1, 1, 1, 1))
    tester((1, 1, 1, 1), (4, 5, 6, 7))
    tester((1, 1, 1), (4, 5, 6, 7))
    tester((5, 6, 7), (4, 5, 6, 7))
    tester((1, 6, 1), (4, 5, 6, 7))
    tester((1, 5, 6, 7), (4, 5, 6, 7))
    tester((1,), (4, 5, 1, 7))
    tester((4, 5, 3, 1), (4, 5, 3, 7))
    tester((4, 5, 3, 7), (4, 5, 3, 7))


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_reshape():
    def tester(ishape, tgt_shape, dtype=None):
        dtype = dtype or np.float32
        inp = tensor(np.random.randn(*ishape), dtype=dtype)
        oshape = F.reshape(inp, tgt_shape).shape
        dout = tensor(np.random.randn(*oshape), dtype=dtype)

        gm = GradManager()

        @jit.xla_trace(without_host=True)
        def func(inp, dout):
            gm.attach([inp])
            with gm:
                out = F.reshape(inp, tgt_shape)
                gm.backward(out, dout)
            return [out, inp.grad]

        mge_rsts = func(inp, dout)
        xla_rsts = func(inp, dout)
        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

    tester((1,), (1,))
    tester((1,), (1, 1, 1, 1))
    tester((2, 3, 4), (24,))
    tester((2, 3, 4), (2, 12))
    tester((2, 3, 4), (4, 3, 2))
    tester((2, 1, 4), (8, 1))
    tester((2, 1, 4), (-1))
    tester((2, 1, 4), (-1, 2))


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_transpose():
    def tester(ishape, permutation, dtype=None):
        dtype = dtype or np.float32
        inp = tensor(np.random.randn(*ishape), dtype=dtype)
        oshape = F.transpose(inp, permutation).shape
        dout = tensor(np.random.randn(*oshape), dtype=dtype)

        gm = GradManager()

        @jit.xla_trace(without_host=True)
        def func(inp, dout):
            gm.attach([inp])
            with gm:
                out = F.transpose(inp, permutation)
                gm.backward(out, dout)
            return [out, inp.grad]

        mge_rsts = func(inp, dout)
        xla_rsts = func(inp, dout)
        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

    tester((1,), (0,))
    tester((2, 3, 4), (0, 2, 1))
    tester((2, 3, 4), (2, 0, 1))
    tester((2, 3, 1), (0, 1, 2))
    tester((2, 3, 1, 4), (3, 1, 0, 2))

    tester((1,), ("x", 0))
    # tester((1,), (0, 'x')) # bug for mge
    tester((1, 2), ("x", 0, 1))
    tester((1, 2), (0, "x", 1))
    # tester((1, 2), (0, 1, 'x')) # bug for mge
    tester((16, 32, 64), (0, "x", 2, "x", 1))


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_expand_dims():
    def tester(ishape, axis, dtype=None):
        dtype = dtype or np.float32
        inp = tensor(np.random.randn(*ishape), dtype=dtype)
        oshape = F.expand_dims(inp, axis).shape
        dout = tensor(np.random.randn(*oshape), dtype=dtype)

        gm = GradManager()

        @jit.xla_trace(without_host=True)
        def func(inp, dout):
            gm.attach([inp])
            with gm:
                out = F.expand_dims(inp, axis)
                gm.backward(out, dout)
            return [out, inp.grad]

        mge_rsts = func(inp, dout)
        xla_rsts = func(inp, dout)
        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

    tester((2, 1, 4), 0)
    tester((2, 3, 4), 1)
    tester((2, 3, 4, 5), -1)


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_concat():
    def tester(*ishapes, axis, dtype=None):
        dtype = dtype or np.float32
        inps = [tensor(np.random.randn(*ishape), dtype=dtype) for ishape in ishapes]
        oshape = F.concat(inps, axis=axis).shape
        dout = tensor(np.random.randn(*oshape), dtype=dtype)

        gm = GradManager()

        @jit.xla_trace(without_host=True)
        def func(*inps, dout):
            gm.attach(inps)
            with gm:
                out = F.concat(inps, axis=axis)
                gm.backward(out, dout)
            rets = [inp.grad for inp in inps] + [out]
            return rets

        mge_rsts = func(*inps, dout=dout)
        xla_rsts = func(*inps, dout=dout)
        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

    tester((6, 5, 4), (6, 3, 4), (6, 1, 4), axis=1)
    tester((6, 5, 2), (6, 5, 1), axis=-1)
    tester((2, 5, 4), (6, 5, 4), axis=0)
    tester((1, 5, 4), (1, 5, 4), axis=0)
    tester((6, 5, 1), axis=-1)


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_split():
    def tester(ishape, axis, nsplit_or_sections, dtype=None):
        dtype = dtype or np.float32
        inp = tensor(np.random.randn(*ishape), dtype=dtype)
        oshapes = [o.shape for o in F.split(inp, nsplit_or_sections, axis)]
        douts = [tensor(np.random.randn(*oshape), dtype=dtype) for oshape in oshapes]

        gm = GradManager()

        @jit.xla_trace(without_host=True)
        def func(inp, douts):
            gm.attach([inp])
            with gm:
                outs = list(F.split(inp, nsplit_or_sections, axis))
                gm.backward(outs, douts)
            rets = outs + [inp.grad]
            return rets

        mge_rsts = func(inp, douts)
        xla_rsts = func(inp, douts)
        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

    tester((32, 16, 8), -2, 5)
    tester((32, 16, 8), 0, [8, 14, 27])
    tester((32, 16, 8), 1, 1)
    tester((32, 16, 8), 1, 16)


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_fill_and_fill_like():
    def tester(ref_shape, value, dtype=None):
        dtype = dtype or np.float32
        ref = tensor(np.random.randn(*ref_shape), dtype=dtype)

        @jit.xla_trace(without_host=True)
        def func(ref):
            return (
                F.full_like(ref, value),
                F.full(ref.shape, value, dtype=dtype),
                F.ones_like(ref),
                F.ones(ref.shape, dtype=dtype),
                F.zeros_like(ref),
                F.zeros(ref.shape, dtype=dtype),
            )

        mge_rst = func(ref)
        xla_rst = func(ref)
        for mge, xla in zip(mge_rst, xla_rst):
            np.testing.assert_allclose(mge.numpy(), xla.numpy(), atol=1e-5)

    tester((1,), 0.1)
    tester((16,), 0.1)
    tester((1, 16), 0.1)
    tester((32, 16), 0.1)
    tester((32, 16), 0)
    tester((1, 1, 16), 1)
