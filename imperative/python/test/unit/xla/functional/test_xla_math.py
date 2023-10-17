import platform

import numpy as np
import pytest

import megengine.functional as F
import megengine.jit as jit
import megengine.tensor as tensor
from megengine import is_cuda_available
from megengine.autodiff.grad_manager import GradManager
from megengine.core.tensor.dtype import QuantDtypeMeta
from megengine.quantization.utils import (
    QuantMode,
    create_qparams,
    fake_quant_tensor,
    tqt_forward,
)


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_matmul():
    def tester(lhs_shape, rhs_shape, lhs_transpose, rhs_transpose, dtype=None):
        lhs = tensor(0.1 * np.random.randn(*lhs_shape), dtype=dtype)
        rhs = tensor(0.1 * np.random.randn(*rhs_shape), dtype=dtype)
        out = F.matmul(lhs, rhs, lhs_transpose, rhs_transpose)
        dout = tensor(0.1 * np.random.randn(*out.shape), dtype=dtype)

        gm = GradManager()

        @jit.xla_trace(without_host=True)
        def func(lhs, rhs, dout):
            gm.attach([lhs, rhs])
            with gm:
                out = F.matmul(lhs, rhs, lhs_transpose, rhs_transpose)
                gm.backward(out, dout)
            return out, lhs.grad, rhs.grad

        mge_rsts = func(lhs, rhs, dout)
        xla_rsts = func(lhs, rhs, dout)

        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

    tester((5,), (5,), False, False)
    tester((4, 5), (5,), False, False)
    tester((5,), (5, 6), False, False)
    tester((5, 4), (5,), True, False)

    tester((4, 5), (5, 6), False, False)
    tester((4, 5), (6, 5), False, True)
    tester((5, 4), (5, 6), True, False)
    tester((5, 4), (6, 5), True, True)

    tester((2, 3, 4, 5), (5, 6), False, False)
    tester((2, 3, 4, 5), (6, 5), False, True)
    tester((2, 1, 5, 4), (5, 6), True, False)
    tester((2, 1, 5, 4), (6, 5), True, True)
    tester((1, 5, 4), (5, 6), True, False)
    tester((1, 5, 4), (6, 5), True, True)

    tester((4, 5), (2, 3, 5, 6), False, False)
    tester((4, 5), (2, 3, 6, 5), False, True)
    tester((5, 4), (2, 1, 5, 6), True, False)
    tester((5, 4), (2, 1, 6, 5), True, True)
    tester((5, 4), (1, 5, 6), True, False)
    tester((5, 4), (1, 6, 5), True, True)

    tester((1, 4, 5), (1, 5, 6), False, False)
    tester((1, 5, 4), (1, 5, 6), True, False)
    tester((3, 4, 5), (3, 5, 6), False, False)
    tester((3, 5, 4), (3, 6, 5), True, True)

    tester((5, 3, 2, 7, 8), (3, 2, 8, 9), False, False)
    tester((5, 1, 2, 7, 8), (1, 2, 9, 8), False, True)
    tester((5, 3, 2, 8, 7), (3, 1, 8, 9), True, False)
    tester((5, 3, 2, 8, 7), (1, 2, 9, 8), True, True)
    tester((5, 3, 2, 8, 7), (1, 8, 9), True, False)
    tester((5, 3, 1, 8, 7), (1, 9, 8), True, True)

    tester((3, 2, 7, 8), (4, 3, 2, 8, 9), False, False)
    tester((3, 1, 7, 8), (4, 3, 1, 9, 8), False, True)
    tester((3, 1, 8, 7), (4, 3, 2, 8, 9), True, False)
    tester((1, 2, 8, 7), (4, 2, 2, 9, 8), True, True)
    tester((1, 8, 7), (4, 3, 2, 8, 9), True, False)
    tester((1, 8, 7), (4, 3, 1, 9, 8), True, True)


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_sort_and_argsort():
    def tester(ishape, descending, dtype=None):
        dtype = dtype or np.float32
        inp1 = tensor(np.random.randn(*ishape), dtype=dtype)
        inp2 = tensor(np.random.randn(*ishape), dtype=dtype)
        dout = tensor(np.random.randn(*ishape), dtype=dtype)

        gm = GradManager()

        @jit.xla_trace(without_host=True)
        def func(inp1, inp2, dout):
            gm.attach([inp1, inp2])
            with gm:
                out, idx1 = F.sort(inp1, descending)
                idx2 = F.argsort(inp2, -descending)
                gm.backward(out, dout)
            return out, idx1, idx2, inp1.grad

        mge_rsts = func(inp1, inp2, dout)
        xla_rsts = func(inp1, inp2, dout)

        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

    for descending in [True, False]:
        tester((16, 32), descending)
        tester((16, 1), descending)
        tester((1, 16), descending)
        tester((1, 1), descending)
        tester((16,), descending)
        tester((1,), descending)


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_topk():
    def tester(ishape, k, descending, kth_only, no_sort, dtype=None):
        dtype = dtype or np.float32
        inp = tensor(np.random.randn(*ishape), dtype=dtype)
        out, _ = F.topk(inp, k, descending, kth_only, no_sort)
        dout = tensor(0.1 * np.random.randn(*out.shape), dtype=dtype)

        gm = GradManager()

        @jit.xla_trace(without_host=True)
        def func(inp, dout):
            gm.attach([inp])
            with gm:
                out, index = F.topk(inp, k, descending, kth_only, no_sort)
                gm.backward(out, dout)
            return out, index, inp.grad

        mge_rsts = func(inp, dout)
        xla_rsts = func(inp, dout)

        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

    for descending in [True, False]:
        tester((2, 16,), 1, descending, False, False)
        tester((2, 16,), 8, descending, False, False)
        tester((1, 16,), 1, descending, False, False)
        tester((1, 16,), 5, descending, False, False)
        tester((16,), 8, descending, False, False)
        tester((16,), 8, descending, False, False)
        tester((1,), 1, descending, False, False)
        tester((1,), 1, descending, False, False)


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_topk_accuracy():
    def tester(batch, nr_class, topk, dtype=None):
        dtype = dtype or np.float32
        logits = tensor(np.random.uniform(0, 1, (batch, nr_class)), dtype=dtype)
        target = tensor(np.random.randint(0, nr_class, (batch,), np.int32))
        out = F.topk_accuracy(logits, target, topk)
        dout = tensor(0.1 * np.random.randn(*out.shape), dtype=dtype)

        gm = GradManager()

        @jit.xla_trace(without_host=True)
        def func(logits, target, dout):
            gm.attach([logits])
            with gm:
                out = F.topk_accuracy(logits, target, topk)
                gm.backward(out, dout)
            return [out]

        mge_rsts = func(logits, target, dout)
        xla_rsts = func(logits, target, dout)

        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

    tester(32, 1000, 10)
    tester(32, 1, 1)
    tester(1, 1000, 10)
    tester(1, 1, 1)


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_fakequant():
    def tester(inp_shape, qmin, qmax, scale, zero_point):
        test_dtype = QuantDtypeMeta("test_qint8", None, "int8", qmin, qmax)
        scale = tensor([scale], dtype=np.float32)
        zero_point = tensor([zero_point], dtype=np.float32)
        qparams = create_qparams(QuantMode.ASYMMERTIC, test_dtype, scale, zero_point)
        inp_data = np.random.uniform(low=-512.0, high=512.0, size=inp_shape)
        inp = tensor(inp_data, dtype=np.float32)
        oup = fake_quant_tensor(inp, qparams)
        dout = tensor(0.1 * np.random.randn(*oup.shape), dtype=np.float32)

        gm = GradManager()

        @jit.xla_trace(without_host=True)
        def func(inp, qparams, dout):
            gm.attach([inp])
            with gm:
                out = fake_quant_tensor(inp, qparams)
                gm.backward(out, dout)
            return [out, inp.grad]

        mge_rsts = func(inp, qparams, dout)
        xla_rsts = func(inp, qparams, dout)

        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

    tester((1, 32, 32, 32), -126, 129, 4, 1)
    tester((32, 32, 32, 32), -126, 129, 4, 1)
    tester((4, 32, 32, 32), -128, 126, 4, -1)
    tester((8, 32, 32, 32), -128, 126, -2, 1)


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_tqt():
    def tester(inp_shape, qmin, qmax, scale):

        scale = tensor([scale], dtype=np.float32)
        inp_data = np.random.uniform(low=-512.0, high=512.0, size=inp_shape)
        inp = tensor(inp_data, dtype=np.float32)

        oup = tqt_forward(qmin, qmax, inp, scale)
        dout = tensor(0.1 * np.random.randn(*oup.shape), dtype=np.float32)

        gm = GradManager()

        @jit.xla_trace(without_host=True)
        def func(inp, scale, qmin, qmax, dout):
            gm.attach([inp, scale])
            with gm:
                out = tqt_forward(qmin, qmax, inp, scale)
                gm.backward(out, dout)
            return [out, inp.grad, scale.grad]

        mge_rsts = func(inp, scale, qmin, qmax, dout)
        xla_rsts = func(inp, scale, qmin, qmax, dout)

        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(
                mge_rst.numpy(), xla_rst.numpy(), rtol=1e-5, atol=1e-5
            )

    tester((1, 32, 32, 32), -126, 129, 4)
    tester((16, 32, 32, 32), -126, 129, 2)
    tester((4, 32, 32, 32), -128, 128, 3)
