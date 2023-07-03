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
def test_dropout():
    def check_dropout(mge_val, xla_val, drop_prob):
        nr_zero = np.sum(np.array(xla_val == 0, np.uint32))
        nr_el = np.prod(xla_val.shape)
        xla_drop_rate = nr_zero * 1.0 / nr_el
        np.testing.assert_allclose(drop_prob, xla_drop_rate, atol=1e-3)

        mge_mask = mge_val == 0
        xla_mask = xla_val == 0
        both_mask = np.bitwise_or(xla_mask, mge_mask)
        both_left = np.bitwise_not(both_mask)
        mge_left = mge_val * both_left
        xla_left = xla_val * both_left
        np.testing.assert_allclose(mge_left, xla_left, atol=1e-6)

    def tester(shape, drop_prob, dtype=None):
        dtype = dtype or np.float32
        x = tensor(np.random.randn(*shape), dtype=dtype)
        dy = tensor(np.random.randn(*shape), dtype=dtype)

        gm = GradManager()

        @jit.xla_trace(without_host=True)
        def func(x, dy):
            gm.attach([x])
            with gm:
                y = F.dropout(x, drop_prob, True)
                gm.backward(y, dy)
            return y, x.grad

        mge_rsts = func(x, dy)
        xla_rsts = func(x, dy)

        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            check_dropout(mge_rst.numpy(), xla_rst.numpy(), drop_prob)

    tester((32, 128, 128, 1, 16), 0.1)
    tester((32, 128, 128, 1, 16), 0.3)
    tester((32, 128, 128, 1, 16), 0.5)
    tester((32, 128, 128, 1, 16), 0.9)
