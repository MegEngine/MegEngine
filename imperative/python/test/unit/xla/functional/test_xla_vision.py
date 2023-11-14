import platform

import numpy as np
import pytest

import megengine as mge
import megengine.functional as F
import megengine.jit as jit
import megengine.tensor as tensor
from megengine import is_cuda_available
from megengine.autodiff.grad_manager import GradManager


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_resize():
    def tester(ishape, osize, fmt, mode, dtype=None):
        dtype = dtype or np.float32
        inp = tensor(np.random.randn(*ishape), dtype=dtype)
        oshape = F.vision.resize(inp, osize, fmt, mode).shape
        dout = tensor(np.random.randn(*oshape), dtype=dtype)
        mge._full_sync()
        gm = GradManager()

        @jit.xla_trace(without_host=True, capture_as_const=True)
        def func(inp, dout, mode):
            gm.attach([inp])
            with gm:
                out = F.vision.resize(inp, osize, fmt, mode)
                gm.backward(out, dout)
            return out, inp.grad

        mge_rsts = func(inp, dout, mode)
        xla_rsts = func(inp, dout, mode)

        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

    for mode in ["linear", "nearest"]:
        tester((4, 3, 1, 1), (1, 1), "NCHW", mode)
        tester((1, 1, 1, 1), (1, 1), "NCHW", mode)
        tester((1, 1, 1, 1), (2, 2), "NCHW", mode)
        tester((2, 3, 1, 1), (4, 3), "NCHW", mode)
        tester((4, 8, 7, 8), (12, 17), "NCHW", mode)
        tester((5, 6, 12, 17), (7, 8), "NCHW", mode)
        tester((1, 1, 3, 4), (6, 8), "NCHW", mode)
