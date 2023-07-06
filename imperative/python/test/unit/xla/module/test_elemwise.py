import platform

import numpy as np
import pytest

import megengine as mge
import megengine.functional as F
import megengine.module as M
import megengine.tensor as tensor
from megengine import is_cuda_available, jit
from megengine.autodiff import GradManager
from megengine.optimizer import Adam


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_elemwise_activation():
    def tester(TestMod, ishape, dtype=None, atol=1e-5, **kwargs):
        dtype = dtype or np.float32
        inp = tensor(0.1 * np.random.randn(*ishape), dtype=dtype)
        doup = tensor(0.1 * np.random.randn(*ishape), dtype=dtype)

        gm = GradManager()
        mod = TestMod(**kwargs)

        @jit.xla_trace(without_host=True)
        def func(mod, inp, doup):
            gm.attach(inp)
            with gm:
                oup = mod(inp)
                gm.backward(oup, doup)
            return oup, inp.grad

        mge_rsts = func(mod, inp, doup)
        xla_rsts = func(mod, inp, doup)
        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=atol)

    tester(M.Sigmoid, (2, 3, 4, 5))
    tester(M.ReLU, (2, 3,))
    tester(M.LeakyReLU, (4, 5))
    tester(M.LeakyReLU, (4, 5), negative_slope=0.3)
    tester(M.PReLU, (8, 6, 5))
    tester(M.PReLU, (8, 6, 5, 7), num_parameters=6, init=0.1)
    tester(M.PReLU, (1,))
    tester(M.SiLU, (4, 8, 3, 2))
    tester(M.SiLU, (1, 1,))
    tester(M.GELU, (1, 1, 2))
