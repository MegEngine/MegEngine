import platform

import numpy as np
import pytest

import megengine.functional as F
import megengine.jit as jit
import megengine.tensor as tensor
from megengine import autodiff, is_cuda_available
from megengine.autodiff.grad_manager import GradManager
from meg_xlalib.xla_extension import ArrayImpl


def test_external_flag_set():

    @xla_trace(capture_as_const=True)
    def test_fun():
        pass






def test_external_value():
    m = Conv2d(9,9, 3,groups=9)
    gm = GradManager()
    gm.attach(m.parameters())

    @xla_trace(capture_as_const=True)
    def conv_grad(inp, model):
        with gm:
            gm.attach(inp)
            rst = model(inp)
            gm.backward(rst.mean())
        ig = inp.grad
        wg = model.weight.grad
        inp.grad = None
        model.weight.grad = None
        return ig, wg

    inp = tensor(np.random.random((9,9, 32, 32)))*100

    a, b = conv_grad(inp, m)
    a1, b1 = conv_grad(inp, m)
    np.testing.assert_allclose(a.numpy(), a1.numpy())