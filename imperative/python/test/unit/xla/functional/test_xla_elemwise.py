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
def test_elemwise():
    np.random.seed(123)
    mge.random.seed(123)

    def tester(felemwise, *inp_shapes, backward=True, dtype=None, atol=1e-5, **kwargs):
        dtype = dtype or np.float32
        if dtype in [np.int16, np.int32, np.uint16, np.uint32]:
            inps = [
                tensor(np.random.randint(0, 10, size=inp_shape), dtype=dtype)
                for inp_shape in inp_shapes
            ]
        else:
            inps = [
                tensor(0.1 * np.random.randn(*inp_shape), dtype=dtype)
                for inp_shape in inp_shapes
            ]
        doup = tensor(
            0.1 * np.random.randn(*felemwise(*inps, **kwargs).shape), dtype=dtype
        )

        gm = GradManager()

        @jit.xla_trace(without_host=True)
        def func(inps, doup):
            if backward:
                gm.attach(inps)
                with gm:
                    oup = felemwise(*inps, **kwargs)
                    gm.backward(oup, doup)
                    return [oup, *[inp.grad for inp in inps]]
            else:
                oup = felemwise(*inps, **kwargs)
                return [oup]

        mge_rsts = func(inps, doup)
        xla_rsts = func(inps, doup)
        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=atol)

    tester(F.neg, (4, 16, 12, 12), dtype=np.float32, atol=1e-5)
    tester(F.abs, (2, 32, 16), dtype=np.float32, atol=1e-5)
    tester(F.sin, (1, 16, 3, 1), dtype=np.float32, atol=1e-5)
    tester(F.cos, (4, 16, 3), dtype=np.float32, atol=1e-5)
    tester(F.tan, (4, 16, 1), dtype=np.float32, atol=1e-5)
    tester(F.sinh, (4, 16, 1), dtype=np.float32, atol=1e-5)
    tester(F.cosh, (3, 16, 1), dtype=np.float32, atol=1e-5)
    tester(F.tanh, (4, 6, 3, 1), dtype=np.float32, atol=5e-4)
    tester(F.asin, (4, 1, 3, 1), dtype=np.float32, atol=1e-5)
    tester(F.acos, (4, 16, 3, 1), dtype=np.float32, atol=1e-5)
    tester(F.atan, (4, 16, 3, 1), dtype=np.float32, atol=1e-5)
    tester(F.asinh, (4, 1, 3, 1), dtype=np.float32, atol=1e-5)
    tester(F.acosh, (4, 1), dtype=np.float32, atol=1e-5)
    tester(F.atanh, (1,), dtype=np.float32, atol=1e-5)
    tester(F.exp, (2, 8), dtype=np.float32, atol=1e-5)
    tester(F.sqrt, (32,), dtype=np.float32, atol=1e-5)
    tester(F.square, (32,), dtype=np.float32, atol=1e-5)
    tester(F.log, (8, 8, 16), dtype=np.float32, atol=1e-5)
    tester(F.log1p, (8, 1, 16), dtype=np.float32, atol=1e-5)
    tester(F.expm1, (6, 8, 2), dtype=np.float32, atol=1e-5)
    tester(F.floor, (4, 16, 1, 1), backward=False, dtype=np.float32, atol=1e-5)
    tester(F.ceil, (4, 1, 1), backward=False, dtype=np.float32, atol=1e-5)
    tester(F.round, (1, 4, 1), backward=False, dtype=np.float32, atol=1e-5)
    tester(F.clip, (4, 16, 1), dtype=np.float32, atol=1e-5, lower=-1.0, upper=1.0)
    tester(F.relu, (1,), dtype=np.float32, atol=1e-5)
    tester(F.gelu, (4, 16, 12, 12), dtype=np.float32, atol=2e-5)
    tester(F.sigmoid, (4, 16, 16, 12), dtype=np.float32, atol=1e-5)
    tester(F.hsigmoid, (4, 16, 16, 12), dtype=np.float32, atol=1e-5)
    tester(F.hswish, (4, 16, 16, 12), dtype=np.float32, atol=1e-5)
    tester(F.relu6, (12, 16, 1), dtype=np.float32, atol=1e-5)
    tester(F.leaky_relu, (1, 16, 1), dtype=np.float32, atol=1e-5)
    tester(F.leaky_relu, (12, 16, 1), dtype=np.float32, atol=1e-5, negative_slope=0.5)
    tester(F.silu, (4, 16, 12, 12), dtype=np.float32, atol=1e-5)
    tester(F.logsigmoid, (4, 16, 12, 12), dtype=np.float32, atol=1e-5)
    tester(F.softplus, (4, 16, 12, 12), dtype=np.float32, atol=1e-5)
    tester(F.add, (4, 16, 12, 12), (4, 16, 12, 12), dtype=np.float32, atol=1e-5)
    tester(F.sub, (4, 16, 12, 12), (4, 16, 1, 1), dtype=np.float32, atol=1e-5)
    tester(F.mul, (4, 16, 12, 12), (1, 1, 12, 12), dtype=np.float32, atol=1e-5)
    tester(F.div, (4, 16, 1, 1), (4, 16, 12, 12), atol=5e-4)
    tester(F.floor_div, (4, 16, 12, 12), (4, 16, 1, 1), backward=False, atol=5e-5)
    # tester(F.mod, (8, 1, 4), (8, 1, 1), backward=False, dtype=np.int32, atol=1e-5) # xla not support
    tester(F.pow, (4, 1, 12, 12), (1, 16, 12, 12), dtype=np.float32, atol=5e-5)
    tester(F.prelu, (4, 16, 12, 12), (1,), dtype=np.float32, atol=1e-5)
    tester(F.prelu, (16, 5, 12), (1, 5, 1), dtype=np.float32, atol=1e-5)
    tester(F.logaddexp, (16, 5, 12), (1, 5, 12), dtype=np.float32, atol=1e-5)
    tester(F.maximum, (1, 5, 1), (1, 5, 12), dtype=np.float32, atol=1e-5)
    tester(F.minimum, (1, 5, 12), (16, 5, 12), dtype=np.float32, atol=1e-5)

    tester(
        F.left_shift, (4, 16, 12, 12), (1, 1, 12, 12), backward=False, dtype=np.int32
    )
    tester(
        F.right_shift, (4, 16, 12, 12), (1, 1, 12, 12), backward=False, dtype=np.int32
    )

    tester(F.equal, (4, 16, 12, 12), (1, 1), backward=False)
    tester(F.not_equal, (4, 16, 12, 12), (4, 16, 1, 1), backward=False)
    tester(F.greater, (4, 16, 1, 1), (4, 16, 12, 12), backward=False)
    tester(F.greater_equal, (16, 1, 1), (4, 16, 12, 12), backward=False)
    tester(F.less, (4, 16, 12, 1), (4, 16, 12, 12), backward=False)
    tester(F.less_equal, (1, 1, 12, 12), (4, 16, 12, 12), backward=False)

    # bool is not support in dlpack now
    # tester(F.logical_and, (4, 16, 12, 12), (1, 1), backward=False, dtype=np.bool8)
    # tester(F.logical_or, (4, 16, 12, 12), (4, 16, 1, 1), backward=False, dtype=np.bool8)
    # tester(
    #     F.logical_xor, (4, 16, 1, 1), (4, 16, 12, 12), backward=False, dtype=np.bool8
    # )
    # tester(F.logical_not, (16, 1, 1), backward=False, dtype=np.bool8)


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_is_inf_nan():
    def tester(test_func, inf_or_nan, inp_shape, dtype=None):
        dtype = dtype or np.float32
        if dtype in [np.int16, np.int32, np.uint16, np.uint32]:
            inp = tensor(np.random.randint(0, 10, size=inp_shape), dtype=dtype)
        else:
            nr_elem = int(np.prod(inp_shape))
            inp = np.random.randn(nr_elem)
            idx = np.random.randint(0, nr_elem, size=(nr_elem,))
            inp[idx] = inf_or_nan
            inp = tensor(inp, dtype=dtype)

        @jit.xla_trace(without_host=True)
        def func(inp):
            oup = test_func(inp)
            return oup

        mge_rst = func(inp)
        xla_rst = func(inp)
        np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy())

    tester(F.isinf, np.inf, (16, 1), np.float32)
    tester(F.isinf, np.inf, (2, 32), np.int32)
    tester(F.isnan, np.nan, (1, 16), np.float32)
    tester(F.isnan, np.nan, (1,), np.float32)
    tester(F.isnan, np.nan, (32,), np.int32)
