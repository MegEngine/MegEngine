import platform

import numpy as np
import pytest

import megengine.distributed as dist
import megengine.functional as F
import megengine.functional.distributed as fdist
import megengine.tensor as tensor
from megengine import autodiff, is_cuda_available
from megengine.autodiff.grad_manager import GradManager
from megengine.core._imperative_rt.core2 import (
    is_external_convert,
    set_external_convert_hook,
)
from megengine.jit import xla_trace
from megengine.module import Conv2d


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_external_tsf_set():
    from mge_xlalib.xla_extension import ArrayImpl

    @xla_trace(capture_as_const=True)
    def test_func(inp):
        return inp

    assert is_external_convert()
    inp = tensor(np.random.random((9, 9, 32, 32)))

    mge_inp = test_func(inp)
    xla_inp = test_func(inp)
    assert xla_inp._is_external_value()
    assert isinstance(xla_inp._external_obj(), ArrayImpl)

    assert mge_inp.shape == xla_inp.shape
    assert mge_inp.dtype == xla_inp.dtype
    assert not xla_inp._is_external_value()


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_external_value():
    m = Conv2d(9, 9, 3, groups=9)
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

    inp = tensor(np.random.random((9, 9, 32, 32))) * 100

    mge_ig, mge_wg = conv_grad(inp, m)
    xla_ig, xla_wg = conv_grad(inp, m)
    np.testing.assert_allclose(mge_ig.numpy(), xla_ig.numpy())
    np.testing.assert_allclose(mge_wg.numpy(), xla_wg.numpy(), atol=1e-5)


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.require_ngpu(2)
@pytest.mark.isolated_distributed
def test_distributed_convert():
    from mge_xlalib.xla_extension import ArrayImpl

    def tester(ishape, n_gpus, dtype=None):
        @dist.launcher(n_gpus=n_gpus)
        def worker(data):
            rank = dist.get_rank()
            inp = tensor(data[rank])

            @xla_trace(without_host=True)
            def func1(inp):
                return fdist.all_reduce_sum(inp)

            mge_rst = func1(inp)
            xla_rst = func1(inp)
            assert xla_rst._is_external_value()
            assert isinstance(xla_rst._external_obj(), ArrayImpl)

            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)
            assert mge_rst.shape == xla_rst.shape
            assert mge_rst.dtype == xla_rst.dtype
            assert not xla_rst._is_external_value()

        x = np.random.randn(*ishape).astype(dtype)
        y = np.random.randn(*ishape).astype(dtype)
        data = (x, y)
        worker(data)

    tester((16, 1, 64,), 2)
