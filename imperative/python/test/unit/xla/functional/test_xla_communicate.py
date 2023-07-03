import platform

import numpy as np
import pytest

import megengine.distributed as dist
import megengine.functional.distributed as fdist
import megengine.jit as jit
import megengine.tensor as tensor
from megengine import is_cuda_available
from megengine.distributed.helper import (
    get_offsets,
    param_pack_concat,
    param_pack_split,
)


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_param_pack_concat():
    def tester(ishapes, dtype=None):
        dtype = dtype or np.float32
        inps = [tensor(np.random.randn(*ishape), dtype=dtype) for ishape in ishapes]
        offset_vals = get_offsets(ishapes)
        offsets = tensor(offset_vals, dtype="int32")

        @jit.xla_trace(without_host=True)
        def func(*inps, offsets):
            return param_pack_concat(inps, offsets, offset_vals)

        mge_rst = func(*inps, offsets=offsets)
        xla_rst = func(*inps, offsets=offsets)
        np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

    tester(ishapes=((1,),))
    tester(ishapes=((1, 2),))
    tester(ishapes=((1,), (2,)))
    tester(ishapes=((1,), (2, 3), (4, 5, 6), (1,), (3, 2)))


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_param_pack_split():
    def tester(ishapes, dtype=None):
        dtype = dtype or np.float32
        offset_vals = get_offsets(ishapes)
        inp = tensor(np.random.randn(offset_vals[-1]), dtype=dtype)

        @jit.xla_trace(without_host=True)
        def func(inp):
            return param_pack_split(inp, offset_vals, ishapes)

        mge_rsts = func(inp)
        xla_rsts = func(inp)
        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

    tester(ishapes=((1,),))
    tester(ishapes=((1, 2),))
    tester(ishapes=((1,), (2,)))
    tester(ishapes=((1,), (2, 3), (4, 5, 6), (1,), (3, 2)))


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.require_ngpu(2)
@pytest.mark.isolated_distributed
def test_all_reduce():
    def tester(reduce_func, ishape, n_gpus, dtype=None):
        @dist.launcher(n_gpus=n_gpus)
        def worker(data):
            rank = dist.get_rank()
            inp = tensor(data[rank])

            @jit.xla_trace(without_host=True)
            def func(inp):
                return reduce_func(inp)

            mge_rst = func(inp)
            xla_rst = func(inp)

            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

        x = np.random.randn(*ishape).astype(dtype)
        y = np.random.randn(*ishape).astype(dtype)
        data = (x, y)
        worker(data)

    for func in [fdist.all_reduce_sum, fdist.all_reduce_min, fdist.all_reduce_max]:
        tester(func, (1,), 2)
        tester(func, (1, 1, 1), 2)
        tester(func, (16, 1, 64,), 2)
        tester(func, (16, 32, 64,), 2)


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.require_ngpu(2)
@pytest.mark.isolated_distributed
def test_all_reduce_multitime():
    def tester(ishape, n_gpus, dtype=None):
        @dist.launcher(n_gpus=n_gpus)
        def worker(data):
            rank = dist.get_rank()
            inp = tensor(data[rank])

            @jit.xla_trace(without_host=True)
            def func1(inp):
                return fdist.all_reduce_sum(inp)

            mge_rst = func1(inp)
            xla_rst = func1(inp)

            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

            @jit.xla_trace(without_host=True)
            def func2(inp):
                return fdist.all_reduce_sum(inp)

            mge_rst = func2(inp)
            xla_rst = func2(inp)

            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

        x = np.random.randn(*ishape).astype(dtype)
        y = np.random.randn(*ishape).astype(dtype)
        data = (x, y)
        worker(data)

    tester((16, 1, 64,), 2)
