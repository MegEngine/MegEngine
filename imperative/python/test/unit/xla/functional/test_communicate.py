import numpy as np
import pytest

import megengine.distributed as dist
import megengine.functional.distributed as fdist
import megengine.jit as jit
import megengine.tensor as tensor
from megengine.distributed.helper import (
    get_offsets,
    param_pack_concat,
    param_pack_split,
)


def test_param_pack_concat():
    def tester(ishapes, dtype=None):
        dtype = dtype or np.float32
        inps = [tensor(np.random.randn(*ishape), dtype=dtype) for ishape in ishapes]
        offset_vals = get_offsets(ishapes)
        offsets = tensor(offset_vals, dtype="int32")

        @jit.trace(without_host=True, use_xla=True)
        def func(*inps, offsets):
            return param_pack_concat(inps, offsets, offset_vals)

        mge_rst = func(*inps, offsets=offsets)
        xla_rst = func(*inps, offsets=offsets)
        np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

    tester(ishapes=((1,),))
    tester(ishapes=((1, 2),))
    tester(ishapes=((1,), (2,)))
    tester(ishapes=((1,), (2, 3), (4, 5, 6), (1,), (3, 2)))


def test_param_pack_split():
    def tester(ishapes, dtype=None):
        dtype = dtype or np.float32
        offset_vals = get_offsets(ishapes)
        inp = tensor(np.random.randn(offset_vals[-1]), dtype=dtype)

        @jit.trace(without_host=True, use_xla=True)
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


# @pytest.mark.require_ngpu(2)
# @pytest.mark.isolated_distributed
def _test_all_reduce():
    def tester(reduce_func, ishape, n_gpus, dtype=None):
        @dist.launcher(n_gpus=n_gpus)
        def worker(data):
            rank = dist.get_rank()
            inp = tensor(data[rank])

            @jit.trace(without_host=True, use_xla=True)
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


if __name__ == "__main__":
    # test_param_pack_concat()
    # test_param_pack_split()
    _test_all_reduce()
