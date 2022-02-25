# -*- coding: utf-8 -*-
import numpy as np
import pytest

import megengine as mge
import megengine.distributed as dist
from megengine import tensor
from megengine.distributed.functional import (
    all_gather,
    all_to_all,
    gather,
    reduce_scatter_sum,
    scatter,
)
from megengine.jit import trace


@pytest.mark.require_ngpu(2)
@pytest.mark.parametrize("shape", [(2, 3), (8, 10), (99, 77), (2, 2, 2, 2)], ids=str)
@pytest.mark.parametrize("symbolic", [False, True], ids=str)
@pytest.mark.parametrize("axis", [0, 1], ids=str)
@pytest.mark.isolated_distributed
def test_all_gather(shape, symbolic, axis):
    @dist.launcher(n_gpus=2)
    def worker(data, expect):
        rank = dist.get_rank()
        inp = tensor(data[rank])

        def func():
            output = all_gather(inp, axis=axis)
            return output

        func = trace(symbolic=symbolic)(func)
        output = func()
        assert np.allclose(output.numpy(), expect[rank])

    x = np.random.random_sample(shape).astype("float32")
    y = np.random.random_sample(shape).astype("float32")
    z = np.concatenate((x, y), axis=axis)
    data = (x, y)
    expect = (z, z)
    worker(data, expect)


@pytest.mark.require_ngpu(2)
@pytest.mark.parametrize(
    "shape,symbolic", [((2, 4, 6, 8), False), ((2, 4, 6, 8), True)], ids=str
)
@pytest.mark.parametrize("axis", [1, 0, 2, 3], ids=str)
@pytest.mark.isolated_distributed
def test_reduce_scatter_sum(shape, symbolic, axis):
    @dist.launcher(n_gpus=2)
    def worker(data, expect):
        rank = dist.get_rank()
        inp = tensor(data[rank])

        def func():
            output = reduce_scatter_sum(inp, axis=axis)
            return output

        func = trace(symbolic=symbolic)(func)
        output = func()
        assert np.allclose(output.numpy(), expect[rank])

    x = np.random.random_sample(shape).astype("float32")
    y = np.random.random_sample(shape).astype("float32")
    z = x + y
    data = (x, y)
    z = np.split(z, 2, axis=axis)
    z = np.concatenate(z, axis=0)
    expect = (z[: z.shape[0] // 2], z[z.shape[0] // 2 :])
    worker(data, expect)


@pytest.mark.require_ngpu(2)
@pytest.mark.parametrize(
    "shape,symbolic", [((2, 4, 6, 8), True), ((2, 4, 6, 8), False)], ids=str
)
@pytest.mark.parametrize("axis", [1, 0, 2, 3], ids=str)
@pytest.mark.isolated_distributed
def test_scatter(shape, symbolic, axis):
    @dist.launcher(n_gpus=2)
    def worker(data, expect):
        rank = dist.get_rank()
        inp = tensor(data[rank])

        def func():
            output = scatter(inp, axis=axis)
            return output

        func = trace(symbolic=symbolic)(func)
        output = func()
        assert np.allclose(output.numpy(), expect[rank])

    x = np.random.random_sample(shape).astype("float32")
    y = x + 1
    data = (x, y)
    _x = np.split(x, 2, axis=axis)
    _x = np.concatenate(_x, axis=0)
    expect = (_x[: _x.shape[0] // 2], _x[_x.shape[0] // 2 :])
    worker(data, expect)


@pytest.mark.require_ngpu(2)
@pytest.mark.parametrize("shape", [(2, 4, 6, 8)], ids=str)
@pytest.mark.parametrize("symbolic", [False, True], ids=str)
@pytest.mark.parametrize(
    "split_axis,concat_axis", [(0, 1), (1, 0), (2, 0), (0, 2), (2, 3)], ids=str
)
@pytest.mark.isolated_distributed
def test_all_to_all(shape, symbolic, split_axis, concat_axis):
    @dist.launcher(n_gpus=2)
    def worker(data):
        rank = dist.get_rank()
        inp = tensor(data[rank])

        def func():
            all_to_all_output = all_to_all(
                inp, split_axis=split_axis, concat_axis=concat_axis
            )
            gather_C = gather(inp, axis=concat_axis)
            gather_B = gather(all_to_all_output, axis=split_axis)
            if rank == 0:
                return gather_B, gather_C
            return all_to_all_output

        func = trace(symbolic=symbolic)(func)
        ret = func()
        if rank == 0:
            assert np.allclose(ret[0], ret[1])

    x = np.random.random_sample(shape).astype("float32")
    y = np.random.random_sample(shape).astype("float32")
    data = (x, y)
    worker(data)
