# -*- coding: utf-8 -*-
import platform

import numpy as np
import pytest

import megengine as mge
import megengine.distributed as dist
from megengine import Parameter, tensor
from megengine.core._imperative_rt.core2 import sync
from megengine.device import get_default_device, set_default_device
from megengine.functional.distributed import (
    all_gather,
    all_reduce_max,
    all_reduce_min,
    all_reduce_sum,
    all_to_all,
    broadcast,
    gather,
    reduce_scatter_sum,
    reduce_sum,
    remote_recv,
    remote_send,
    scatter,
)


def run_reduce_sum(shape, dtype):
    @dist.launcher(n_gpus=2)
    def worker(data, expect):
        rank = dist.get_rank()
        inp = tensor(data[rank])
        output = reduce_sum(inp)
        if rank == 0:
            assert np.allclose(output.numpy(), expect[rank])
        else:
            assert output is None

    x = np.random.random_sample(shape).astype(dtype)
    y = np.random.random_sample(shape).astype(dtype)
    z = x + y
    data = (x, y)
    expect = (z, None)
    worker(data, expect)


@pytest.mark.require_ngpu(2)
@pytest.mark.parametrize("shape", [(), (1,), (2, 3), (8, 10), (99, 77)], ids=str)
@pytest.mark.isolated_distributed
def test_reduce_sum_multishape(shape):
    run_reduce_sum(shape, "float32")


@pytest.mark.require_ngpu(2)
@pytest.mark.parametrize("dtype", ["float32", "int32", "int8", "uint8"], ids=str)
@pytest.mark.isolated_distributed
def test_reduce_sum_multidtype(dtype):
    run_reduce_sum((8, 10), dtype)


def run_broadcast(shape, dtype):
    @dist.launcher(n_gpus=2)
    def worker(data, expect):
        rank = dist.get_rank()
        inp = tensor(data[rank])
        output = broadcast(inp)
        assert np.allclose(output.numpy(), expect[rank])

    x = np.random.random_sample(shape).astype(dtype)
    y = x + 1
    data = (x, y)
    expect = (x, x)
    worker(data, expect)


@pytest.mark.require_ngpu(2)
@pytest.mark.parametrize("shape", [(), (1,), (2, 3), (8, 10), (99, 77)], ids=str)
@pytest.mark.isolated_distributed
def test_broadcast_multishape(shape):
    run_broadcast(shape, "float32")


@pytest.mark.require_ngpu(2)
@pytest.mark.parametrize("dtype", ["float32", "int32", "int8", "uint8"], ids=str)
@pytest.mark.isolated_distributed
def test_broadcast_multidtype(dtype):
    run_broadcast((8, 10), dtype)


def run_all_gather(shape, dtype):
    @dist.launcher(n_gpus=2)
    def worker(data, expect):
        rank = dist.get_rank()
        inp = tensor(data[rank])
        output = all_gather(inp)
        assert np.allclose(output.numpy(), expect[rank])

    x = np.random.random_sample(shape).astype(dtype)
    y = np.random.random_sample(shape).astype(dtype)
    z = np.concatenate((x, y))
    data = (x, y)
    expect = (z, z)
    worker(data, expect)


@pytest.mark.require_ngpu(2)
@pytest.mark.parametrize("shape", [(1,), (2, 3), (8, 10), (99, 77)], ids=str)
@pytest.mark.isolated_distributed
def test_all_gather_multishape(shape):
    run_all_gather(shape, "float32")


@pytest.mark.require_ngpu(2)
@pytest.mark.parametrize("dtype", ["float32", "int32", "int8", "uint8"], ids=str)
@pytest.mark.isolated_distributed
def test_all_gather_multidtype(dtype):
    run_all_gather((8, 10), dtype)


def run_reduce_scatter_sum(shape, dtype):
    @dist.launcher(n_gpus=2)
    def worker(data, expect):
        rank = dist.get_rank()
        inp = tensor(data[rank])
        output = reduce_scatter_sum(inp)
        assert np.allclose(output.numpy(), expect[rank])

    x = np.random.random_sample(shape).astype(dtype)
    y = np.random.random_sample(shape).astype(dtype)
    z = x + y
    data = (x, y)
    expect = (z[: shape[0] // 2], z[shape[0] // 2 :])
    worker(data, expect)


@pytest.mark.require_ngpu(2)
@pytest.mark.parametrize("shape", [(2, 3), (8, 10), (88, 44)], ids=str)
@pytest.mark.isolated_distributed
def test_reduce_scatter_sum_multishape(shape):
    run_reduce_scatter_sum(shape, "float32")


@pytest.mark.require_ngpu(2)
@pytest.mark.parametrize("dtype", ["float32", "int32", "int8", "uint8"], ids=str)
@pytest.mark.isolated_distributed
def test_reduce_scatter_sum_multidtype(dtype):
    run_reduce_scatter_sum((8, 10), dtype)


def run_all_reduce_sum(shape, dtype):
    @dist.launcher(n_gpus=2)
    def worker(data, expect):
        rank = dist.get_rank()
        inp = tensor(data[rank])
        output = all_reduce_sum(inp)
        assert np.allclose(output.numpy(), expect[rank])

    x = np.random.random_sample(shape).astype(dtype)
    y = np.random.random_sample(shape).astype(dtype)
    z = x + y
    data = (x, y)
    expect = (z, z)
    worker(data, expect)


@pytest.mark.require_ngpu(2)
@pytest.mark.parametrize("shape", [(), (1,), (2, 3), (8, 10), (99, 77)], ids=str)
@pytest.mark.isolated_distributed
def test_all_reduce_sum_multishape(shape):
    run_all_reduce_sum(shape, "float32")


@pytest.mark.require_ngpu(2)
@pytest.mark.parametrize("dtype", ["float32", "int32", "int8", "uint8"], ids=str)
@pytest.mark.isolated_distributed
def test_all_reduce_sum_multidtype(dtype):
    run_all_reduce_sum((8, 10), dtype)


def run_all_reduce_max(shape, dtype):
    @dist.launcher(n_gpus=2)
    def worker(data, expect):
        rank = dist.get_rank()
        inp = tensor(data[rank])
        output = all_reduce_max(inp)
        assert np.allclose(output.numpy(), expect[rank])

    x = np.random.random_sample(shape).astype(dtype)
    y = np.random.random_sample(shape).astype(dtype)
    z = np.maximum(x, y)
    data = (x, y)
    expect = (z, z)
    worker(data, expect)


@pytest.mark.require_ngpu(2)
@pytest.mark.parametrize("shape", [(), (1,), (2, 3), (8, 10), (99, 77)], ids=str)
@pytest.mark.isolated_distributed
def test_all_reduce_max_multishape(shape):
    run_all_reduce_max(shape, "float32")


@pytest.mark.require_ngpu(2)
@pytest.mark.parametrize("dtype", ["float32", "int32", "int8", "uint8"], ids=str)
@pytest.mark.isolated_distributed
def test_all_reduce_max_multidtype(dtype):
    run_all_reduce_max((8, 10), dtype)


def run_all_reduce_min(shape, dtype):
    @dist.launcher(n_gpus=2)
    def worker(data, expect):
        rank = dist.get_rank()
        inp = tensor(data[rank])
        output = all_reduce_min(inp)
        assert np.allclose(output.numpy(), expect[rank])

    x = np.random.random_sample(shape).astype(dtype)
    y = np.random.random_sample(shape).astype(dtype)
    z = np.minimum(x, y)
    data = (x, y)
    expect = (z, z)
    worker(data, expect)


@pytest.mark.require_ngpu(2)
@pytest.mark.parametrize("shape", [(), (1,), (2, 3), (8, 10), (99, 77)], ids=str)
@pytest.mark.isolated_distributed
def test_all_reduce_min_multishape(shape):
    run_all_reduce_min(shape, "float32")


@pytest.mark.require_ngpu(2)
@pytest.mark.parametrize("dtype", ["float32", "int32", "int8", "uint8"], ids=str)
@pytest.mark.isolated_distributed
def test_all_reduce_min_multidtype(dtype):
    run_all_reduce_min((8, 10), dtype)


def run_gather(shape, dtype):
    @dist.launcher(n_gpus=2)
    def worker(data, expect):
        rank = dist.get_rank()
        inp = tensor(data[rank])
        output = gather(inp)
        if rank == 0:
            assert np.allclose(output.numpy(), expect[rank])
        else:
            assert output is None

    x = np.random.random_sample(shape).astype(dtype)
    y = np.random.random_sample(shape).astype(dtype)
    z = np.concatenate((x, y))
    data = (x, y)
    expect = (z, None)
    worker(data, expect)


@pytest.mark.require_ngpu(2)
@pytest.mark.parametrize("shape", [(2, 3), (8, 10), (99, 77)], ids=str)
@pytest.mark.isolated_distributed
def test_gather_multishape(shape):
    run_gather(shape, "float32")


@pytest.mark.require_ngpu(2)
@pytest.mark.parametrize("dtype", ["float32", "int32", "int8", "uint8"], ids=str)
@pytest.mark.isolated_distributed
def test_gather_multidtype(dtype):
    run_gather((8, 10), dtype)


def run_scatter(shape, dtype):
    @dist.launcher(n_gpus=2)
    def worker(data, expect):
        rank = dist.get_rank()
        inp = tensor(data[rank])
        output = scatter(inp)
        assert np.allclose(output.numpy(), expect[rank])

    x = np.random.random_sample(shape).astype(dtype)
    y = x + 1
    data = (x, y)
    expect = (x[: shape[0] // 2], x[shape[0] // 2 :])
    worker(data, expect)


@pytest.mark.require_ngpu(2)
@pytest.mark.parametrize("shape", [(2, 3), (8, 10), (100, 77)], ids=str)
@pytest.mark.isolated_distributed
def test_scatter_multishape(shape):
    run_scatter(shape, "float32")


@pytest.mark.require_ngpu(2)
@pytest.mark.parametrize("dtype", ["float32", "int32", "int8", "uint8"], ids=str)
@pytest.mark.isolated_distributed
def test_scatter_multidtype(dtype):
    run_scatter((8, 10), dtype)


def run_all_to_all(shape, dtype):
    @dist.launcher(n_gpus=2)
    def worker(data, expect):
        rank = dist.get_rank()
        inp = tensor(data[rank])
        output = all_to_all(inp)
        assert np.allclose(output.numpy(), expect[rank])

    x = np.random.random_sample(shape).astype(dtype)
    y = np.random.random_sample(shape).astype(dtype)
    a = np.concatenate((x[: shape[0] // 2], y[: shape[0] // 2]))
    b = np.concatenate((x[shape[0] // 2 :], y[shape[0] // 2 :]))
    data = (x, y)
    expect = (a, b)
    worker(data, expect)


@pytest.mark.require_ngpu(2)
@pytest.mark.parametrize("shape", [(2, 3), (8, 10), (100, 77)], ids=str)
@pytest.mark.isolated_distributed
def test_all_to_all_multishape(shape):
    run_all_to_all(shape, "float32")


@pytest.mark.require_ngpu(2)
@pytest.mark.parametrize("dtype", ["float32", "int32", "int8", "uint8"], ids=str)
@pytest.mark.isolated_distributed
def test_all_to_all_multidtype(dtype):
    run_all_to_all((8, 10), dtype)


def run_io_remote(shape, dtype):
    @dist.launcher(n_gpus=2)
    def worker(val, shape):
        rank = dist.get_rank()
        if rank == 0:  # remote send
            x = tensor(val, device="xpu0")
            remote_send(x, 1)
            sync()
        else:  # remote recv
            y = remote_recv(0)
            assert y.device == get_default_device()
            np.testing.assert_almost_equal(val, y.numpy())

    val = np.random.random_sample(shape).astype(dtype)
    worker(val, shape)


@pytest.mark.require_ngpu(2)
@pytest.mark.isolated_distributed
@pytest.mark.parametrize("shape", [(), (1,), (4, 5)], ids=str)
def test_io_remote_multishape(shape):
    run_io_remote(shape, "float32")


@pytest.mark.require_ngpu(2)
@pytest.mark.isolated_distributed
@pytest.mark.parametrize("dtype", ["float32", "int32", "int8", "uint8"], ids=str)
def test_io_remote_multidtype(dtype):
    run_io_remote((8, 10), dtype)


@pytest.mark.require_ngpu(2)
def test_cuda_init_before_fork():
    a = mge.tensor(1, device="gpu0")

    @dist.launcher(n_gpus=2)
    def worker():
        a += 1
        b = mge.tensor(2)

    with pytest.raises(AssertionError):
        worker()
