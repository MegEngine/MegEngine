# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import platform

import numpy as np
import pytest

import megengine as mge
import megengine.distributed as dist
from megengine import Parameter, Tensor, tensor
from megengine.core._imperative_rt.core2 import sync
from megengine.device import get_default_device, set_default_device
from megengine.distributed.helper import get_device_count_by_fork
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


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="windows disable MGB_ENABLE_OPR_MM"
)
@pytest.mark.skipif(get_device_count_by_fork("gpu") < 2, reason="need more gpu device")
@pytest.mark.isolated_distributed
def test_reduce_sum():
    @dist.launcher(n_gpus=2)
    def worker(data, expect):
        rank = dist.get_rank()
        inp = tensor(data[rank])
        output = reduce_sum(inp)
        if rank == 0:
            assert np.allclose(output.numpy(), expect[rank])
        else:
            assert np.allclose(output.numpy(), 0)

    def check(shape):
        x = np.random.rand(*shape).astype("float32")
        y = np.random.rand(*shape).astype("float32")
        z = x + y
        data = (x, y)
        expect = (z, None)
        worker(data, expect)

    for shape in [(2, 3), (8, 10), (99, 77)]:
        check(shape)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="windows disable MGB_ENABLE_OPR_MM"
)
@pytest.mark.skipif(get_device_count_by_fork("gpu") < 2, reason="need more gpu device")
@pytest.mark.isolated_distributed
def test_broadcast():
    @dist.launcher(n_gpus=2)
    def worker(data, expect):
        rank = dist.get_rank()
        inp = tensor(data[rank])
        output = broadcast(inp)
        assert np.allclose(output.numpy(), expect[rank])

    def check(shape):
        x = np.random.rand(*shape).astype("float32")
        y = x + 1
        data = (x, y)
        expect = (x, x)
        worker(data, expect)

    for shape in [(2, 3), (8, 10), (99, 77)]:
        check(shape)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="windows disable MGB_ENABLE_OPR_MM"
)
@pytest.mark.skipif(get_device_count_by_fork("gpu") < 2, reason="need more gpu device")
@pytest.mark.isolated_distributed
def test_all_gather():
    @dist.launcher(n_gpus=2)
    def worker(data, expect):
        rank = dist.get_rank()
        inp = tensor(data[rank])
        output = all_gather(inp)
        assert np.allclose(output.numpy(), expect[rank])

    def check(shape):
        x = np.random.rand(*shape).astype("float32")
        y = np.random.rand(*shape).astype("float32")
        z = np.concatenate((x, y))
        data = (x, y)
        expect = (z, z)
        worker(data, expect)

    for shape in [(2, 3), (8, 10), (99, 77)]:
        check(shape)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="windows disable MGB_ENABLE_OPR_MM"
)
@pytest.mark.skipif(get_device_count_by_fork("gpu") < 2, reason="need more gpu device")
@pytest.mark.isolated_distributed
def test_reduce_scatter_sum():
    @dist.launcher(n_gpus=2)
    def worker(data, expect):
        rank = dist.get_rank()
        inp = tensor(data[rank])
        output = reduce_scatter_sum(inp)
        assert np.allclose(output.numpy(), expect[rank])

    def check(shape):
        x = np.random.rand(*shape).astype("float32")
        y = np.random.rand(*shape).astype("float32")
        z = x + y
        data = (x, y)
        expect = (z[: shape[0] // 2], z[shape[0] // 2 :])
        worker(data, expect)

    for shape in [(2, 4), (8, 10), (88, 44)]:
        check(shape)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="windows disable MGB_ENABLE_OPR_MM"
)
@pytest.mark.skipif(get_device_count_by_fork("gpu") < 2, reason="need more gpu device")
@pytest.mark.isolated_distributed
def test_all_reduce_sum():
    @dist.launcher(n_gpus=2)
    def worker(data, expect):
        rank = dist.get_rank()
        inp = tensor(data[rank])
        output = all_reduce_sum(inp)
        assert np.allclose(output.numpy(), expect[rank])

    def check(shape):
        x = np.random.rand(*shape).astype("float32")
        y = np.random.rand(*shape).astype("float32")
        z = x + y
        data = (x, y)
        expect = (z, z)
        worker(data, expect)

    for shape in [(2, 3), (8, 10), (99, 77)]:
        check(shape)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="windows disable MGB_ENABLE_OPR_MM"
)
@pytest.mark.skipif(get_device_count_by_fork("gpu") < 2, reason="need more gpu device")
@pytest.mark.isolated_distributed
def test_all_reduce_max():
    @dist.launcher(n_gpus=2)
    def worker(data, expect):
        rank = dist.get_rank()
        inp = tensor(data[rank])
        output = all_reduce_max(inp)
        assert np.allclose(output.numpy(), expect[rank])

    def check(shape):
        x = np.random.rand(*shape).astype("float32")
        y = np.random.rand(*shape).astype("float32")
        z = np.maximum(x, y)
        data = (x, y)
        expect = (z, z)
        worker(data, expect)

    for shape in [(2, 3), (8, 10), (99, 77)]:
        check(shape)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="windows disable MGB_ENABLE_OPR_MM"
)
@pytest.mark.skipif(get_device_count_by_fork("gpu") < 2, reason="need more gpu device")
@pytest.mark.isolated_distributed
def test_all_reduce_min():
    @dist.launcher(n_gpus=2)
    def worker(data, expect):
        rank = dist.get_rank()
        inp = tensor(data[rank])
        output = all_reduce_min(inp)
        assert np.allclose(output.numpy(), expect[rank])

    def check(shape):
        x = np.random.rand(*shape).astype("float32")
        y = np.random.rand(*shape).astype("float32")
        z = np.minimum(x, y)
        data = (x, y)
        expect = (z, z)
        worker(data, expect)

    for shape in [(2, 3), (8, 10), (99, 77)]:
        check(shape)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="windows disable MGB_ENABLE_OPR_MM"
)
@pytest.mark.skipif(get_device_count_by_fork("gpu") < 2, reason="need more gpu device")
@pytest.mark.isolated_distributed
def test_gather():
    @dist.launcher(n_gpus=2)
    def worker(data, expect):
        rank = dist.get_rank()
        inp = tensor(data[rank])
        output = gather(inp)
        if rank == 0:
            assert np.allclose(output.numpy(), expect[rank])
        else:
            assert np.allclose(output.numpy(), 0)

    def check(shape):
        x = np.random.rand(*shape).astype("float32")
        y = np.random.rand(*shape).astype("float32")
        z = np.concatenate((x, y))
        data = (x, y)
        expect = (z, None)
        worker(data, expect)

    for shape in [(2, 3), (8, 10), (99, 77)]:
        check(shape)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="windows disable MGB_ENABLE_OPR_MM"
)
@pytest.mark.skipif(get_device_count_by_fork("gpu") < 2, reason="need more gpu device")
@pytest.mark.isolated_distributed
def test_scatter():
    @dist.launcher(n_gpus=2)
    def worker(data, expect):
        rank = dist.get_rank()
        inp = tensor(data[rank])
        output = scatter(inp)
        assert np.allclose(output.numpy(), expect[rank])

    def check(shape):
        x = np.random.rand(*shape).astype("float32")
        y = x + 1
        data = (x, y)
        expect = (x[: shape[0] // 2], x[shape[0] // 2 :])
        worker(data, expect)

    for shape in [(2, 3), (8, 10), (100, 77)]:
        check(shape)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="windows disable MGB_ENABLE_OPR_MM"
)
@pytest.mark.skipif(get_device_count_by_fork("gpu") < 2, reason="need more gpu device")
@pytest.mark.isolated_distributed
def test_all_to_all():
    @dist.launcher(n_gpus=2)
    def worker(data, expect):
        rank = dist.get_rank()
        inp = tensor(data[rank])
        output = all_to_all(inp)
        assert np.allclose(output.numpy(), expect[rank])

    def check(shape):
        x = np.random.rand(*shape).astype("float32")
        y = np.random.rand(*shape).astype("float32")
        a = np.concatenate((x[: shape[0] // 2], y[: shape[0] // 2]))
        b = np.concatenate((x[shape[0] // 2 :], y[shape[0] // 2 :]))
        data = (x, y)
        expect = (a, b)
        worker(data, expect)

    for shape in [(2, 3), (8, 10), (100, 77)]:
        check(shape)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="windows disable MGB_ENABLE_OPR_MM"
)
@pytest.mark.skipif(get_device_count_by_fork("gpu") < 2, reason="need more gpu device")
@pytest.mark.isolated_distributed
def test_io_remote():
    val = np.random.rand(4, 5).astype(np.float32)

    @dist.launcher(n_gpus=2)
    def worker():
        rank = dist.get_rank()
        if rank == 0:  # remote send
            x = Tensor(val, device="gpu0")
            remote_send(x, 1)
            sync()
        else:  # remote recv
            y = remote_recv(0, val.shape, val.dtype)
            assert y.device == "gpu1"
            np.testing.assert_almost_equal(val, y.numpy())

    worker()
