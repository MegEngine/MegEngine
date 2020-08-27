# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import multiprocessing as mp
import platform

import numpy as np
import pytest

import megengine as mge
import megengine.distributed as dist
from megengine import Parameter, Tensor, tensor
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
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.isolated_distributed
def test_reduce_sum():
    world_size = 2
    port = dist.get_free_ports(1)[0]
    server = dist.Server(port)

    def worker(rank, data, expect, port):
        if mge.get_device_count("gpu") < world_size:
            return
        dist.init_process_group("localhost", port, world_size, rank, rank)
        inp = tensor(data)
        output = reduce_sum(inp)
        if rank == 0:
            assert np.allclose(output.numpy(), expect)
        else:
            assert np.allclose(output.numpy(), 0)

    def check(shape):
        x = np.random.rand(*shape).astype("float32")
        y = np.random.rand(*shape).astype("float32")
        z = x + y
        p0 = mp.Process(target=worker, args=(0, x, z, port))
        p1 = mp.Process(target=worker, args=(1, y, None, port))

        p0.start()
        p1.start()

        p0.join(10)
        p1.join(10)

        assert p0.exitcode == 0 and p1.exitcode == 0

    for shape in [(2, 3), (8, 10), (99, 77)]:
        check(shape)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.isolated_distributed
def test_broadcast():
    world_size = 2
    port = dist.get_free_ports(1)[0]
    server = dist.Server(port)

    def worker(rank, data, expect, port):
        if mge.get_device_count("gpu") < world_size:
            return
        dist.init_process_group("localhost", port, world_size, rank, rank)
        inp = tensor(data)
        output = broadcast(inp)
        assert np.allclose(output.numpy(), expect)

    def check(shape):
        x = np.random.rand(*shape).astype("float32")
        y = x + 1
        p0 = mp.Process(target=worker, args=(0, x, x, port))
        p1 = mp.Process(target=worker, args=(1, y, x, port))

        p0.start()
        p1.start()

        p0.join(10)
        p1.join(10)

        assert p0.exitcode == 0 and p1.exitcode == 0

    for shape in [(2, 3), (8, 10), (99, 77)]:
        check(shape)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.isolated_distributed
def test_all_gather():
    world_size = 2
    port = dist.get_free_ports(1)[0]
    server = dist.Server(port)

    def worker(rank, data, expect, port):
        if mge.get_device_count("gpu") < world_size:
            return
        dist.init_process_group("localhost", port, world_size, rank, rank)
        inp = tensor(data)
        output = all_gather(inp)
        assert np.allclose(output.numpy(), expect)

    def check(shape):
        x = np.random.rand(*shape).astype("float32")
        y = np.random.rand(*shape).astype("float32")
        z = np.concatenate((x, y))
        p0 = mp.Process(target=worker, args=(0, x, z, port))
        p1 = mp.Process(target=worker, args=(1, y, z, port))

        p0.start()
        p1.start()

        p0.join(10)
        p1.join(10)

        assert p0.exitcode == 0 and p1.exitcode == 0

    for shape in [(2, 3), (8, 10), (99, 77)]:
        check(shape)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.isolated_distributed
def test_reduce_scatter_sum():
    world_size = 2
    port = dist.get_free_ports(1)[0]
    server = dist.Server(port)

    def worker(rank, data, expect, port):
        if mge.get_device_count("gpu") < world_size:
            return
        dist.init_process_group("localhost", port, world_size, rank, rank)
        inp = tensor(data)
        output = reduce_scatter_sum(inp)
        assert np.allclose(output.numpy(), expect)

    def check(shape):
        x = np.random.rand(*shape).astype("float32")
        y = np.random.rand(*shape).astype("float32")
        z = x + y
        p0 = mp.Process(target=worker, args=(0, x, z[: shape[0] // 2], port))
        p1 = mp.Process(target=worker, args=(1, y, z[shape[0] // 2 :], port))

        p0.start()
        p1.start()

        p0.join(10)
        p1.join(10)

        assert p0.exitcode == 0 and p1.exitcode == 0

    for shape in [(2, 4), (8, 10), (88, 44)]:
        check(shape)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.isolated_distributed
def test_all_reduce_sum():
    world_size = 2
    port = dist.get_free_ports(1)[0]
    server = dist.Server(port)

    def worker(rank, data, expect, port):
        if mge.get_device_count("gpu") < world_size:
            return
        dist.init_process_group("localhost", port, world_size, rank, rank)
        inp = tensor(data)
        output = all_reduce_sum(inp)
        assert np.allclose(output.numpy(), expect)

    def check(shape):
        x = np.random.rand(*shape).astype("float32")
        y = np.random.rand(*shape).astype("float32")
        z = x + y
        p0 = mp.Process(target=worker, args=(0, x, z, port))
        p1 = mp.Process(target=worker, args=(1, y, z, port))

        p0.start()
        p1.start()

        p0.join(10)
        p1.join(10)

        assert p0.exitcode == 0 and p1.exitcode == 0

    for shape in [(2, 3), (8, 10), (99, 77)]:
        check(shape)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.isolated_distributed
def test_all_reduce_max():
    world_size = 2
    port = dist.get_free_ports(1)[0]
    server = dist.Server(port)

    def worker(rank, data, expect, port):
        if mge.get_device_count("gpu") < world_size:
            return
        dist.init_process_group("localhost", port, world_size, rank, rank)
        inp = tensor(data)
        output = all_reduce_max(inp)
        assert np.allclose(output.numpy(), expect)

    def check(shape):
        x = np.random.rand(*shape).astype("float32")
        y = np.random.rand(*shape).astype("float32")
        z = np.maximum(x, y)
        p0 = mp.Process(target=worker, args=(0, x, z, port))
        p1 = mp.Process(target=worker, args=(1, y, z, port))

        p0.start()
        p1.start()

        p0.join(10)
        p1.join(10)

        assert p0.exitcode == 0 and p1.exitcode == 0

    for shape in [(2, 3), (8, 10), (99, 77)]:
        check(shape)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.isolated_distributed
def test_all_reduce_min():
    world_size = 2
    port = dist.get_free_ports(1)[0]
    server = dist.Server(port)

    def worker(rank, data, expect, port):
        if mge.get_device_count("gpu") < world_size:
            return
        dist.init_process_group("localhost", port, world_size, rank, rank)
        inp = tensor(data)
        output = all_reduce_min(inp)
        assert np.allclose(output.numpy(), expect)

    def check(shape):
        x = np.random.rand(*shape).astype("float32")
        y = np.random.rand(*shape).astype("float32")
        z = np.minimum(x, y)
        p0 = mp.Process(target=worker, args=(0, x, z, port))
        p1 = mp.Process(target=worker, args=(1, y, z, port))

        p0.start()
        p1.start()

        p0.join(10)
        p1.join(10)

        assert p0.exitcode == 0 and p1.exitcode == 0

    for shape in [(2, 3), (8, 10), (99, 77)]:
        check(shape)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.isolated_distributed
def test_gather():
    world_size = 2
    port = dist.get_free_ports(1)[0]
    server = dist.Server(port)

    def worker(rank, data, expect, port):
        if mge.get_device_count("gpu") < world_size:
            return
        dist.init_process_group("localhost", port, world_size, rank, rank)
        inp = tensor(data)
        output = gather(inp)
        if rank == 0:
            assert np.allclose(output.numpy(), expect)
        else:
            assert np.allclose(output.numpy(), 0)

    def check(shape):
        x = np.random.rand(*shape).astype("float32")
        y = np.random.rand(*shape).astype("float32")
        z = np.concatenate((x, y))
        p0 = mp.Process(target=worker, args=(0, x, z, port))
        p1 = mp.Process(target=worker, args=(1, y, None, port))

        p0.start()
        p1.start()

        p0.join(10)
        p1.join(10)

        assert p0.exitcode == 0 and p1.exitcode == 0

    for shape in [(2, 3), (8, 10), (99, 77)]:
        check(shape)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.isolated_distributed
def test_scatter():
    world_size = 2
    port = dist.get_free_ports(1)[0]
    server = dist.Server(port)

    def worker(rank, data, expect, port):
        if mge.get_device_count("gpu") < world_size:
            return
        dist.init_process_group("localhost", port, world_size, rank, rank)
        inp = tensor(data)
        output = scatter(inp)
        assert np.allclose(output.numpy(), expect)

    def check(shape):
        x = np.random.rand(*shape).astype("float32")
        y = x + 1
        p0 = mp.Process(target=worker, args=(0, x, x[: shape[0] // 2], port))
        p1 = mp.Process(target=worker, args=(1, y, x[shape[0] // 2 :], port))

        p0.start()
        p1.start()

        p0.join(10)
        p1.join(10)

        assert p0.exitcode == 0 and p1.exitcode == 0

    for shape in [(2, 3), (8, 10), (100, 77)]:
        check(shape)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.isolated_distributed
def test_all_to_all():
    world_size = 2
    port = dist.get_free_ports(1)[0]
    server = dist.Server(port)

    def worker(rank, data, expect, port):
        if mge.get_device_count("gpu") < world_size:
            return
        dist.init_process_group("localhost", port, world_size, rank, rank)
        inp = tensor(data)
        output = all_to_all(inp)
        assert np.allclose(output.numpy(), expect)

    def check(shape):
        x = np.random.rand(*shape).astype("float32")
        y = np.random.rand(*shape).astype("float32")
        a = np.concatenate((x[: shape[0] // 2], y[: shape[0] // 2]))
        b = np.concatenate((x[shape[0] // 2 :], y[shape[0] // 2 :]))
        p0 = mp.Process(target=worker, args=(0, x, a, port))
        p1 = mp.Process(target=worker, args=(1, y, b, port))

        p0.start()
        p1.start()

        p0.join(10)
        p1.join(10)

        assert p0.exitcode == 0 and p1.exitcode == 0

    for shape in [(2, 3), (8, 10), (100, 77)]:
        check(shape)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.isolated_distributed
def test_io_remote():
    world_size = 2
    port = dist.get_free_ports(1)[0]
    server = dist.Server(port)
    val = np.random.rand(4, 5).astype(np.float32)

    def worker(rank):
        if mge.get_device_count("gpu") < world_size:
            return
        if rank == 0:  # remote send
            dist.init_process_group("localhost", port, world_size, rank, rank)
            x = Tensor(val, device="gpu0")
            y = remote_send(x, 1)
            assert y.numpy()[0] == 0
        else:  # remote recv
            dist.init_process_group("localhost", port, world_size, rank, rank)
            y = remote_recv(0, val.shape, val.dtype, cn="gpu1")
            np.testing.assert_almost_equal(val, y.numpy())

    procs = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank,))
        p.start()
        procs.append(p)

    for p in procs:
        p.join(10)
        assert p.exitcode == 0
