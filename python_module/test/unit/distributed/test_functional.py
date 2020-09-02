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
from megengine.core import Parameter, tensor


def _init_process_group_wrapper(world_size, rank, dev, backend, q):
    if rank == 0:
        dist.init_process_group("localhost", 0, world_size, rank, dev, backend)
        q.put(dist.get_master_port())
    else:
        port = q.get()
        dist.init_process_group("localhost", port, world_size, rank, dev, backend)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.isolated_distributed
def test_reduce_sum():
    world_size = 2

    def worker(rank, data, backend, expect, port_queue):
        if mge.get_device_count("gpu") < world_size:
            return
        _init_process_group_wrapper(world_size, rank, rank, backend, port_queue)
        inp = tensor(data)
        output = dist.functional.reduce_sum(inp)
        if rank == 0:
            assert np.allclose(output.numpy(), expect)
        else:
            assert np.allclose(output.numpy(), 0)

    def check(shape, backend):
        port_queue = mp.Queue()
        x = np.random.rand(*shape).astype("float32")
        y = np.random.rand(*shape).astype("float32")
        z = x + y
        p0 = mp.Process(target=worker, args=(0, x, backend, z, port_queue))
        p1 = mp.Process(target=worker, args=(1, y, backend, None, port_queue))

        p0.start()
        p1.start()

        p0.join(10)
        p1.join(10)

        assert p0.exitcode == 0 and p1.exitcode == 0

    for shape in [(2, 3), (8, 10), (99, 77)]:
        for backend in ["nccl"]:
            check(shape, backend)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.isolated_distributed
def test_gather():
    world_size = 2

    def worker(rank, data, backend, expect, port_queue):
        if mge.get_device_count("gpu") < world_size:
            return
        _init_process_group_wrapper(world_size, rank, rank, backend, port_queue)
        inp = tensor(data)
        output = dist.functional.gather(inp)
        if rank == 0:
            assert np.allclose(output.numpy(), expect)
        else:
            assert np.allclose(output.numpy(), 0)

    def check(shape, backend):
        port_queue = mp.Queue()
        x = np.random.rand(*shape).astype("float32")
        y = np.random.rand(*shape).astype("float32")
        z = np.concatenate((x, y))
        p0 = mp.Process(target=worker, args=(0, x, backend, z, port_queue))
        p1 = mp.Process(target=worker, args=(1, y, backend, None, port_queue))

        p0.start()
        p1.start()

        p0.join(10)
        p1.join(10)

        assert p0.exitcode == 0 and p1.exitcode == 0

    for shape in [(2, 3), (8, 10), (99, 77)]:
        for backend in ["nccl"]:
            check(shape, backend)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.isolated_distributed
def test_broadcast():
    world_size = 2

    def worker(rank, data, backend, expect, port_queue):
        if mge.get_device_count("gpu") < world_size:
            return
        _init_process_group_wrapper(world_size, rank, rank, backend, port_queue)
        inp = tensor(data)
        output = dist.functional.broadcast(inp)
        assert np.allclose(output.numpy(), expect)

    def check(shape, backend):
        port_queue = mp.Queue()
        x = np.random.rand(*shape).astype("float32")
        y = x + 1
        p0 = mp.Process(target=worker, args=(0, x, backend, x, port_queue))
        p1 = mp.Process(target=worker, args=(1, y, backend, x, port_queue))

        p0.start()
        p1.start()

        p0.join(10)
        p1.join(10)

        assert p0.exitcode == 0 and p1.exitcode == 0

    for shape in [(2, 3), (8, 10), (99, 77)]:
        for backend in ["nccl"]:
            check(shape, backend)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.isolated_distributed
def test_scatter():
    world_size = 2

    def worker(rank, data, backend, expect, port_queue):
        if mge.get_device_count("gpu") < world_size:
            return
        _init_process_group_wrapper(world_size, rank, rank, backend, port_queue)
        inp = tensor(data)
        output = dist.functional.scatter(inp)
        assert np.allclose(output.numpy(), expect)

    def check(shape, backend):
        port_queue = mp.Queue()
        x = np.random.rand(*shape).astype("float32")
        y = x + 1
        p0 = mp.Process(
            target=worker, args=(0, x, backend, x[: shape[0] // 2], port_queue)
        )
        p1 = mp.Process(
            target=worker, args=(1, y, backend, x[shape[0] // 2 :], port_queue)
        )

        p0.start()
        p1.start()

        p0.join(10)
        p1.join(10)

        assert p0.exitcode == 0 and p1.exitcode == 0

    for shape in [(2, 3), (8, 10), (100, 77)]:
        for backend in ["nccl"]:
            check(shape, backend)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.isolated_distributed
def test_all_to_all():
    world_size = 2

    def worker(rank, data, backend, expect, port_queue):
        if mge.get_device_count("gpu") < world_size:
            return
        _init_process_group_wrapper(world_size, rank, rank, backend, port_queue)
        inp = tensor(data)
        output = dist.functional.all_to_all(inp)
        assert np.allclose(output.numpy(), expect)

    def check(shape, backend):
        port_queue = mp.Queue()
        x = np.random.rand(*shape).astype("float32")
        y = np.random.rand(*shape).astype("float32")
        a = np.concatenate((x[: shape[0] // 2], y[: shape[0] // 2]))
        b = np.concatenate((x[shape[0] // 2 :], y[shape[0] // 2 :]))
        p0 = mp.Process(target=worker, args=(0, x, backend, a, port_queue))
        p1 = mp.Process(target=worker, args=(1, y, backend, b, port_queue))

        p0.start()
        p1.start()

        p0.join(10)
        p1.join(10)

        assert p0.exitcode == 0 and p1.exitcode == 0

    for shape in [(2, 3), (8, 10), (100, 77)]:
        for backend in ["nccl"]:
            check(shape, backend)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.isolated_distributed
def test_all_gather():
    world_size = 2

    def worker(rank, data, backend, expect, port_queue):
        if mge.get_device_count("gpu") < world_size:
            return
        _init_process_group_wrapper(world_size, rank, rank, backend, port_queue)
        inp = tensor(data)
        output = dist.functional.all_gather(inp)
        assert np.allclose(output.numpy(), expect)

    def check(shape, backend):
        port_queue = mp.Queue()
        x = np.random.rand(*shape).astype("float32")
        y = np.random.rand(*shape).astype("float32")
        z = np.concatenate((x, y))
        p0 = mp.Process(target=worker, args=(0, x, backend, z, port_queue))
        p1 = mp.Process(target=worker, args=(1, y, backend, z, port_queue))

        p0.start()
        p1.start()

        p0.join(10)
        p1.join(10)

        assert p0.exitcode == 0 and p1.exitcode == 0

    for shape in [(2, 3), (8, 10), (99, 77)]:
        for backend in ["nccl"]:
            check(shape, backend)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.isolated_distributed
def test_reduce_scatter_sum():
    world_size = 2

    def worker(rank, data, backend, expect, port_queue):
        if mge.get_device_count("gpu") < world_size:
            return
        _init_process_group_wrapper(world_size, rank, rank, backend, port_queue)
        inp = tensor(data)
        output = dist.functional.reduce_scatter_sum(inp)
        assert np.allclose(output.numpy(), expect)

    def check(shape, backend):
        port_queue = mp.Queue()
        x = np.random.rand(*shape).astype("float32")
        y = np.random.rand(*shape).astype("float32")
        z = x + y
        p0 = mp.Process(
            target=worker, args=(0, x, backend, z[: shape[0] // 2], port_queue)
        )
        p1 = mp.Process(
            target=worker, args=(1, y, backend, z[shape[0] // 2 :], port_queue)
        )

        p0.start()
        p1.start()

        p0.join(10)
        p1.join(10)

        assert p0.exitcode == 0 and p1.exitcode == 0

    for shape in [(2, 4), (8, 10), (88, 44)]:
        for backend in ["nccl"]:
            check(shape, backend)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.isolated_distributed
def test_all_reduce_sum():
    world_size = 2

    def worker(rank, data, backend, expect, port_queue):
        if mge.get_device_count("gpu") < world_size:
            return
        _init_process_group_wrapper(world_size, rank, rank, backend, port_queue)
        inp = tensor(data)
        output = dist.functional.all_reduce_sum(inp)
        assert np.allclose(output.numpy(), expect)

    def check(shape, backend):
        port_queue = mp.Queue()
        x = np.random.rand(*shape).astype("float32")
        y = np.random.rand(*shape).astype("float32")
        z = x + y
        p0 = mp.Process(target=worker, args=(0, x, backend, z, port_queue))
        p1 = mp.Process(target=worker, args=(1, y, backend, z, port_queue))

        p0.start()
        p1.start()

        p0.join(10)
        p1.join(10)

        assert p0.exitcode == 0 and p1.exitcode == 0

    for shape in [(2, 3), (8, 10), (99, 77)]:
        for backend in ["nccl"]:
            check(shape, backend)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.isolated_distributed
def test_all_reduce_max():
    world_size = 2

    def worker(rank, data, backend, expect, port_queue):
        if mge.get_device_count("gpu") < world_size:
            return
        _init_process_group_wrapper(world_size, rank, rank, backend, port_queue)
        inp = tensor(data)
        output = dist.functional.all_reduce_max(inp)
        assert np.allclose(output.numpy(), expect)

    def check(shape, backend):
        port_queue = mp.Queue()
        x = np.random.rand(*shape).astype("float32")
        y = np.random.rand(*shape).astype("float32")
        z = np.maximum(x, y)
        p0 = mp.Process(target=worker, args=(0, x, backend, z, port_queue))
        p1 = mp.Process(target=worker, args=(1, y, backend, z, port_queue))

        p0.start()
        p1.start()

        p0.join(10)
        p1.join(10)

        assert p0.exitcode == 0 and p1.exitcode == 0

    for shape in [(2, 3), (8, 10), (99, 77)]:
        for backend in ["nccl"]:
            check(shape, backend)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.isolated_distributed
def test_all_reduce_min():
    world_size = 2

    def worker(rank, data, backend, expect, port_queue):
        if mge.get_device_count("gpu") < world_size:
            return
        _init_process_group_wrapper(world_size, rank, rank, backend, port_queue)
        inp = tensor(data)
        output = dist.functional.all_reduce_min(inp)
        assert np.allclose(output.numpy(), expect)

    def check(shape, backend):
        port_queue = mp.Queue()
        x = np.random.rand(*shape).astype("float32")
        y = np.random.rand(*shape).astype("float32")
        z = np.minimum(x, y)
        p0 = mp.Process(target=worker, args=(0, x, backend, z, port_queue))
        p1 = mp.Process(target=worker, args=(1, y, backend, z, port_queue))

        p0.start()
        p1.start()

        p0.join(10)
        p1.join(10)

        assert p0.exitcode == 0 and p1.exitcode == 0

    for shape in [(2, 3), (8, 10), (99, 77)]:
        for backend in ["nccl"]:
            check(shape, backend)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.isolated_distributed
def test_bcast_param():
    world_size = 2

    def worker(rank, data, backend, expect, port_queue):
        if mge.get_device_count("gpu") < world_size:
            return
        _init_process_group_wrapper(world_size, rank, rank, backend, port_queue)
        inp = Parameter(data)
        dist.functional.bcast_param(inp)
        assert np.allclose(inp.numpy(), expect)

    def check(shape, backend):
        port_queue = mp.Queue()
        x = np.random.rand(*shape).astype("float32")
        y = x + 1
        p0 = mp.Process(target=worker, args=(0, x, backend, x, port_queue))
        p1 = mp.Process(target=worker, args=(1, y, backend, x, port_queue))

        p0.start()
        p1.start()

        p0.join(10)
        p1.join(10)

        assert p0.exitcode == 0 and p1.exitcode == 0

    for shape in [(2, 3), (8, 10), (99, 77)]:
        for backend in ["nccl"]:
            check(shape, backend)
