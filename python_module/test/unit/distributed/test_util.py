# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import multiprocessing as mp
import queue
from time import sleep

import pytest

import megengine as mge
import megengine._internal as mgb
import megengine.distributed as dist

_LOCALHOST = "127.0.0.1"


def _assert_q_empty(q):
    try:
        res = q.get(timeout=1)
    except Exception as e:
        assert isinstance(e, queue.Empty)
    else:
        assert False, "queue is not empty"


def _assert_q_val(q, val):
    ret = q.get()
    assert ret == val


def _init_process_group_wrapper(world_size, rank, dev, backend, q):
    if rank == 0:
        dist.init_process_group(_LOCALHOST, 0, world_size, rank, dev, backend)
        q.put(dist.get_master_port())
    else:
        port = q.get()
        dist.init_process_group(_LOCALHOST, port, world_size, rank, dev, backend)


@pytest.mark.isolated_distributed
def test_create_mm_server():
    def worker():
        if not mge.is_cuda_available():
            return
        port = mgb.config.create_mm_server("0.0.0.0", 0)
        assert port > 0
        res = mgb.config.create_mm_server("0.0.0.0", port)
        assert res == -1

    p = mp.Process(target=worker)

    p.start()

    p.join(10)

    assert p.exitcode == 0


@pytest.mark.isolated_distributed
def test_init_process_group():
    world_size = 2

    def worker(rank, backend, q):
        if not mge.is_cuda_available():
            return
        _init_process_group_wrapper(world_size, rank, rank, backend, q)
        assert dist.is_distributed() == True
        assert dist.get_master_ip() == _LOCALHOST
        assert dist.get_master_port() > 0
        assert dist.get_world_size() == world_size
        assert dist.get_rank() == rank
        assert dist.get_backend() == backend

    def check(backend):
        Q = mp.Queue()
        p0 = mp.Process(target=worker, args=(0, backend, Q))
        p1 = mp.Process(target=worker, args=(1, backend, Q))

        p0.start()
        p1.start()

        p0.join(10)
        p1.join(10)

        assert p0.exitcode == 0 and p1.exitcode == 0

    check("nccl")
    check("ucx")


@pytest.mark.isolated_distributed
def test_group_barrier():
    world_size = 2
    ip = "127.0.0.1"
    backend = "nccl"

    def worker(rank, q):
        if not mge.is_cuda_available():
            return
        _init_process_group_wrapper(world_size, rank, rank, backend, q)
        dist.group_barrier()
        if rank == 0:
            dist.group_barrier()
            q.put(0)  # to be observed in rank 1
        else:
            _assert_q_empty(q)  # q.put(0) is not executed in rank 0
            dist.group_barrier()
            _assert_q_val(q, 0)  # q.put(0) executed in rank 0

    Q = mp.Queue()
    p0 = mp.Process(target=worker, args=(0, Q))
    p1 = mp.Process(target=worker, args=(1, Q))

    p0.start()
    p1.start()

    p0.join(10)
    p1.join(10)

    assert p0.exitcode == 0 and p1.exitcode == 0


@pytest.mark.isolated_distributed
def test_synchronized():
    world_size = 2
    backend = "nccl"

    @dist.synchronized
    def func(rank, q):
        q.put(rank)

    def worker(rank, q):
        if not mge.is_cuda_available():
            return
        _init_process_group_wrapper(world_size, rank, rank, backend, q)
        dist.group_barrier()
        if rank == 0:
            func(0, q)  # q.put(0)
            q.put(2)
        else:
            _assert_q_val(q, 0)  # func executed in rank 0
            _assert_q_empty(q)  # q.put(2) is not executed
            func(1, q)
            _assert_q_val(
                q, 1
            )  # func in rank 1 executed earlier than q.put(2) in rank 0
            _assert_q_val(q, 2)  # q.put(2) executed in rank 0

    Q = mp.Queue()
    p0 = mp.Process(target=worker, args=(0, Q))
    p1 = mp.Process(target=worker, args=(1, Q))

    p0.start()
    p1.start()

    p0.join(10)
    p1.join(10)

    assert p0.exitcode == 0 and p1.exitcode == 0
