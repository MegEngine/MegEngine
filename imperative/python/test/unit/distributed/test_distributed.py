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
import queue

import numpy as np
import pytest

import megengine as mge
import megengine.distributed as dist
from megengine.core.ops.builtin import CollectiveComm, ParamPackConcat, ParamPackSplit
from megengine.distributed.helper import (
    get_device_count_by_fork,
    param_pack_concat,
    param_pack_split,
)


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


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.skipif(get_device_count_by_fork("gpu") < 2, reason="need more gpu device")
@pytest.mark.isolated_distributed
def test_init_process_group():
    world_size = 2
    port = dist.get_free_ports(1)[0]
    server = dist.Server(port)

    def worker(rank, backend):
        dist.init_process_group("localhost", port, world_size, rank, rank, backend)
        assert dist.is_distributed() == True
        assert dist.get_rank() == rank
        assert dist.get_world_size() == world_size
        assert dist.get_backend() == backend

        py_server_addr = dist.get_py_server_addr()
        assert py_server_addr[0] == "localhost"
        assert py_server_addr[1] == port

        mm_server_addr = dist.get_mm_server_addr()
        assert mm_server_addr[0] == "localhost"
        assert mm_server_addr[1] > 0

        assert isinstance(dist.get_client(), dist.Client)

    def check(backend):
        procs = []
        for rank in range(world_size):
            p = mp.Process(target=worker, args=(rank, backend))
            p.start()
            procs.append(p)

        for p in procs:
            p.join(20)
            assert p.exitcode == 0

    check("nccl")


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.skipif(get_device_count_by_fork("gpu") < 2, reason="need more gpu device")
@pytest.mark.isolated_distributed
def test_new_group():
    world_size = 3
    ranks = [2, 0]
    port = dist.get_free_ports(1)[0]
    server = dist.Server(port)

    def worker(rank):
        dist.init_process_group("localhost", port, world_size, rank, rank)
        if rank in ranks:
            group = dist.new_group(ranks)
            assert group.size == 2
            assert group.key == "2,0"
            assert group.rank == ranks.index(rank)
            assert group.comp_node == "gpu{}:2".format(rank)

    procs = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank,))
        p.start()
        procs.append(p)

    for p in procs:
        p.join(20)
        assert p.exitcode == 0


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.skipif(get_device_count_by_fork("gpu") < 2, reason="need more gpu device")
@pytest.mark.isolated_distributed
def test_group_barrier():
    world_size = 2
    port = dist.get_free_ports(1)[0]
    server = dist.Server(port)

    def worker(rank, q):
        dist.init_process_group("localhost", port, world_size, rank, rank)
        dist.group_barrier()
        if rank == 0:
            dist.group_barrier()
            q.put(0)  # to be observed in rank 1
        else:
            _assert_q_empty(q)  # q.put(0) is not executed in rank 0
            dist.group_barrier()
            _assert_q_val(q, 0)  # q.put(0) executed in rank 0

    Q = mp.Queue()
    procs = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, Q))
        p.start()
        procs.append(p)

    for p in procs:
        p.join(20)
        assert p.exitcode == 0


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.skipif(get_device_count_by_fork("gpu") < 2, reason="need more gpu device")
@pytest.mark.isolated_distributed
def test_synchronized():
    world_size = 2
    port = dist.get_free_ports(1)[0]
    server = dist.Server(port)

    @dist.synchronized
    def func(rank, q):
        q.put(rank)

    def worker(rank, q):
        dist.init_process_group("localhost", port, world_size, rank, rank)
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
    procs = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, Q))
        p.start()
        procs.append(p)

    for p in procs:
        p.join(20)
        assert p.exitcode == 0


def test_oprmm_hashable():
    lhs = (CollectiveComm(), ParamPackConcat(), ParamPackSplit())
    rhs = (CollectiveComm(), ParamPackConcat(), ParamPackSplit())
    assert lhs == rhs
    assert hash(lhs) == hash(rhs)


def test_param_pack_split():
    a = mge.Tensor(np.ones((10,), np.int32))
    b, c = param_pack_split(a, [0, 1, 1, 10], [(1,), (3, 3)])
    assert np.allclose(b.numpy(), a.numpy()[1])
    assert np.allclose(c.numpy(), a.numpy()[1:].reshape(3, 3))


def test_param_pack_concat():
    a = mge.Tensor(np.ones((1,), np.int32))
    b = mge.Tensor(np.ones((3, 3), np.int32))
    offsets_val = [0, 1, 1, 10]
    offsets = mge.Tensor(offsets_val, np.int32)
    c = param_pack_concat([a, b], offsets, offsets_val)
    assert np.allclose(np.concatenate([a.numpy(), b.numpy().flatten()]), c.numpy())
