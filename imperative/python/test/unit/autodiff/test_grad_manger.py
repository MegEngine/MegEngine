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
import megengine.functional as F
import megengine.module as M
import megengine.optimizer as optim
from megengine.autodiff import GradManager
from megengine.core._imperative_rt.imperative import sync
from megengine.distributed.helper import get_device_count_by_fork
from megengine.jit import trace


def test_basic():
    x = mge.tensor([1.0, 3.0, 5.0]).reshape(1, 3)
    w = mge.tensor([2.0, 4.0, 6.0]).reshape(3, 1)
    b = mge.tensor(-1.0)

    gm = GradManager().attach([w, b])
    gm.record()

    p = F.matmul(x, w)
    y = p + b

    gm.backward(y)
    gm.release()  # is not necessary
    np.testing.assert_equal(w.grad.numpy(), [[1], [3], [5]])
    np.testing.assert_equal(b.grad.numpy(), [1])

    gm.clear_grad()
    with gm:
        p = F.matmul(x, w)
        y = p + b
        gm.backward(y)

    np.testing.assert_equal(w.grad.numpy(), [[1], [3], [5]])
    np.testing.assert_equal(b.grad.numpy(), [1])


def test_attach_in_with_block():
    a = mge.Parameter([1.0])
    gm = GradManager()
    with gm:
        b = a * 3
        gm.attach(b)
        c = b + 1
        gm.backward(c)
    assert int(b.grad.numpy()) == 1


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="windows disable MGB_ENABLE_OPR_MM"
)
@pytest.mark.skipif(get_device_count_by_fork("gpu") < 2, reason="need more gpu device")
@pytest.mark.isolated_distributed
def test_remote_grad():
    @dist.launcher
    def worker():
        rank = dist.get_rank()
        size = dist.get_world_size()
        x = mge.tensor(np.random.randn(1, rank * 2 + 2), dtype=np.float32)
        m = M.Linear(rank * 2 + 2, rank * 2 + 4)
        gm = GradManager().attach(m.parameters())
        opt = optim.SGD(m.parameters(), 1e-3, momentum=0.9)

        @trace(symbolic=True)
        def train_func(x):
            with gm:
                if rank != 0:
                    x = dist.functional.remote_recv(
                        rank - 1, shape=(1, rank * 2 + 2), dtype=np.float32
                    )
                y = m(x)
                if rank != size - 1:
                    y = dist.functional.remote_send(y, dest_rank=rank + 1)
                if rank == size - 1:
                    y = y.mean()
                    gm.backward(y)
                else:
                    gm.backward()
                opt.step().clear_grad()

        for i in range(3):
            train_func(x)

        for param in m.parameters():
            param.numpy()

    worker()
