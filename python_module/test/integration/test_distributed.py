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
import subprocess
import sys

import numpy as np
import pytest


def worker(master_ip, master_port, world_size, rank, dev, trace):
    import megengine.distributed as dist
    import megengine.functional as F
    from megengine import is_cuda_available
    from megengine import jit
    from megengine.module import Linear, Module
    from megengine.optimizer import SGD

    if not is_cuda_available():
        return

    class MLP(Module):
        def __init__(self):
            super().__init__()
            self.fc0 = Linear(3 * 224 * 224, 500)
            self.fc1 = Linear(500, 10)

        def forward(self, x):
            x = self.fc0(x)
            x = F.relu(x)
            x = self.fc1(x)
            return x

    dist.init_process_group(
        master_ip=master_ip, master_port=3456, world_size=world_size, rank=rank, dev=dev
    )
    net = MLP()

    opt = SGD(net.parameters(requires_grad=True), lr=0.02)

    data = np.random.random((64, 3 * 224 * 224)).astype(np.float32)
    label = np.random.randint(0, 10, size=(64,)).astype(np.int32)

    jit.trace.enabled = trace

    @jit.trace()
    def train_func(data, label):
        pred = net(data)
        loss = F.cross_entropy_with_softmax(pred, label)
        opt.backward(loss)
        return loss

    for i in range(5):
        opt.zero_grad()
        loss = train_func(data, label)
        opt.step()


def start_workers(worker, world_size, trace=False):
    def run_subproc(rank):
        cmd = "from test.integration.test_distributed import worker\n"
        cmd += "worker('localhost', 3456, {}, {}, {}, {})".format(
            world_size, rank, rank, "True" if trace else "False"
        )
        cmd = [sys.executable, "-c", cmd]
        ret = subprocess.run(
            cmd, stdout=sys.stdout, stderr=sys.stderr, universal_newlines=True
        )
        assert ret.returncode == 0, "subprocess failed"

    procs = []
    for rank in range(world_size):
        p = mp.Process(target=run_subproc, args=(rank,))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        assert p.exitcode == 0


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
def test_distributed():
    start_workers(worker, 2, trace=True)
    start_workers(worker, 2, trace=False)
