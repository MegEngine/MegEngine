# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import platform

import pytest


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.isolated_distributed
def test_syncbn():
    import numpy as np
    import multiprocessing as mp
    from megengine.distributed.group import Server

    nr_chan = 8
    nr_ranks = 4
    data_shape = (3, nr_chan, 4, nr_ranks * 8)
    momentum = 0.9
    eps = 1e-5
    running_mean = np.zeros((1, nr_chan, 1, 1), dtype=np.float32)
    running_var = np.ones((1, nr_chan, 1, 1), dtype=np.float32)
    steps = 4
    server = Server(0)
    port = server.py_server_port

    def worker(rank, data, yv_expect, running_mean, running_var):
        import megengine as mge
        import megengine.distributed as dist
        from megengine import tensor
        from megengine.module import SyncBatchNorm
        from megengine.distributed.group import Group
        from megengine.test import assertTensorClose

        if mge.get_device_count("gpu") < nr_ranks:
            return
        dist.init_process_group("localhost", port, nr_ranks, rank, rank)
        group = Group([i for i in range(nr_ranks)])
        bn = SyncBatchNorm(nr_chan, eps=eps, momentum=momentum, group=group)
        data_tensor = None
        for i in range(steps):
            if data_tensor is None:
                data_tensor = tensor(data[i], device=f"gpu{rank}:0")
            else:
                data_tensor.set_value(data[i])
            yv = bn(data_tensor)

        assertTensorClose(yv_expect, yv.numpy(), max_err=5e-6)
        assertTensorClose(running_mean, bn.running_mean.numpy(), max_err=5e-6)
        assertTensorClose(running_var, bn.running_var.numpy(), max_err=5e-6)

    xv = []
    for i in range(steps):
        xv.append(np.random.normal(loc=2.3, size=data_shape).astype(np.float32))
        xv_transposed = np.transpose(xv[i], [0, 2, 3, 1]).reshape(
            (data_shape[0] * data_shape[2] * data_shape[3], nr_chan)
        )

        mean = np.mean(xv_transposed, axis=0).reshape(1, nr_chan, 1, 1)

        var_biased = np.var(xv_transposed, axis=0).reshape((1, nr_chan, 1, 1))
        sd = np.sqrt(var_biased + eps)

        var_unbiased = np.var(xv_transposed, axis=0, ddof=1).reshape((1, nr_chan, 1, 1))
        running_mean = running_mean * momentum + mean * (1 - momentum)
        running_var = running_var * momentum + var_unbiased * (1 - momentum)

        yv_expect = (xv[i] - mean) / sd

    data = []
    for i in range(nr_ranks):
        data.append([])
        for j in range(steps):
            data[i].append(xv[j][:, :, :, i * 8 : i * 8 + 8])

    procs = []
    for rank in range(nr_ranks):
        p = mp.Process(
            target=worker,
            args=(
                rank,
                data[rank],
                yv_expect[:, :, :, rank * 8 : rank * 8 + 8],
                running_mean,
                running_var,
            ),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join(10)
        assert p.exitcode == 0


def test_module_conv2d():
    from megengine.module.conv import Conv2d

    conv = Conv2d(2, 3, 1)
