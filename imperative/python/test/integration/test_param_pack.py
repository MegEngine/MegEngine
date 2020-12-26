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

import megengine
import megengine.autodiff as ad
import megengine.distributed as dist
import megengine.optimizer as optimizer
from megengine import Parameter, tensor
from megengine.distributed.helper import get_device_count_by_fork
from megengine.module import Module
from megengine.optimizer import SGD


class Simple(Module):
    def __init__(self):
        super().__init__()
        self.params = [Parameter(1.0, dtype=np.float32) for i in range(10)]

    def forward(self, x):
        for p in self.params:
            x = x * p
        return x


@pytest.mark.skipif(get_device_count_by_fork("gpu") < 2, reason="need more gpu device")
@pytest.mark.isolated_distributed
@pytest.mark.skipif(
    platform.system() == "Windows", reason="windows disable MGB_ENABLE_OPR_MM"
)
def test_param_pack():
    data = np.ones([1], dtype="float32")

    @dist.launcher
    def worker():
        net = Simple()
        opt = SGD(net.parameters(), lr=0.1)

        gm = ad.GradManager().attach(
            net.parameters(), callbacks=[dist.make_allreduce_cb("MEAN", dist.WORLD)]
        )

        opt.clear_grad()
        with gm:
            x = tensor(data)
            loss = net(x)
            loss = loss.sum()
            gm.backward(loss)
        for p in net.params:
            np.testing.assert_equal(p.grad.numpy(), 1)

    worker()


@pytest.mark.skipif(get_device_count_by_fork("gpu") < 2, reason="need more gpu device")
@pytest.mark.isolated_distributed
@pytest.mark.skipif(
    platform.system() == "Windows", reason="windows disable MGB_ENABLE_OPR_MM"
)
def test_param_pack_with_no_param():
    data = np.ones([1], dtype="float32")

    @dist.launcher
    def worker():
        net = Simple()
        opt = SGD(net.parameters(), lr=0.1)

        allreduce_cb = dist.make_allreduce_cb("MEAN", dist.WORLD)
        allreduce_cb._param_pack_thd = 0
        gm = ad.GradManager().attach(net.parameters(), callbacks=[allreduce_cb])

        opt.clear_grad()
        with gm:
            x = tensor(data)
            loss = net(x)
            loss = loss.sum()
            gm.backward(loss)
        for p in net.params:
            np.testing.assert_equal(p.grad.numpy(), 1)

    worker()
