# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np
import pytest

import megengine
import megengine.autodiff as ad
import megengine.distributed as dist
import megengine.functional as F
import megengine.optimizer as optimizer
from megengine import Parameter, tensor
from megengine.distributed.helper import get_device_count_by_fork
from megengine.jit import trace
from megengine.module import BatchNorm2d, Module, SyncBatchNorm


def run_frozen_bn(BNModule, use_trace=False, use_symbolic=False):
    nchannel = 3
    m = BNModule(nchannel, freeze=True)
    var = 4.0
    bias = 1.0
    shape = (1, nchannel, 1, 1)
    m.running_var[...] = var * F.ones(shape)
    m.running_mean[...] = bias * F.ones(shape)

    saved_var = m.running_var.numpy()
    saved_mean = m.running_mean.numpy()
    saved_wt = m.weight.numpy()
    saved_bias = m.bias.numpy()

    gm = ad.GradManager().attach(m.parameters())
    optim = optimizer.SGD(m.parameters(), lr=1.0)
    optim.clear_grad()

    data = np.random.random((6, nchannel, 2, 2)).astype("float32")

    def train_fn(d):
        for _ in range(3):
            with gm:
                loss = m(d).mean()
                gm.backward(loss)
            optim.step()
        return loss

    if use_trace:
        train_fn = trace(train_fn, symbolic=use_symbolic)

    for _ in range(3):
        loss = train_fn(megengine.Tensor(data))
        np.testing.assert_equal(m.running_var.numpy(), saved_var)
        np.testing.assert_equal(m.running_mean.numpy(), saved_mean)
        np.testing.assert_equal(m.weight.numpy(), saved_wt)
        np.testing.assert_equal(m.bias.numpy(), saved_bias)
        np.testing.assert_almost_equal(
            loss.numpy(), ((data - bias) / np.sqrt(var)).mean(), 5
        )


def test_frozen_bn():
    run_frozen_bn(BatchNorm2d)
    run_frozen_bn(BatchNorm2d, True, False)
    run_frozen_bn(BatchNorm2d, True, True)


@pytest.mark.skipif(get_device_count_by_fork("gpu") < 2, reason="need more gpu device")
@pytest.mark.isolated_distributed
def test_frozen_synced_bn():
    @dist.launcher(n_gpus=2)
    def worker():
        run_frozen_bn(SyncBatchNorm)
        run_frozen_bn(SyncBatchNorm, True, False)
        run_frozen_bn(SyncBatchNorm, True, True)

    worker()


def test_bn_no_track_stat():
    nchannel = 3
    m = BatchNorm2d(nchannel, track_running_stats=False)

    gm = ad.GradManager().attach(m.parameters())
    optim = optimizer.SGD(m.parameters(), lr=1.0)
    optim.clear_grad()

    data = np.random.random((6, nchannel, 2, 2)).astype("float32")
    with gm:
        loss = m(data).sum()
        gm.backward(loss)
    optim.step()


def test_bn_no_track_stat2():
    nchannel = 3
    m = BatchNorm2d(nchannel)  # Init with track_running_stat = True
    m.track_running_stats = False

    # m.running_var and m.running_mean created during init time
    saved_var = m.running_var.numpy()
    assert saved_var is not None
    saved_mean = m.running_mean.numpy()
    assert saved_mean is not None

    gm = ad.GradManager().attach(m.parameters())
    optim = optimizer.SGD(m.parameters(), lr=1.0)
    optim.clear_grad()

    data = np.random.random((6, nchannel, 2, 2)).astype("float32")
    with gm:
        loss = m(data).sum()
        gm.backward(loss)
    optim.step()

    np.testing.assert_equal(m.running_var.numpy(), saved_var)
    np.testing.assert_equal(m.running_mean.numpy(), saved_mean)


def test_bn_no_track_stat3():
    nchannel = 3
    m = BatchNorm2d(nchannel, track_running_stats=False)
    m.track_running_stats = True
    data = np.random.random((6, nchannel, 2, 2)).astype("float32")
    with pytest.raises(Exception):
        m(data)


def test_trace_bn_forward_twice():
    class Simple(Module):
        def __init__(self):
            super().__init__()
            self.bn = BatchNorm2d(1)

        def forward(self, inp):
            x = self.bn(inp)
            x = self.bn(x)
            return x

    @trace(symbolic=True)
    def train_bn(inp, net=None):
        net.train()
        pred = net(inp)
        return pred

    x = np.ones((1, 1, 32, 32), dtype=np.float32)
    y = train_bn(x, net=Simple())
    np.testing.assert_equal(y.numpy(), 0)


# https://github.com/MegEngine/MegEngine/issues/145
def test_frozen_bn_no_affine():
    nchannel = 3
    m = BatchNorm2d(nchannel, freeze=True, affine=False)
    data = megengine.Tensor(np.random.random((6, nchannel, 2, 2)).astype("float32"))
    m(data).numpy()
