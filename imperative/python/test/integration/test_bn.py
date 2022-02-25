# -*- coding: utf-8 -*-
import numpy as np
import pytest

import megengine
import megengine.autodiff as ad
import megengine.distributed as dist
import megengine.functional as F
import megengine.optimizer as optimizer
from megengine import tensor
from megengine.jit import trace
from megengine.module import BatchNorm2d, Conv2d, Module, Sequential, SyncBatchNorm


def run_frozen_bn(BNModule, is_training, use_trace, use_symbolic):
    nchannel = 3
    m = BNModule(nchannel, freeze=True)
    if is_training:
        m.train()
    else:
        m.eval()
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
        loss = train_fn(megengine.tensor(data))
        if not is_training:
            np.testing.assert_equal(m.running_var.numpy(), saved_var)
            np.testing.assert_equal(m.running_mean.numpy(), saved_mean)
            np.testing.assert_almost_equal(
                loss.numpy(), ((data - bias) / np.sqrt(var)).mean(), 5
            )
        np.testing.assert_equal(m.weight.numpy(), saved_wt)
        np.testing.assert_equal(m.bias.numpy(), saved_bias)


@pytest.mark.parametrize("is_training", [False, True])
@pytest.mark.parametrize("use_trace", [False, True])
@pytest.mark.parametrize("use_symbolic", [False, True])
def test_frozen_bn(is_training, use_trace, use_symbolic):
    run_frozen_bn(BatchNorm2d, is_training, use_trace, use_symbolic)


@pytest.mark.require_ngpu(2)
@pytest.mark.isolated_distributed
@pytest.mark.parametrize("is_training", [False, True])
@pytest.mark.parametrize("use_trace", [False, True])
@pytest.mark.parametrize("use_symbolic", [False, True])
def test_frozen_synced_bn(is_training, use_trace, use_symbolic):
    @dist.launcher(n_gpus=2)
    def worker():
        run_frozen_bn(SyncBatchNorm, is_training, use_trace, use_symbolic)

    worker()


def test_bn_no_track_stat():
    nchannel = 3
    m = BatchNorm2d(nchannel, track_running_stats=False)

    gm = ad.GradManager().attach(m.parameters())
    optim = optimizer.SGD(m.parameters(), lr=1.0)
    optim.clear_grad()

    data = tensor(np.random.random((6, nchannel, 2, 2)).astype("float32"))
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

    data = tensor(np.random.random((6, nchannel, 2, 2)).astype("float32"))
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

    x = tensor(np.ones((1, 1, 32, 32), dtype=np.float32))
    y = train_bn(x, net=Simple())
    np.testing.assert_equal(y.numpy(), 0)


def run_syncbn(trace_mode):
    x = F.ones([2, 16, 4, 4], dtype="float32")

    net = Sequential(
        Conv2d(16, 16, 1), SyncBatchNorm(16), Conv2d(16, 16, 1), SyncBatchNorm(16),
    )

    gm = ad.GradManager().attach(
        net.parameters(), callbacks=dist.make_allreduce_cb("MEAN")
    )
    opt = optimizer.SGD(net.parameters(), 1e-3)

    def train_func(x):
        with gm:
            y = net(x)
            loss = y.mean()
            gm.backward(loss)
            opt.step().clear_grad()
        return loss

    if trace_mode is not None:
        train_func = trace(train_func, symbolic=trace_mode)

    for _ in range(3):
        loss = train_func(x)
        loss.numpy()


@pytest.mark.require_ngpu(2)
@pytest.mark.isolated_distributed
@pytest.mark.parametrize("trace_mode", [None, True, False])
def test_trace_several_syncbn(trace_mode):
    @dist.launcher(n_gpus=2)
    def worker():
        run_syncbn(trace_mode)

    worker()


# https://github.com/MegEngine/MegEngine/issues/145
@pytest.mark.parametrize("is_training", [False, True])
def test_frozen_bn_no_affine(is_training):
    nchannel = 3
    m = BatchNorm2d(nchannel, freeze=True, affine=False)
    if is_training:
        m.train()
    else:
        m.eval()
    data = megengine.tensor(np.random.random((6, nchannel, 2, 2)).astype("float32"))
    m(data).numpy()
