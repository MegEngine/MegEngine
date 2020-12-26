# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import functools
import platform

import numpy as np
import pytest

import megengine as mge
import megengine.distributed as dist
from megengine import Tensor
from megengine.core._trace_option import use_symbolic_shape
from megengine.distributed.helper import get_device_count_by_fork
from megengine.module import BatchNorm1d, BatchNorm2d, SyncBatchNorm

_assert_allclose = functools.partial(np.testing.assert_allclose, atol=5e-6, rtol=5e-6)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="windows disable MGB_ENABLE_OPR_MM"
)
@pytest.mark.skipif(get_device_count_by_fork("gpu") < 2, reason="need more gpu device")
@pytest.mark.isolated_distributed
def test_syncbn():
    nr_chan = 8
    data_shape = (3, nr_chan, 4, 16)
    momentum = 0.9
    eps = 1e-5
    running_mean = np.zeros((1, nr_chan, 1, 1), dtype=np.float32)
    running_var = np.ones((1, nr_chan, 1, 1), dtype=np.float32)
    steps = 4
    nr_ranks = 2
    server = dist.Server()
    port = server.py_server_port

    @dist.launcher(n_gpus=2)
    def worker(data, yv_expect, running_mean, running_var):
        rank = dist.get_rank()
        bn = SyncBatchNorm(nr_chan, momentum=momentum, eps=eps)
        for i in range(steps):
            yv = bn(Tensor(data[rank][i]))

        _assert_allclose(yv.numpy(), yv_expect[rank])
        _assert_allclose(bn.running_mean.numpy(), running_mean)
        _assert_allclose(bn.running_var.numpy(), running_var)

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

    yv_expect = [yv_expect[:, :, :, i * 8 : i * 8 + 8] for i in range(nr_ranks)]

    worker(data, yv_expect, running_mean, running_var)


def test_batchnorm():
    nr_chan = 8
    data_shape = (3, nr_chan, 4)
    momentum = 0.9
    bn = BatchNorm1d(nr_chan, momentum=momentum)
    running_mean = np.zeros((1, nr_chan, 1), dtype=np.float32)
    running_var = np.ones((1, nr_chan, 1), dtype=np.float32)
    for i in range(3):
        xv = np.random.normal(loc=2.3, size=data_shape).astype(np.float32)
        mean = np.mean(np.mean(xv, axis=0, keepdims=True), axis=2, keepdims=True)
        xv_transposed = np.transpose(xv, [0, 2, 1]).reshape(
            (data_shape[0] * data_shape[2], nr_chan)
        )

        var_biased = np.var(xv_transposed, axis=0).reshape((1, nr_chan, 1))
        sd = np.sqrt(var_biased + bn.eps)

        var_unbiased = np.var(xv_transposed, axis=0, ddof=1).reshape((1, nr_chan, 1))
        running_mean = running_mean * momentum + mean * (1 - momentum)
        running_var = running_var * momentum + var_unbiased * (1 - momentum)

        yv = bn(Tensor(xv))
        yv_expect = (xv - mean) / sd

        _assert_allclose(yv.numpy(), yv_expect)
        _assert_allclose(bn.running_mean.numpy().reshape(-1), running_mean.reshape(-1))
        _assert_allclose(bn.running_var.numpy().reshape(-1), running_var.reshape(-1))

    # test set 'training' flag to False
    mean_backup = bn.running_mean.numpy()
    var_backup = bn.running_var.numpy()
    bn.training = False
    xv = np.random.normal(loc=2.3, size=data_shape).astype(np.float32)
    data = Tensor(xv)
    yv1 = bn(data)
    yv2 = bn(data)
    np.testing.assert_equal(yv1.numpy(), yv2.numpy())
    np.testing.assert_equal(mean_backup, bn.running_mean.numpy())
    np.testing.assert_equal(var_backup, bn.running_var.numpy())
    yv_expect = (xv - running_mean) / np.sqrt(running_var + bn.eps)
    _assert_allclose(yv1.numpy(), yv_expect)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
def test_syncbn1d():
    nr_chan = 8
    data_shape = (3, nr_chan, 4)
    momentum = 0.9
    bn = SyncBatchNorm(nr_chan, momentum=momentum)
    running_mean = np.zeros((1, nr_chan, 1), dtype=np.float32)
    running_var = np.ones((1, nr_chan, 1), dtype=np.float32)
    for i in range(3):
        xv = np.random.normal(loc=2.3, size=data_shape).astype(np.float32)
        mean = np.mean(np.mean(xv, axis=0, keepdims=True), axis=2, keepdims=True)
        xv_transposed = np.transpose(xv, [0, 2, 1]).reshape(
            (data_shape[0] * data_shape[2], nr_chan)
        )

        var_biased = np.var(xv_transposed, axis=0).reshape((1, nr_chan, 1))
        sd = np.sqrt(var_biased + bn.eps)

        var_unbiased = np.var(xv_transposed, axis=0, ddof=1).reshape((1, nr_chan, 1))
        running_mean = running_mean * momentum + mean * (1 - momentum)
        running_var = running_var * momentum + var_unbiased * (1 - momentum)

        yv = bn(Tensor(xv))
        yv_expect = (xv - mean) / sd

        _assert_allclose(yv.numpy(), yv_expect)
        _assert_allclose(bn.running_mean.numpy().reshape(-1), running_mean.reshape(-1))
        _assert_allclose(bn.running_var.numpy().reshape(-1), running_var.reshape(-1))

    # test set 'training' flag to False
    mean_backup = bn.running_mean.numpy()
    var_backup = bn.running_var.numpy()
    bn.training = False
    xv = np.random.normal(loc=2.3, size=data_shape).astype(np.float32)
    data = Tensor(xv)
    yv1 = bn(data)
    yv2 = bn(data)
    np.testing.assert_equal(yv1.numpy(), yv2.numpy())
    np.testing.assert_equal(mean_backup, bn.running_mean.numpy())
    np.testing.assert_equal(var_backup, bn.running_var.numpy())
    yv_expect = (xv - running_mean) / np.sqrt(running_var + bn.eps)
    _assert_allclose(yv1.numpy(), yv_expect)


def test_batchnorm2d():
    nr_chan = 8
    data_shape = (3, nr_chan, 16, 16)
    momentum = 0.9
    bn = BatchNorm2d(nr_chan, momentum=momentum)
    running_mean = np.zeros((1, nr_chan, 1, 1), dtype=np.float32)
    running_var = np.ones((1, nr_chan, 1, 1), dtype=np.float32)
    for i in range(3):
        xv = np.random.normal(loc=2.3, size=data_shape).astype(np.float32)
        xv_transposed = np.transpose(xv, [0, 2, 3, 1]).reshape(
            (data_shape[0] * data_shape[2] * data_shape[3], nr_chan)
        )

        mean = np.mean(xv_transposed, axis=0).reshape(1, nr_chan, 1, 1)

        var_biased = np.var(xv_transposed, axis=0).reshape((1, nr_chan, 1, 1))
        sd = np.sqrt(var_biased + bn.eps)

        var_unbiased = np.var(xv_transposed, axis=0, ddof=1).reshape((1, nr_chan, 1, 1))
        running_mean = running_mean * momentum + mean * (1 - momentum)
        running_var = running_var * momentum + var_unbiased * (1 - momentum)

        yv = bn(Tensor(xv))
        yv_expect = (xv - mean) / sd

        _assert_allclose(yv.numpy(), yv_expect)
        _assert_allclose(bn.running_mean.numpy(), running_mean)
        _assert_allclose(bn.running_var.numpy(), running_var)

    # test set 'training' flag to False
    mean_backup = bn.running_mean.numpy()
    var_backup = bn.running_var.numpy()
    bn.training = False
    xv = np.random.normal(loc=2.3, size=data_shape).astype(np.float32)
    data = Tensor(xv)
    yv1 = bn(data)
    yv2 = bn(data)
    np.testing.assert_equal(yv1.numpy(), yv2.numpy())
    np.testing.assert_equal(mean_backup, bn.running_mean.numpy())
    np.testing.assert_equal(var_backup, bn.running_var.numpy())
    yv_expect = (xv - running_mean) / np.sqrt(running_var + bn.eps)
    _assert_allclose(yv1.numpy(), yv_expect)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
def test_syncbn2d():
    nr_chan = 8
    data_shape = (3, nr_chan, 16, 16)
    momentum = 0.9
    bn = SyncBatchNorm(nr_chan, momentum=momentum)
    running_mean = np.zeros((1, nr_chan, 1, 1), dtype=np.float32)
    running_var = np.ones((1, nr_chan, 1, 1), dtype=np.float32)
    for i in range(3):
        xv = np.random.normal(loc=2.3, size=data_shape).astype(np.float32)
        xv_transposed = np.transpose(xv, [0, 2, 3, 1]).reshape(
            (data_shape[0] * data_shape[2] * data_shape[3], nr_chan)
        )

        mean = np.mean(xv_transposed, axis=0).reshape(1, nr_chan, 1, 1)

        var_biased = np.var(xv_transposed, axis=0).reshape((1, nr_chan, 1, 1))
        sd = np.sqrt(var_biased + bn.eps)

        var_unbiased = np.var(xv_transposed, axis=0, ddof=1).reshape((1, nr_chan, 1, 1))
        running_mean = running_mean * momentum + mean * (1 - momentum)
        running_var = running_var * momentum + var_unbiased * (1 - momentum)

        yv = bn(Tensor(xv))
        yv_expect = (xv - mean) / sd

        _assert_allclose(yv.numpy(), yv_expect)
        _assert_allclose(bn.running_mean.numpy(), running_mean)
        _assert_allclose(bn.running_var.numpy(), running_var)

    # test set 'training' flag to False
    mean_backup = bn.running_mean.numpy()
    var_backup = bn.running_var.numpy()
    bn.training = False
    xv = np.random.normal(loc=2.3, size=data_shape).astype(np.float32)
    data = Tensor(xv)
    yv1 = bn(data)
    yv2 = bn(data)
    np.testing.assert_equal(yv1.numpy(), yv2.numpy())
    np.testing.assert_equal(mean_backup, bn.running_mean.numpy())
    np.testing.assert_equal(var_backup, bn.running_var.numpy())
    yv_expect = (xv - running_mean) / np.sqrt(running_var + bn.eps)
    _assert_allclose(yv1.numpy(), yv_expect)


def test_batchnorm_no_stats():
    nr_chan = 8
    data_shape = (3, nr_chan, 4)
    bn = BatchNorm1d(8, track_running_stats=False)
    for i in range(4):
        if i == 2:
            bn.training = False
        xv = np.random.normal(loc=2.3, size=data_shape).astype(np.float32)
        mean = np.mean(np.mean(xv, axis=0, keepdims=True), axis=2, keepdims=True)
        var = np.var(
            np.transpose(xv, [0, 2, 1]).reshape(
                (data_shape[0] * data_shape[2], nr_chan)
            ),
            axis=0,
        ).reshape((1, nr_chan, 1))
        sd = np.sqrt(var + bn.eps)

        yv = bn(Tensor(xv))
        yv_expect = (xv - mean) / sd

        _assert_allclose(yv.numpy(), yv_expect)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
def test_syncbn_no_stats():
    nr_chan = 8
    data_shape = (3, nr_chan, 4)
    bn = SyncBatchNorm(8, track_running_stats=False)
    for i in range(4):
        if i == 2:
            bn.training = False
        xv = np.random.normal(loc=2.3, size=data_shape).astype(np.float32)
        mean = np.mean(np.mean(xv, axis=0, keepdims=True), axis=2, keepdims=True)
        var = np.var(
            np.transpose(xv, [0, 2, 1]).reshape(
                (data_shape[0] * data_shape[2], nr_chan)
            ),
            axis=0,
        ).reshape((1, nr_chan, 1))
        sd = np.sqrt(var + bn.eps)

        yv = bn(Tensor(xv))
        yv_expect = (xv - mean) / sd

        _assert_allclose(yv.numpy(), yv_expect)


def test_batchnorm2d_no_stats():
    nr_chan = 8
    data_shape = (3, nr_chan, 16, 16)
    bn = BatchNorm2d(8, track_running_stats=False)
    for i in range(4):
        if i == 2:
            bn.training = False
        xv = np.random.normal(loc=2.3, size=data_shape).astype(np.float32)
        xv_transposed = np.transpose(xv, [0, 2, 3, 1]).reshape(
            (data_shape[0] * data_shape[2] * data_shape[3], nr_chan)
        )

        mean = np.mean(xv_transposed, axis=0).reshape(1, nr_chan, 1, 1)
        var = np.var(xv_transposed, axis=0).reshape((1, nr_chan, 1, 1))
        sd = np.sqrt(var + bn.eps)

        yv = bn(Tensor(xv))
        yv_expect = (xv - mean) / sd

        _assert_allclose(yv.numpy(), yv_expect)


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
def test_syncbn2d_no_stats():
    nr_chan = 8
    data_shape = (3, nr_chan, 16, 16)
    bn = SyncBatchNorm(8, track_running_stats=False)
    for i in range(4):
        if i == 2:
            bn.training = False
        xv = np.random.normal(loc=2.3, size=data_shape).astype(np.float32)
        xv_transposed = np.transpose(xv, [0, 2, 3, 1]).reshape(
            (data_shape[0] * data_shape[2] * data_shape[3], nr_chan)
        )

        mean = np.mean(xv_transposed, axis=0).reshape(1, nr_chan, 1, 1)
        var = np.var(xv_transposed, axis=0).reshape((1, nr_chan, 1, 1))
        sd = np.sqrt(var + bn.eps)

        yv = bn(Tensor(xv))
        yv_expect = (xv - mean) / sd

        _assert_allclose(yv.numpy(), yv_expect)
