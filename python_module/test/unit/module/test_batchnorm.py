# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np
import pytest

import megengine as mge
from megengine.core import tensor
from megengine.module import BatchNorm1d, BatchNorm2d
from megengine.test import assertTensorClose


def test_batchnorm():
    nr_chan = 8
    data_shape = (3, nr_chan, 4)
    momentum = 0.9
    bn = BatchNorm1d(nr_chan, momentum=momentum)
    running_mean = np.zeros((1, nr_chan, 1), dtype=np.float32)
    running_var = np.ones((1, nr_chan, 1), dtype=np.float32)
    data = tensor()
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

        data.set_value(xv)
        yv = bn(data)
        yv_expect = (xv - mean) / sd

        assertTensorClose(yv_expect, yv.numpy(), max_err=5e-6)
        assertTensorClose(
            running_mean.reshape(-1), bn.running_mean.numpy().reshape(-1), max_err=5e-6
        )
        assertTensorClose(
            running_var.reshape(-1), bn.running_var.numpy().reshape(-1), max_err=5e-6
        )

    # test set 'training' flag to False
    mean_backup = bn.running_mean.numpy()
    var_backup = bn.running_var.numpy()
    bn.training = False
    xv = np.random.normal(loc=2.3, size=data_shape).astype(np.float32)
    data.set_value(xv)
    yv1 = bn(data)
    yv2 = bn(data)
    assertTensorClose(yv1.numpy(), yv2.numpy(), max_err=0)
    assertTensorClose(mean_backup, bn.running_mean.numpy(), max_err=0)
    assertTensorClose(var_backup, bn.running_var.numpy(), max_err=0)
    yv_expect = (xv - running_mean) / np.sqrt(running_var + bn.eps)
    assertTensorClose(yv_expect, yv1.numpy(), max_err=5e-6)


def test_batchnorm2d():
    nr_chan = 8
    data_shape = (3, nr_chan, 16, 16)
    momentum = 0.9
    bn = BatchNorm2d(nr_chan, momentum=momentum)
    running_mean = np.zeros((1, nr_chan, 1, 1), dtype=np.float32)
    running_var = np.ones((1, nr_chan, 1, 1), dtype=np.float32)
    data = tensor()
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

        data.set_value(xv)
        yv = bn(data)
        yv_expect = (xv - mean) / sd

        assertTensorClose(yv_expect, yv.numpy(), max_err=5e-6)
        assertTensorClose(running_mean, bn.running_mean.numpy(), max_err=5e-6)
        assertTensorClose(running_var, bn.running_var.numpy(), max_err=5e-6)

    # test set 'training' flag to False
    mean_backup = bn.running_mean.numpy()
    var_backup = bn.running_var.numpy()
    bn.training = False
    xv = np.random.normal(loc=2.3, size=data_shape).astype(np.float32)
    data.set_value(xv)
    yv1 = bn(data)
    yv2 = bn(data)
    assertTensorClose(yv1.numpy(), yv2.numpy(), max_err=0)
    assertTensorClose(mean_backup, bn.running_mean.numpy(), max_err=0)
    assertTensorClose(var_backup, bn.running_var.numpy(), max_err=0)
    yv_expect = (xv - running_mean) / np.sqrt(running_var + bn.eps)
    assertTensorClose(yv_expect, yv1.numpy(), max_err=5e-6)


def test_batchnorm_no_stats():
    nr_chan = 8
    data_shape = (3, nr_chan, 4)
    bn = BatchNorm1d(8, track_running_stats=False)
    data = tensor()
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

        data.set_value(xv)
        yv = bn(data)
        yv_expect = (xv - mean) / sd

        assertTensorClose(yv_expect, yv.numpy(), max_err=5e-6)


def test_batchnorm2d_no_stats():
    nr_chan = 8
    data_shape = (3, nr_chan, 16, 16)
    bn = BatchNorm2d(8, track_running_stats=False)
    data = tensor()
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

        data.set_value(xv)
        yv = bn(data)
        yv_expect = (xv - mean) / sd

        assertTensorClose(yv_expect, yv.numpy(), max_err=5e-6)
