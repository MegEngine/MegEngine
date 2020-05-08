# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import os
import re
import subprocess
import sys

import numpy as np

import megengine as mge
import megengine.functional as F
from megengine import jit, tensor
from megengine.functional.debug_param import set_conv_execution_strategy
from megengine.jit import SublinearMemoryConfig
from megengine.module import AvgPool2d, BatchNorm2d, Conv2d, Linear, Module
from megengine.optimizer import SGD
from megengine.test import assertTensorClose


def get_gpu_name():
    try:
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"]
        )
        gpu_info = gpu_info.decode("ascii").split("\n")[0]
    except:
        gpu_info = "None"
    return gpu_info


def get_cpu_name():
    cpu_info = "None"
    try:
        cpu_info = subprocess.check_output(["cat", "/proc/cpuinfo"]).decode("ascii")
        for line in cpu_info.split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, 1).strip()
    except:
        pass
    return cpu_info


def get_xpu_name():
    if mge.is_cuda_available():
        return get_gpu_name()
    else:
        return get_cpu_name()


class MnistNet(Module):
    def __init__(self, has_bn=False):
        super().__init__()
        self.conv0 = Conv2d(1, 20, kernel_size=5, bias=True)
        self.pool0 = AvgPool2d(2)
        self.conv1 = Conv2d(20, 20, kernel_size=5, bias=True)
        self.pool1 = AvgPool2d(2)
        self.fc0 = Linear(20 * 4 * 4, 500, bias=True)
        self.fc1 = Linear(500, 10, bias=True)
        self.bn0 = None
        self.bn1 = None
        if has_bn:
            self.bn0 = BatchNorm2d(20)
            self.bn1 = BatchNorm2d(20)

    def forward(self, x):
        x = self.conv0(x)
        if self.bn0:
            x = self.bn0(x)
        x = F.relu(x)
        x = self.pool0(x)
        x = self.conv1(x)
        if self.bn1:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = F.flatten(x, 1)
        x = self.fc0(x)
        x = F.relu(x)
        x = self.fc1(x)
        return x


def train(data, label, net, opt):

    pred = net(data)
    loss = F.cross_entropy_with_softmax(pred, label)
    opt.backward(loss)
    return loss


def update_model(model_path):
    """
    Update the dumped model with test cases for new reference values.

    The model with pre-trained weights is trained for one iter with the test data attached.
    The loss and updated net state dict is dumped.

    .. code-block:: python

        from test_correctness import update_model
        update_model('mnist_model_with_test.mge') # for gpu
        update_model('mnist_model_with_test_cpu.mge') # for cpu

    """
    net = MnistNet(has_bn=True)
    checkpoint = mge.load(model_path)
    net.load_state_dict(checkpoint["net_init"])
    lr = checkpoint["sgd_lr"]
    opt = SGD(net.parameters(), lr=lr)

    data = tensor(dtype=np.float32)
    label = tensor(dtype=np.int32)
    data.set_value(checkpoint["data"])
    label.set_value(checkpoint["label"])

    opt.zero_grad()
    loss = train(data, label, net=net, opt=opt)
    opt.step()

    xpu_name = get_xpu_name()

    checkpoint.update(
        {"net_updated": net.state_dict(), "loss": loss.numpy(), "xpu": xpu_name}
    )
    mge.save(checkpoint, model_path)


def run_test(
    model_path, use_jit, use_symbolic, sublinear_memory_config=None, max_err=None,
):

    """
    Load the model with test cases and run the training for one iter.
    The loss and updated weights are compared with reference value to verify the correctness.

    Dump a new file with updated result by calling update_model
    if you think the test fails due to numerical rounding errors instead of bugs.
    Please think twice before you do so.

    """
    net = MnistNet(has_bn=True)
    checkpoint = mge.load(model_path)
    net.load_state_dict(checkpoint["net_init"])
    lr = checkpoint["sgd_lr"]
    opt = SGD(net.parameters(), lr=lr)

    data = tensor(dtype=np.float32)
    label = tensor(dtype=np.int32)
    data.set_value(checkpoint["data"])
    label.set_value(checkpoint["label"])

    if max_err is None:
        max_err = 1e-5

    train_func = train
    if use_jit:
        train_func = jit.trace(
            train_func,
            symbolic=use_symbolic,
            sublinear_memory_config=sublinear_memory_config,
        )

    opt.zero_grad()
    loss = train_func(data, label, net=net, opt=opt)
    opt.step()

    assertTensorClose(loss.numpy(), checkpoint["loss"], max_err=max_err)

    for param, param_ref in zip(
        net.state_dict().items(), checkpoint["net_updated"].items()
    ):
        assert param[0] == param_ref[0]
        assertTensorClose(param[1], param_ref[1], max_err=max_err)


def test_correctness():

    if mge.is_cuda_available():
        model_name = "mnist_model_with_test.mge"
    else:
        model_name = "mnist_model_with_test_cpu.mge"
    model_path = os.path.join(os.path.dirname(__file__), model_name)
    set_conv_execution_strategy("HEURISTIC_REPRODUCIBLE")

    run_test(model_path, False, False)
    run_test(model_path, True, False)
    run_test(model_path, True, True)

    # sublinear
    config = SublinearMemoryConfig(genetic_nr_iter=10)
    run_test(
        model_path, True, True, sublinear_memory_config=config, max_err=1e-5,
    )
