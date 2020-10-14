# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import multiprocessing as mp
import os
import platform
import re
import subprocess
import sys
from math import ceil

import numpy as np
import pytest

import megengine as mge
import megengine.autodiff as ad
import megengine.distributed as dist
import megengine.functional as F
from megengine.device import get_default_device, set_default_device
from megengine.distributed.helper import get_device_count_by_fork
from megengine.functional.debug_param import set_conv_execution_strategy
from megengine.module import AvgPool2d, BatchNorm2d, Conv2d, Linear, Module
from megengine.optimizer import SGD
from megengine.tensor import Tensor

p_num = 4


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
    def __init__(self, has_bn=True):
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


def train(data, label, net, opt, gm):
    opt.clear_grad()
    with gm:
        pred = net(data)
        loss = F.nn.cross_entropy(pred, label)
        gm.backward(loss)
    opt.step()
    return loss


def update_model(model_path):
    """
    Update the dumped model with test cases for new reference values.

    The model with pre-trained weights is trained for one iter with the test data attached.
    The loss and updated net state dict is dumped.

    .. code-block:: python

        from test_dp_correctness import update_model
        update_model('mnist_model_with_test.mge') # for gpu
        update_model('mnist_model_with_test_cpu.mge') # for cpu

    """
    net = MnistNet(has_bn=True)
    checkpoint = mge.load(model_path)
    net.load_state_dict(checkpoint["net_init"])
    lr = checkpoint["sgd_lr"]
    opt = SGD(net.parameters(), lr=lr)

    gm = ad.GradManager().attach(
        net.parameters(), callbacks=[dist.make_allreduce_cb("MEAN", dist.WORLD)]
    )

    data = Tensor(checkpoint["data"], dtype=np.float32)
    label = Tensor(checkpoint["label"], dtype=np.int32)

    opt.clear_grad()
    loss = train(data, label, net=net, opt=opt)
    opt.step()

    xpu_name = get_xpu_name()

    checkpoint.update(
        {"net_updated": net.state_dict(), "loss": loss.numpy(), "xpu": xpu_name}
    )
    mge.serialization.save(checkpoint, model_path)


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
    checkpoint = mge.load(model_path)
    data = checkpoint["data"]
    label = checkpoint["label"]
    port = dist.get_free_ports(1)[0]
    server = dist.Server(port)

    def worker(rank, max_err):
        dist.init_process_group("localhost", port, p_num, rank, rank)
        net = MnistNet(has_bn=True)
        net.load_state_dict(checkpoint["net_init"])
        lr = checkpoint["sgd_lr"]
        opt = SGD(net.parameters(), lr=lr)

        gm = ad.GradManager().attach(
            net.parameters(), callbacks=[dist.make_allreduce_cb("MEAN", dist.WORLD)]
        )

        # use same data and label for all gpu's
        # such that the result does not depend on number of gpu
        data_train = Tensor(data)
        label_train = Tensor(label)

        loss = train(data_train, label_train, net, opt, gm)

        np.testing.assert_allclose(loss.numpy(), checkpoint["loss"], atol=max_err)

        if dist.get_rank():
            return
        for param, param_ref in zip(
            net.state_dict().items(), checkpoint["net_updated"].items()
        ):
            assert param[0] == param_ref[0]
            np.testing.assert_allclose(param[1], param_ref[1], atol=max_err)

    procs = []
    for rank in range(p_num):
        p = mp.Process(target=worker, args=(rank, max_err,))
        p.start()
        procs.append(p)

    for p in procs:
        p.join(20)
        assert p.exitcode == 0


@pytest.mark.skipif(get_device_count_by_fork("gpu") < 4, reason="need more gpu device")
@pytest.mark.isolated_distributed
@pytest.mark.skipif(
    platform.system() == "Windows", reason="windows disable MGB_ENABLE_OPR_MM"
)
def test_dp_correctness():
    model_name = "mnist_model_with_test.mge"
    model_path = os.path.join(os.path.dirname(__file__), model_name)
    set_conv_execution_strategy("HEURISTIC_REPRODUCIBLE")
    run_test(model_path, False, False, max_err=1e-5)
