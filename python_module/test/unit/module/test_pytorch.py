# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np
import torch
from helpers import randomTorch

import megengine as mge
import megengine._internal as mgb
import megengine.functional
import megengine.optimizer as optimizer
from megengine import get_default_device, set_default_device
from megengine.core import Parameter, tensor
from megengine.jit import trace
from megengine.module import Module as MGEModule
from megengine.module.pytorch import PyTorchModule
from megengine.test import assertTensorClose


def test_pytorch_forward():
    class APlusB(torch.nn.Module):
        def __init__(self):
            super(APlusB, self).__init__()

        def forward(self, a, b):
            return a + b

    a = randomTorch(15, 15)
    b = randomTorch(15, 15)

    def get_pytorch_forward():
        return APlusB()(a, b)

    def get_mge_forward():
        mge_module = PyTorchModule(APlusB())
        mge_a = tensor(a.numpy(), dtype=np.float32)
        mge_b = tensor(b.numpy(), dtype=np.float32)
        return mge_module(mge_a, mge_b)

    assertTensorClose(get_pytorch_forward().numpy(), get_mge_forward().numpy())


def test_pytorch_backward():
    class APlusB(torch.nn.Module):
        def __init__(self):
            super(APlusB, self).__init__()

        def forward(self, a, b):
            return a + b

    a = randomTorch(15, 15)
    b = randomTorch(15, 15)

    def get_pytorch_backward():
        parameter_a = a.clone()
        parameter_a.requires_grad = True
        c = APlusB()(parameter_a, b)
        d = APlusB()(c, b)
        e = torch.sum(d)
        e.backward()
        return parameter_a.grad

    def get_mge_backward():
        mge_module = PyTorchModule(APlusB())
        mge_a = Parameter(a.numpy(), dtype=np.float32)
        mge_b = tensor(b.numpy(), dtype=np.float32)
        mge_c = mge_module(mge_a, mge_b)
        mge_d = mge_module(mge_c, mge_b)
        mge_e = mge.functional.sum(mge_d)
        return mge.functional.grad(mge_e, mge_a, use_virtual_grad=False)

    assertTensorClose(get_pytorch_backward().numpy(), get_mge_backward().numpy())


def test_pytorch_mixed():

    init_param = (np.array([2.0], dtype=np.float32), np.array([3.0], dtype=np.float32))
    lr = 1.0

    class Mixed(MGEModule):
        class SubModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.multiplier = torch.nn.Parameter(torch.tensor(init_param[0]))

            def forward(self, inp):
                return inp * self.multiplier

        def __init__(self):
            super().__init__()
            self.torch_module = PyTorchModule(self.SubModule())
            self.multiplier = Parameter(init_param[1], dtype=np.float32)

        def forward(self, inp):
            return self.torch_module(inp) * self.multiplier

    def run(step, enable_trace, use_symbolic):
        def train_func(data, net=None, opt=None):
            pred = net(data)
            opt.backward(pred)
            return pred

        if enable_trace:
            train_func = trace(train_func, symbolic=use_symbolic)

        net = Mixed()
        data = tensor()
        opt = optimizer.SGD(net.parameters(), lr=lr)

        saved_param = init_param
        for i in range(step):
            opt.zero_grad()
            data.set_value([i + 1.0])
            output = train_func(data, net=net, opt=opt)
            opt.step()

            expect_param = (
                saved_param[0] - lr * saved_param[1] * data.numpy(),
                saved_param[1] - lr * saved_param[0] * data.numpy(),
            )
            assertTensorClose(
                output.numpy(), saved_param[0] * saved_param[1] * data.numpy()
            )
            torch_param = net.torch_module._torch_params[0].detach().cpu()
            assertTensorClose(torch_param.numpy(), expect_param[0])
            assertTensorClose(net.multiplier.numpy(), expect_param[1])
            saved_param = expect_param

    run(1, False, False)
    run(1, True, True)
    run(1, True, False)

    run(2, False, False)
    run(2, True, True)
    run(2, True, False)
