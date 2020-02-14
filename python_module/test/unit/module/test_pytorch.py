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
from megengine import get_default_device, set_default_device
from megengine.core import Parameter, tensor
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
