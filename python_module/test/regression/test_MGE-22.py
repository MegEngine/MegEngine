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

from megengine.core import tensor
from megengine.module import Linear, Module
from megengine.optimizer import SGD


class Blur(Module):
    def __init__(self, dim1=16, dim2=128, dim3=1):
        super().__init__()

        self.fc1 = Linear(dim1, dim2)
        self.fc2 = Linear(dim2, dim3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x.mean(axis=1, keepdims=True)


@pytest.mark.regression
def test_blur():
    net = Blur()
    data = tensor(np.random.random((32, 16)).astype("float32"))

    opt = SGD(net.parameters(requires_grad=True), lr=0.1)
    opt.zero_grad()

    loss = net(data)
    opt.backward(loss.sum())
