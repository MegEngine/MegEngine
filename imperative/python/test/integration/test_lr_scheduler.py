# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from bisect import bisect_right

import numpy as np

from megengine import Parameter, tensor
from megengine.module import Module
from megengine.optimizer import SGD, MultiStepLR, CosineAnnealingLR


class Simple(Module):
    def __init__(self):
        super().__init__()
        self.a = Parameter([1.23], dtype=np.float32)

    def forward(self, x):
        x = x * self.a
        return x


def test_multi_step_lr():
    net = Simple()
    opt = SGD(net.parameters(), lr=0.01, momentum=0.9)
    scheduler = MultiStepLR(opt, [3, 6, 8])

    lr = np.array(0.01, dtype=np.float32)
    for i in range(10):
        for group in opt.param_groups:
            np.testing.assert_almost_equal(
                np.array(group["lr"], dtype=np.float32),
                (lr * 0.1 ** bisect_right([3, 6, 8], i)).astype(np.float32),
            )
        scheduler.step()

def test_cosine_annealing_lr():
    epochs = 10
    eta_min = 10e-6

    net = Simple()
    opt = SGD(net.parameters(), lr=0.01, momentum=0.9)
    scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=eta_min)

    # closed form
    def get_closed_form(epoch):
        return eta_min + (0.01 - eta_min) * (1 + 
            math.cos(math.pi * epoch / epochs)) / 2

    for i in range(epochs):
        for group in opt.param_groups:
            np.testing.assert_almost_equal(
                np.array(group["lr"], dtype=np.float32),
                get_closed_form(i),
            )
        scheduler.step()