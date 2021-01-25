# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import itertools
import os

import numpy as np
import pytest

import megengine
import megengine.autodiff as ad
import megengine.optimizer as optimizer
from megengine import Parameter, tensor
from megengine.jit import trace
from megengine.module import Module


class Simple(Module):
    def __init__(self):
        super().__init__()
        self.a = Parameter([1.23], dtype="float32")

    def forward(self, x):
        x = x * self.a
        return x


@pytest.mark.parametrize("trace_mode", [True, False, None])
@pytest.mark.parametrize("inplace_mode", [True, False])
def test_sgd_momentum(monkeypatch, trace_mode, inplace_mode):
    with monkeypatch.context() as mk:
        mk.setenv("MEGENGINE_INPLACE_UPDATE", str(int(inplace_mode)))

        def train_func(data, *, model=None, optim=None, gm=None):
            optim.clear_grad()
            with gm:
                loss = net(data)
                gm.backward(loss)
            optim.step()
            return loss

        if trace_mode is not None:
            train_func = trace(symbolic=trace_mode)(train_func)

        def eval_func(data, *, model=None, optim=None, gm=None):
            loss = net(data)
            return loss

        if trace_mode is not None:
            eval_func = trace(symbolic=trace_mode)(eval_func)

        net = Simple()
        optim = optimizer.SGD(net.parameters(), lr=1.0, momentum=0.9)
        gm = ad.GradManager().attach(net.parameters())
        data = tensor([2.34])
        train_func(data, model=net, optim=optim, gm=gm)
        np.testing.assert_almost_equal(
            optim._state[net.a]["momentum_buffer"].numpy(), 2.34
        )

        # do 3 steps of infer
        for _ in range(3):
            loss = eval_func(data)
            np.testing.assert_almost_equal(loss.numpy(), 2.34 * (1.23 - 2.34), 5)
            np.testing.assert_almost_equal(
                optim._state[net.a]["momentum_buffer"].numpy(), 2.34
            )

        # do a step of train
        train_func(data, model=net, optim=optim, gm=gm)
        np.testing.assert_almost_equal(loss.numpy(), 2.34 * (1.23 - 2.34), 5)
        np.testing.assert_almost_equal(
            optim._state[net.a]["momentum_buffer"].numpy(), 0.9 * 2.34 + 2.34, 5
        )
