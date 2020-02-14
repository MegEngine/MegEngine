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
from helpers import MLP

import megengine._internal as mgb
import megengine.functional as F
from megengine.core import Graph
from megengine.module import Linear, Module
from megengine.optimizer import SGD
from megengine.test import assertTensorClose


def test_compile_multi_times_eager():
    return  # XXX: rewrite or remove this test
    data = Input("data", shape=(2, 28))
    label = Input("label", shape=(2,), dtype=np.int32)

    mlp = MLP()
    opt = SGD(mlp.parameters(requires_grad=True), lr=0.01)

    pred0 = mlp(data)
    pred = F.softmax(pred0)
    loss = F.square_loss(pred, label.reshape(2, 1))
    opt.zero_grad()
    grads = opt.backward(loss)
    opt.step()

    f0 = compile(pred, None)
    f1 = compile([pred, loss], grads, copy=False)
    for _ in range(3):
        data = np.random.random((2, 28)).astype(np.float32)
        label = np.random.randint(0, 10, (2,)).astype(np.float32)
        out0 = f0(data=data)
        out1 = f1(data=data, label=label)
        assertTensorClose(out0[0], out1[0])


def test_compile_multi_times_static():
    return  # XXX: rewrite or remove this test
    with Graph() as cg:
        cg.set_option("eager_evaluation", False)
        data = Input("data", shape=(2, 28))
        label = Input("label", shape=(2,), dtype=np.int32)

        mlp = MLP()
        opt = SGD(mlp.parameters(requires_grad=True), lr=0.01)

        pred0 = mlp(data)
        pred = F.softmax(pred0)
        loss = F.square_loss(pred, label.reshape(2, 1))
        opt.zero_grad()
        grads = opt.backward(loss)
        opt.step()

        f0 = compile(pred, None)
        f1 = compile([pred, loss], grads, copy=True)

        data = np.random.random((2, 28)).astype(np.float32)
        label = np.random.randint(0, 10, (2,)).astype(np.float32)
        out0 = f0(data=data)
        out1 = f1(data=data, label=label)
        assertTensorClose(out0[0], out1[0])

        _ = compile([pred, loss], grads, copy=False)
        with pytest.raises(mgb.MegBrainError):
            f0(data=data)
