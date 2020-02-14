# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import time

import numpy as np
import pytest
from helpers import has_gpu

import megengine as mge
import megengine.functional as F
from megengine.optimizer import SGD


@pytest.mark.skip
@pytest.mark.slow
def test_release_memory():
    mnist_datasets = load_mnist_datasets()
    data_train, label_train = mnist_datasets["train"]

    batch_size = 15000
    data_shape = (batch_size, 1, 28, 28)
    label_shape = (batch_size,)

    data = nn.Input("data", shape=data_shape, dtype=np.float32)
    label = nn.Input(
        "label", shape=label_shape, dtype=np.int32, value=np.zeros(label_shape)
    )

    net = MnistNet()
    opt = SGD(net.parameters(), lr=0.01)

    pred = F.softmax(net(data))
    loss = F.cross_entropy(pred, label)

    opt.zero_grad()
    opt.backward(loss)
    add_updates = opt.step()

    mge.graph._default_graph.get_default().clear_device_memory()

    f = mge.graph.compile(loss, add_updates)

    for _ in range(3):
        train_loss = 0.0
        for i in range(0, data_train.shape[0], batch_size):
            opt.zero_grad()
            data = data_train[i : i + batch_size, :, :, :]
            label = label_train[i : i + batch_size]
            loss = f(data=data, label=label)[0]
            train_loss += loss[0]
