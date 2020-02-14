# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import itertools

import numpy as np
import pytest

import megengine as mge
from megengine.core import tensor
from megengine.functional import cross_entropy_with_softmax, tanh
from megengine.jit import trace
from megengine.module import Linear, Module
from megengine.optimizer import SGD

batch_size = 64
data_shape = (batch_size, 2)
label_shape = (batch_size,)


def minibatch_generator():
    while True:
        inp_data = np.zeros((batch_size, 2))
        label = np.zeros(batch_size, dtype=np.int32)
        for i in range(batch_size):
            # [x0, x1], sampled from U[-1, 1]
            inp_data[i, :] = np.random.rand(2) * 2 - 1
            label[i] = 0 if np.prod(inp_data[i]) < 0 else 1
        yield inp_data.astype(np.float32), label.astype(np.int32)


def calculate_precision(data: np.ndarray, pred: np.ndarray) -> float:
    """ Calculate precision for given data and prediction.

    :type data: [[x, y], ...]
    :param data: Input data
    :type pred: [[x_pred, y_pred], ...]
    :param pred: Network output data
    """
    correct = 0
    assert len(data) == len(pred)
    for inp_data, pred_output in zip(data, pred):
        label = 0 if np.prod(inp_data) < 0 else 1
        pred_label = np.argmax(pred_output)
        if pred_label == label:
            correct += 1
    return float(correct) / len(data)


class XORNet(Module):
    def __init__(self):
        self.mid_layers = 14
        self.num_class = 2
        super().__init__()

        self.fc0 = Linear(self.num_class, self.mid_layers, bias=True)
        self.fc1 = Linear(self.mid_layers, self.mid_layers, bias=True)

        self.fc2 = Linear(self.mid_layers, self.num_class, bias=True)

    def forward(self, x):
        x = self.fc0(x)
        x = tanh(x)
        x = self.fc1(x)
        x = tanh(x)
        x = self.fc2(x)
        return x


@pytest.mark.slow
def test_training_converge():
    net = XORNet()
    opt = SGD(
        net.parameters(requires_grad=True), lr=0.01, momentum=0.9, weight_decay=5e-4
    )

    @trace
    def train(data, label):
        pred = net(data)
        opt.zero_grad()
        loss = cross_entropy_with_softmax(pred, label)
        opt.backward(loss)
        return loss

    @trace
    def infer(data):
        return net(data)

    train_dataset = minibatch_generator()
    losses = []

    for data, label in itertools.islice(train_dataset, 2000):
        # opt.zero_grad()
        loss = train(data, label)
        loss = loss[0][0]
        opt.step()
        losses.append(loss.numpy())

    assert np.mean(losses[-100:]) < 0.1, "Final training Loss must be low enough"

    data, _ = next(train_dataset)
    pred = infer(data).numpy()
    assert calculate_precision(data, pred) > 0.95, "Test precision must be high enough"
