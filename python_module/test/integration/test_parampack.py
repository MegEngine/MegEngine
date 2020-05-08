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
from megengine.module import Linear, Module, ParamPack
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
def test_static_graph_parampack():
    net = XORNet()
    net = ParamPack(
        net, nr_ignore_first=0, max_size_per_group=10, max_nr_params_per_group=100
    )
    opt = SGD(
        net.parameters(requires_grad=True), lr=0.01, momentum=0.9, weight_decay=5e-4
    )

    @trace(symbolic=True)
    def train(data, label):
        pred = net(data)
        opt.zero_grad()
        loss = cross_entropy_with_softmax(pred, label)
        opt.backward(loss)
        return loss

    @trace(symbolic=True)
    def infer(data):
        return net(data)

    train_dataset = minibatch_generator()
    losses = []

    for data, label in itertools.islice(train_dataset, 2000):
        loss = train(data, label)
        loss = loss[0][0]
        opt.step()
        losses.append(loss.numpy())

    assert np.mean(losses[-100:]) < 0.1, "Final training Loss must be low enough"

    ngrid = 10
    x = np.linspace(-1.0, 1.0, ngrid)
    xx, yy = np.meshgrid(x, x)
    xx = xx.reshape((ngrid * ngrid, 1))
    yy = yy.reshape((ngrid * ngrid, 1))
    data = np.concatenate((xx, yy), axis=1).astype(np.float32)

    pred = infer(data).numpy()
    assert calculate_precision(data, pred) == 1.0, "Test precision must be high enough"


@pytest.mark.slow
def test_nopack_parampack():
    net = XORNet()
    net = ParamPack(net, max_size_per_group=0, max_nr_params_per_group=0)
    opt = SGD(
        net.parameters(requires_grad=True), lr=0.01, momentum=0.9, weight_decay=5e-4
    )

    @trace(symbolic=True)
    def train(data, label):
        pred = net(data)
        opt.zero_grad()
        loss = cross_entropy_with_softmax(pred, label)
        opt.backward(loss)
        return loss

    @trace(symbolic=True)
    def infer(data):
        return net(data)

    train_dataset = minibatch_generator()
    losses = []

    for data, label in itertools.islice(train_dataset, 2000):
        loss = train(data, label)
        loss = loss[0][0]
        opt.step()
        losses.append(loss.numpy())
    assert np.mean(losses[-100:]) < 0.1, "Final training Loss must be low enough"

    ngrid = 10
    x = np.linspace(-1.0, 1.0, ngrid)
    xx, yy = np.meshgrid(x, x)
    xx = xx.reshape((ngrid * ngrid, 1))
    yy = yy.reshape((ngrid * ngrid, 1))
    data = np.concatenate((xx, yy), axis=1).astype(np.float32)

    pred = infer(data).numpy()
    assert calculate_precision(data, pred) == 1.0, "Test precision must be high enough"


@pytest.mark.slow
def test_dynamic_graph_parampack():
    net = XORNet()
    net = ParamPack(
        net, nr_ignore_first=0, max_size_per_group=10, max_nr_params_per_group=100
    )
    opt = SGD(
        net.parameters(requires_grad=True), lr=0.01, momentum=0.9, weight_decay=5e-4
    )

    @trace(symbolic=False)
    def train(data, label):
        pred = net(data)
        opt.zero_grad()
        loss = cross_entropy_with_softmax(pred, label)
        opt.backward(loss)
        return loss

    @trace(symbolic=False)
    def infer(data):
        return net(data)

    train_dataset = minibatch_generator()
    losses = []

    for data, label in itertools.islice(train_dataset, 2000):
        loss = train(data, label)
        loss = loss[0][0]
        opt.step()
        losses.append(loss.numpy())

    assert np.mean(losses[-100:]) < 0.1, "Final training Loss must be low enough"

    ngrid = 10
    x = np.linspace(-1.0, 1.0, ngrid)
    xx, yy = np.meshgrid(x, x)
    xx = xx.reshape((ngrid * ngrid, 1))
    yy = yy.reshape((ngrid * ngrid, 1))
    data = np.concatenate((xx, yy), axis=1).astype(np.float32)

    pred = infer(data).numpy()
    assert calculate_precision(data, pred) == 1.0, "Test precision must be high enough"


@pytest.mark.slow
def test_correctness_parampack():
    net1 = XORNet()
    net2 = XORNet()
    params1 = net1.parameters()
    params2 = net2.parameters()
    for param1, param2 in zip(params1, params2):
        param1.set_value(param2.numpy())
    net1 = ParamPack(
        net1, nr_ignore_first=0, max_size_per_group=10, max_nr_params_per_group=100
    )
    opt1 = SGD(
        net1.parameters(requires_grad=True), lr=0.01, momentum=0.9, weight_decay=5e-4
    )

    opt2 = SGD(
        net2.parameters(requires_grad=True), lr=0.01, momentum=0.9, weight_decay=5e-4
    )

    @trace(symbolic=False)
    def train1(data, label):
        pred = net1(data)
        opt1.zero_grad()
        loss = cross_entropy_with_softmax(pred, label)
        opt1.backward(loss)
        return loss

    @trace(symbolic=False)
    def train2(data, label):
        pred = net2(data)
        opt2.zero_grad()
        loss = cross_entropy_with_softmax(pred, label)
        opt2.backward(loss)
        return loss

    @trace(symbolic=False)
    def infer1(data):
        return net1(data)

    @trace(symbolic=False)
    def infer2(data):
        return net2(data)

    train_dataset = minibatch_generator()

    for data, label in itertools.islice(train_dataset, 2000):
        train1(data, label)
        opt1.step()

        train2(data, label)
        opt2.step()

    data, _ = next(train_dataset)
    pred1 = infer1(data).numpy()
    pred2 = infer2(data).numpy()
    assert np.allclose(pred1, pred2)


def test_parampack_group_func():
    net = XORNet()
    net = ParamPack(
        net,
        nr_ignore_first=1,
        max_size_per_group=10,
        max_nr_params_per_group=100,
        group_func=lambda n, p: "weight" in n,
    )
    for p in net.parameters(requires_grad=True):
        assert p.pack_group_key is not None
    for n, p in net.named_parameters(requires_grad=True):
        assert p.pack_group_key is not None
