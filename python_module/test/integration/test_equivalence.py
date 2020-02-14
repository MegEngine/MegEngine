# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import copy
import itertools
import os
from typing import Callable

import numpy as np
import pytest

import megengine as mge
import megengine.module.init as init
from megengine.core import tensor
from megengine.functional import cross_entropy_with_softmax, relu
from megengine.jit import trace
from megengine.module import Linear, Module
from megengine.optimizer import SGD, Optimizer
from megengine.test import assertTensorClose

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


class SimpleNet(Module):
    def __init__(self):
        self.mid_layers = 14
        self.num_class = 2
        super().__init__()

        self.fc0 = Linear(self.num_class, self.mid_layers, bias=True)
        fan_in, _ = init.calculate_fan_in_and_fan_out(self.fc0.weight)
        init.normal_(self.fc0.weight, std=np.sqrt(float(1.0) / fan_in))
        init.zeros_(self.fc0.bias)

        self.fc1 = Linear(self.mid_layers, self.mid_layers, bias=True)
        fan_in, _ = init.calculate_fan_in_and_fan_out(self.fc1.weight)
        init.normal_(self.fc1.weight, std=np.sqrt(float(1.0) / fan_in))
        init.zeros_(self.fc1.bias)

        self.fc2 = Linear(self.mid_layers, self.num_class, bias=True)
        fan_in, _ = init.calculate_fan_in_and_fan_out(self.fc2.weight)
        init.normal_(self.fc2.weight, std=np.sqrt(float(1.0) / fan_in))
        init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc0(x)
        x = relu(x)  # Should use tanh but it's not stable now.
        x = self.fc1(x)
        x = relu(x)  # Should use tanh but it's not stable now.
        x = self.fc2(x)
        return x


def generate_eager_step(net: Module, opt_factory: Callable[[Module], Optimizer]):
    data_inp = tensor(np.zeros(data_shape), dtype=np.float32)
    label_inp = tensor(np.zeros(label_shape), dtype=np.int32)
    opt = opt_factory(net)

    def step(data, label):
        opt.zero_grad()
        data_inp.set_value(data)
        label_inp.set_value(label)

        pred = net(data_inp)
        loss = cross_entropy_with_softmax(pred, label_inp)
        opt.backward(loss)
        opt.step()

        return loss.numpy()[0]

    return step


def generate_static_step(net: Module, opt_factory: Callable[[Module], Optimizer]):
    data = tensor(np.zeros(data_shape), dtype=np.float32)
    label = tensor(np.zeros(label_shape), dtype=np.int32)
    opt = opt_factory(net)

    # Save state to reset parameters later.
    state = copy.deepcopy(net.state_dict())

    # Evaluate network in eager mode once.
    pred = net(data)
    loss = cross_entropy_with_softmax(pred, label)
    opt.zero_grad()
    grads = opt.backward(loss)

    f = mge.graph.compile(loss, grads)

    def step(data, label):
        opt.zero_grad()
        out = f(data=data, label=label)
        opt.step()
        loss = out[0][0]
        return loss

    # Reset parameters.
    net.load_state_dict(state)
    return step


def generate_trace_step(
    net: Module, opt_factory: Callable[[Module], Optimizer], enable: bool
):
    opt = opt_factory(net)

    @trace
    def train(data, label):
        pred = net(data)
        loss = cross_entropy_with_softmax(pred, label)
        opt.zero_grad()
        opt.backward(loss)
        return loss

    train.enabled = enable

    def step(data, label):
        out = train(data, label)
        opt.step()
        loss = out[0][0]
        return loss

    return step


def assert_network_equvilence(nets):
    net_state = [net.state_dict() for net in nets]

    for state in net_state[1:]:
        assert len(net_state[0]) == len(state)

    for k, v in net_state[0].items():
        for state in net_state[1:]:
            assert k in state
            assertTensorClose(v, state[k])


@pytest.mark.slow
def test_eager_equvilence():
    eager_net = SimpleNet()
    trace_enable_net = copy.deepcopy(eager_net)
    trace_disable_net = copy.deepcopy(eager_net)

    opt_factory = lambda net: SGD(
        net.parameters(requires_grad=True), lr=0.01, momentum=0.01
    )

    estep = generate_eager_step(eager_net, opt_factory)
    te_step = generate_trace_step(trace_enable_net, opt_factory, True)
    td_step = generate_trace_step(trace_disable_net, opt_factory, False)

    assert_network_equvilence([eager_net, trace_enable_net, trace_disable_net])

    # Use hard code number as limit, may increase if needed.
    for data, label in itertools.islice(minibatch_generator(), 200):
        eloss = estep(data, label)
        te_loss = te_step(data, label)
        td_loss = td_step(data, label)

        assertTensorClose(eloss, te_loss)
        assertTensorClose(eloss, td_loss)
        assert_network_equvilence(
            [eager_net, trace_enable_net, trace_disable_net,]
        )
