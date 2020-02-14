# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from io import BytesIO

import numpy as np
from helpers import MLP, graph_mode

import megengine.functional as F
from megengine import load, save
from megengine.core import TensorDict, tensor
from megengine.jit import trace
from megengine.optimizer import SGD, Adam
from megengine.test import assertTensorClose


def get_input():
    batch_size = 2
    input_dim = 28
    data_shape = (batch_size, input_dim)
    label_shape = (batch_size,)
    data = tensor()
    label = tensor(dtype=np.int32)
    data.set_value(np.random.random(data_shape).astype(np.float32))
    label.set_value(np.random.randint(0, 10, label_shape))
    return data, data_shape, label, label_shape


def test_sgd_simple():
    data, data_shape, label, label_shape = get_input()
    mlp = MLP()
    opt = SGD(mlp.parameters(), lr=0.01, weight_decay=0.1)
    for idx in range(3):
        data.set_value(np.random.random(data_shape).astype(np.float32))
        label.set_value(np.random.randint(0, 10, label_shape))
        pred = mlp(data)
        loss = F.square_loss(pred, label.reshape(-1, 1))
        if idx % 2:
            opt.zero_grad()
        else:
            mlp.zero_grad()
        opt.backward(loss)
        grads = TensorDict()
        orig_params = TensorDict()
        for param in mlp.parameters():
            grad = F.grad(loss, param, use_virtual_grad=False)
            assertTensorClose(grad.numpy(), param.grad.numpy())
            grads[param] = np.copy(grad.numpy())
            orig_params[param] = np.copy(param.numpy())
        opt.step()
        for param in mlp.parameters():
            assertTensorClose(
                param.numpy(), orig_params[param] * 0.999 - grads[param] * 0.01
            )


def test_sgd_momentum():
    data, data_shape, label, label_shape = get_input()
    mlp = MLP()
    opt = SGD(mlp.parameters(), lr=0.01, momentum=0.9)
    slots = TensorDict()
    for param in mlp.parameters():
        slots[param] = np.zeros(param.shape).astype(np.float32)
    for _ in range(3):
        data.set_value(np.random.random(data_shape).astype(np.float32))
        label.set_value(np.random.randint(0, 10, label_shape))
        pred = mlp(data)
        loss = F.square_loss(pred, label.reshape(-1, 1))
        opt.zero_grad()
        opt.backward(loss)
        orig_params = TensorDict()
        grads = TensorDict()
        for param in mlp.parameters():
            orig_params[param] = np.copy(param.numpy())
            grads[param] = np.copy(param.grad.numpy())
        opt.step()
        for param in mlp.parameters():
            slot = slots[param]
            orig_param = orig_params[param]
            slot *= 0.9
            slot -= param.grad.numpy() * 0.01
            assertTensorClose(param.numpy(), orig_param + slot)


# TODO: put opt.step() inside trace
def test_sgd_momentum_static():
    _, data_shape, _, label_shape = get_input()
    mlp = MLP()
    opt = SGD(mlp.parameters(), lr=0.01, momentum=0.9)

    @trace
    def f(data, label):
        pred = mlp(data)
        loss = F.square_loss(pred, label.reshape(-1, 1))
        opt.zero_grad()
        opt.backward(loss)

    slots = TensorDict()
    for param in mlp.parameters():
        slots[param] = np.zeros(param.shape).astype(np.float32)
    for _ in range(3):
        f(
            np.random.random(data_shape).astype(np.float32),
            np.random.randint(0, 10, label_shape).astype(np.int32),
        )
        orig_params = TensorDict()
        grads = TensorDict()
        for param in mlp.parameters():
            orig_params[param] = np.copy(param.numpy())
            grads[param] = np.copy(param.grad.numpy())
        opt.step()
        for param in mlp.parameters():
            slot = slots[param]
            orig_param = orig_params[param]
            slot *= 0.9
            slot -= param.grad.numpy() * 0.01
            assertTensorClose(param.numpy(), orig_param + slot)


def test_update_lr():
    data, data_shape, label, label_shape = get_input()
    mlp = MLP()
    opt = SGD(mlp.parameters(), lr=0.01)
    pred = mlp(data)
    loss = F.square_loss(pred, label.reshape(-1, 1))
    opt.zero_grad()
    opt.backward(loss)
    opt.step()
    for group in opt.param_groups:
        group["lr"] += 0.02
    for _ in range(3):
        data.set_value(np.random.random(data_shape).astype(np.float32))
        label.set_value(np.random.randint(0, 10, label_shape))
        pred = mlp(data)
        loss = F.square_loss(pred, label.reshape(-1, 1))
        opt.zero_grad()
        opt.backward(loss)
        for param in mlp.parameters():
            grad = F.grad(loss, param, use_virtual_grad=False)
            assertTensorClose(grad.numpy(), param.grad.numpy())
        orig_params = []
        for param in mlp.parameters():
            orig_params.append(np.copy(param.numpy()))
        opt.step()
        for param, orig_param in zip(mlp.parameters(), orig_params):
            assertTensorClose(param.numpy(), orig_param - param.grad.numpy() * 0.03)


def test_adam():
    data, data_shape, label, label_shape = get_input()
    mlp = MLP()
    beta0 = 0.8
    beta1 = 0.9
    eps = 1e-4
    opt = Adam(mlp.parameters(), lr=0.01, betas=(beta0, beta1), eps=eps)
    m_slots = TensorDict()
    v_slots = TensorDict()
    for param in mlp.parameters():
        m_slots[param] = np.zeros(param.shape).astype(np.float32)
        v_slots[param] = np.zeros(param.shape).astype(np.float32)
    step_size = 0

    def check_value():
        for param in mlp.parameters():
            grad = param.grad.numpy()
            orig_param = orig_params[param]
            m = m_slots[param]
            v = v_slots[param]
            m *= beta0
            m += (1 - beta0) * grad
            v *= beta1
            v += (1 - beta1) * grad * grad
            update = (m / (1 - beta0 ** step_size)) / (
                np.sqrt(v / (1 - beta1 ** step_size)) + eps
            )
            assertTensorClose(param.numpy(), orig_param - 0.01 * update)

    # eager
    for _ in range(3):
        data.set_value(np.random.random(data_shape).astype(np.float32))
        label.set_value(np.random.randint(0, 10, label_shape))
        pred = mlp(data)
        loss = F.square_loss(pred, label.reshape(-1, 1))
        opt.zero_grad()
        grads = opt.backward(loss)
        orig_params = TensorDict()
        for param in mlp.parameters():
            orig_params[param] = np.copy(param.numpy())
        opt.step()
        step_size += 1
        check_value()

    # static
    @trace
    def f(data, label):
        pred = mlp(data)
        loss = F.square_loss(pred, label.reshape(-1, 1))
        opt.backward(loss)

    for _ in range(3):
        opt.zero_grad()
        orig_params = TensorDict()
        for param in mlp.parameters():
            orig_params[param] = np.copy(param.numpy())
        f(
            np.random.random(data_shape).astype(np.float32),
            np.random.randint(0, 10, label_shape).astype(np.int32),
        )
        opt.step()
        step_size += 1
        check_value()


@graph_mode("eager", "static")
def test_optimizer_serialization():
    data, data_shape, label, label_shape = get_input()
    mlp = MLP()
    opt = SGD(mlp.parameters(), lr=0.01, momentum=0.9)
    slots = TensorDict()
    for param in mlp.parameters():
        slots[param] = np.zeros(param.shape).astype(np.float32)

    pred = mlp(data)
    loss = F.square_loss(pred, label.reshape(-1, 1))
    opt.zero_grad()
    opt.backward(loss)
    opt.step()
    for param in mlp.parameters():
        slot = slots[param]
        slot *= 0.9
        slot -= param.grad.numpy() * 0.01

    with BytesIO() as fout:
        save(opt.state_dict(), fout)
        fout.seek(0)
        state_dict = load(fout)
        opt1 = SGD(mlp.parameters(), lr=0.02, momentum=0.8)
        opt1.load_state_dict(state_dict)

        data.set_value(np.random.random(data_shape).astype(np.float32))
        label.set_value(np.random.randint(0, 10, label_shape))
        pred = mlp(data)
        loss = F.square_loss(pred, label.reshape(-1, 1))
        opt1.zero_grad()
        opt1.backward(loss)
        orig_params = TensorDict()
        for param in mlp.parameters():
            orig_params[param] = np.copy(param.numpy())
        opt1.step()
        for param in mlp.parameters():
            orig_param = orig_params[param]
            slot = slots[param]
            slot *= 0.9
            slot -= param.grad.numpy() * 0.01
            assertTensorClose(param.numpy(), orig_param + slot)
