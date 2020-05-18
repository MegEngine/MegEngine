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
from megengine import load, optimizer, save
from megengine.core import TensorDict, tensor
from megengine.jit import trace
from megengine.test import assertTensorClose


def get_input():
    batch_size, input_dim = 2, 28
    data_shape, label_shape = (batch_size, input_dim), (batch_size,)
    data, label = tensor(dtype=np.float32), tensor(dtype=np.int32)
    data.set_value(np.random.random(data_shape).astype(np.float32))
    label.set_value(np.random.randint(0, 10, label_shape))
    return data, data_shape, label, label_shape


@graph_mode("eager", "static")
def test_optimizer_serialization():
    data, data_shape, label, label_shape = get_input()
    mlp = MLP()
    opt = optimizer.SGD(mlp.parameters(), lr=0.01, momentum=0.9)
    slots = TensorDict()
    for param in mlp.parameters():
        slots[param] = np.zeros(param.shape).astype(np.float32)

    pred = mlp(data)
    loss = F.square_loss(pred, label.reshape(-1, 1))
    opt.zero_grad()
    opt.backward(loss)
    opt.step()
    for param in mlp.parameters():
        slots[param] = slots[param] * 0.9 + param.grad.numpy()

    with BytesIO() as fout:
        save(opt.state_dict(), fout)
        fout.seek(0)
        state_dict = load(fout)
        opt1 = optimizer.SGD(mlp.parameters(), lr=0.02, momentum=0.8)
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
            slots[param] = slots[param] * 0.9 + param.grad.numpy()
            assertTensorClose(param.numpy(), orig_param - 0.01 * slots[param])


def _test_optimizer(opt_str, test_case, check_class, update_lr=False):
    iter_num = 3
    data, data_shape, label, label_shape = get_input()

    net = MLP()
    opt = getattr(optimizer, opt_str)(net.parameters(), **test_case)
    check_func = check_class(net, **test_case)

    step = 0

    # eager graph
    for i in range(iter_num):
        if update_lr and i == 1:  # change learning rate
            for group in opt.param_groups:
                group["lr"] += 0.01
            check_func.lr += 0.01
        data.set_value(np.random.random(data_shape).astype(np.float32))
        label.set_value(np.random.randint(0, 10, label_shape))
        pred = net(data)
        loss = F.square_loss(pred, label.reshape(-1, 1))
        opt.zero_grad()
        opt.backward(loss)
        ori_params = TensorDict()
        for param in net.parameters():
            ori_params[param] = np.copy(param.numpy())
        opt.step()
        step += 1
        check_func(ori_params, net.parameters(), step)

    # static graph
    @trace
    def train_func(data, label):
        pred = net(data)
        loss = F.square_loss(pred, label.reshape(-1, 1))
        opt.backward(loss)

    for i in range(iter_num):
        if update_lr and i == 1:  # change learning rate
            for group in opt.param_groups:
                group["lr"] += 0.01
            check_func.lr += 0.01
        opt.zero_grad()
        ori_params = TensorDict()
        for param in net.parameters():
            ori_params[param] = np.copy(param.numpy())
        train_func(
            np.random.random(data_shape).astype(np.float32),
            np.random.randint(0, 10, label_shape).astype(np.int32),
        )
        opt.step()
        step += 1
        check_func(ori_params, net.parameters(), step)


def test_sgd():
    class CheckValue:
        def __init__(self, net, **kwarg):
            self.slots = TensorDict()
            for param in net.parameters():
                self.slots[param] = np.zeros(param.shape).astype(np.float32)
            for k, v in kwarg.items():
                setattr(self, k, v)

        def __call__(self, ori_params, new_params, step):
            for param in new_params:
                grad = param.grad.numpy()
                if hasattr(self, "momentum"):
                    self.slots[param] = grad + self.slots[param] * self.momentum
                    delta = -self.lr * self.slots[param]
                else:
                    delta = -self.lr * grad
                assertTensorClose(param.numpy(), ori_params[param] + delta)

    cases = [
        {"momentum": 0.9, "lr": 0.01},  # SGD with momentum
        {"lr": 0.01},  # simple SGD
        {"weight_decay": 0.1, "lr": 0.01},  # with weight_decay
    ]
    for case in cases:
        _test_optimizer("SGD", case, CheckValue)
        _test_optimizer("SGD", case, CheckValue, update_lr=True)


def test_adam():
    class CheckValue:
        def __init__(self, net, **kwarg):
            self.m_slots = TensorDict()
            self.v_slots = TensorDict()
            for param in net.parameters():
                self.m_slots[param] = np.zeros(param.shape).astype(np.float32)
                self.v_slots[param] = np.zeros(param.shape).astype(np.float32)
            for k, v in kwarg.items():
                setattr(self, k, v)

        def __call__(self, ori_params, new_params, step):
            for param in new_params:
                grad = param.grad.numpy()
                m = self.m_slots[param]
                v = self.v_slots[param]
                m *= self.betas[0]
                m += (1 - self.betas[0]) * grad
                v *= self.betas[1]
                v += (1 - self.betas[1]) * grad * grad
                delta = (m / (1 - self.betas[0] ** step)) / (
                    np.sqrt(v / (1 - self.betas[1] ** step)) + self.eps
                )
                assertTensorClose(param.numpy(), ori_params[param] - self.lr * delta)

    cases = [
        {"betas": (0.8, 0.9), "eps": 1e-04, "lr": 0.01},
        {
            "betas": (0.8, 0.9),
            "eps": 1e-04,
            "lr": 0.01,
            "weight_decay": 0.1,
        },  # with weight_decay
    ]
    for case in cases:
        _test_optimizer("Adam", case, CheckValue)
        _test_optimizer("Adam", case, CheckValue, update_lr=True)


def test_adagrad():
    class CheckValue:
        def __init__(self, net, **kwarg):
            self.s_slots = TensorDict()
            for param in net.parameters():
                self.s_slots[param] = np.zeros(param.shape).astype(np.float32)
            for k, v in kwarg.items():
                setattr(self, k, v)

        def __call__(self, ori_params, new_params, step):
            for param in new_params:
                grad = param.grad.numpy()
                self.s_slots[param] += grad ** 2
                delta = grad / (self.s_slots[param] + self.eps) ** 0.5
                delta *= -(self.lr / (1 + (step - 1) * self.lr_decay))
                assertTensorClose(param.numpy(), ori_params[param] + delta)

    cases = [
        {"lr": 0.01, "eps": 1e-06, "lr_decay": 0.01},
        {"lr": 0.01, "eps": 1e-06, "lr_decay": 0.0},  # without lr_decay
        {
            "lr": 0.01,
            "eps": 1e-06,
            "lr_decay": 0.01,
            "weight_decay": 0.1,
        },  # with weight_decay
    ]
    for case in cases:
        _test_optimizer("Adagrad", case, CheckValue)
        _test_optimizer("Adagrad", case, CheckValue, update_lr=True)


def test_adadelta():
    class CheckValue:
        def __init__(self, net, **kwarg):
            self.s_slots = TensorDict()
            self.a_slots = TensorDict()
            for param in net.parameters():
                self.s_slots[param] = np.zeros(param.shape).astype(np.float32)
                self.a_slots[param] = np.zeros(param.shape).astype(np.float32)
            for k, v in kwarg.items():
                setattr(self, k, v)

        def __call__(self, ori_params, new_params, step):
            for param in new_params:
                grad = param.grad.numpy()
                self.s_slots[param] = self.s_slots[param] * self.rho + grad ** 2 * (
                    1 - self.rho
                )
                delta = (
                    grad
                    * ((self.a_slots[param] + self.eps) ** 0.5)
                    / (self.s_slots[param] + self.eps) ** 0.5
                )
                self.a_slots[param] = self.a_slots[param] * self.rho + delta ** 2 * (
                    1 - self.rho
                )
                delta *= -self.lr
                assertTensorClose(param.numpy(), ori_params[param] + delta)

    cases = [
        {"lr": 1.0, "eps": 1e-06, "rho": 0.9},
        {"lr": 1.0, "eps": 1e-06, "rho": 0.9, "weight_decay": 0.9},  # with weight_decay
    ]
    for case in cases:
        _test_optimizer("Adadelta", case, CheckValue)
        _test_optimizer("Adadelta", case, CheckValue, update_lr=True)
