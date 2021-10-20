# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import os

import numpy as np
import pytest

import megengine.autodiff as ad
import megengine.functional as F
from megengine import Parameter, optimizer
from megengine.jit import trace
from megengine.module import Linear, Module
from megengine.tensor import Tensor


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.dense0 = Linear(28, 50)
        self.dense1 = Linear(50, 20)

    def forward(self, x):
        x = self.dense0(x)
        x = F.relu(x)
        x = self.dense1(x)
        return x


class Simple(Module):
    def __init__(self):
        super().__init__()
        self.a = Parameter(1.23, dtype=np.float32)

    def forward(self, x):
        x = x * self.a
        return x


def _test_optimizer(opt_str, test_case, check_class, update_lr=False):
    iter_num = 3
    net = Simple()
    opt = getattr(optimizer, opt_str)(net.parameters(), **test_case)
    check_func = check_class(net, **test_case)
    gm = ad.GradManager().attach(net.parameters())

    step = 0
    data_shape = (2, 28)

    for i in range(iter_num):
        if update_lr and i == 1:  # change learning rate
            for group in opt.param_groups:
                group["lr"] += 0.01
            check_func.lr += 0.01
        data = Tensor(np.random.random(data_shape).astype(np.float32))

        opt.clear_grad()
        with gm:
            pred = net(data)
            loss = pred.sum()
            gm.backward(loss)

        ori_params = {}
        ori_grads = {}
        for param in net.parameters():
            assert param._tuple_shape is ()
            ori_params[param] = np.copy(param.numpy())
            ori_grads[param] = np.copy(param.grad.numpy())
        opt.step()
        # check grad not change
        for param in net.parameters():
            assert np.equal(
                ori_grads[param], param.grad.numpy()
            ), "step should not change param.grad"
        step += 1
        check_func(ori_params, net.parameters(), step)

    # static graph
    for symbolic in (False, True):

        @trace(symbolic=symbolic)
        def train_func(data, *, opt=None, gm=None):
            opt.clear_grad()
            with gm:
                pred = net(data)
                loss = pred.sum()
                gm.backward(loss)
            opt.step()

        # reset net and opt
        net = Simple()
        opt = getattr(optimizer, opt_str)(net.parameters(), **test_case)
        gm = ad.GradManager().attach(net.parameters())
        check_func = check_class(net, **test_case)
        step = 0
        for i in range(iter_num):
            if update_lr and i == 1:  # change learning rate
                for group in opt.param_groups:
                    group["lr"] += 0.01
                check_func.lr += 0.01

            ori_params = {}
            for param in net.parameters():
                assert param._tuple_shape is ()
                ori_params[param] = np.copy(param.numpy())

            train_func(
                Tensor(np.random.random(data_shape).astype(np.float32)), opt=opt, gm=gm
            )
            step += 1
            check_func(ori_params, net.parameters(), step)
            try_state_dict = {
                "net": net.state_dict(),
                "opt": opt.state_dict(),
            }


@pytest.mark.parametrize(
    "case",
    [
        {"momentum": 0.9, "lr": 0.01},  # SGD with momentum
        {"momentum": 0.9, "lr": 0.01, "nesterov": True},  # with nesterov momentum
        {"lr": 0.01},  # simple SGD
        {"weight_decay": 0.1, "lr": 0.01},  # with weight_decay
    ],
)
@pytest.mark.parametrize("update_lr", [False, True])
@pytest.mark.parametrize("inplace_mode", [False, True])
def test_sgd(monkeypatch, case, update_lr, inplace_mode):
    class CheckValue:
        def __init__(self, net, **kwarg):
            self.slots = {}
            for param in net.parameters():
                self.slots[param] = np.zeros(param.shape).astype(np.float32)
            for k, v in kwarg.items():
                setattr(self, k, v)

        def __call__(self, ori_params, new_params, step):
            for param in new_params:
                grad = param.grad.numpy()
                if hasattr(self, "weight_decay") and self.weight_decay != 0.0:
                    grad = grad + ori_params[param] * self.weight_decay
                if hasattr(self, "momentum") and self.momentum != 0.0:
                    self.slots[param] = grad + self.slots[param] * self.momentum
                    if hasattr(self, "nesterov") and self.nesterov:
                        delta = -self.lr * (grad + self.slots[param] * self.momentum)
                    else:
                        delta = -self.lr * self.slots[param]
                else:
                    delta = -self.lr * grad
                np.testing.assert_almost_equal(
                    param.numpy(), ori_params[param] + delta, decimal=6
                )

    with monkeypatch.context() as mk:
        mk.setenv("MEGENGINE_INPLACE_UPDATE", str(int(inplace_mode)))
        _test_optimizer("SGD", case, CheckValue, update_lr=update_lr)


@pytest.mark.parametrize(
    "case",
    [
        {"betas": (0.8, 0.9), "eps": 1e-04, "lr": 0.01},
        {
            "betas": (0.8, 0.9),
            "eps": 1e-04,
            "lr": 0.01,
            "weight_decay": 0.1,
        },  # with weight_decay
    ],
)
@pytest.mark.parametrize("update_lr", [False, True])
@pytest.mark.parametrize("inplace_mode", [False, True])
def test_adam(monkeypatch, case, update_lr, inplace_mode):
    class CheckValue:
        def __init__(self, net, **kwarg):
            self.m_slots = {}
            self.v_slots = {}
            for param in net.parameters():
                self.m_slots[param] = np.zeros(param.shape).astype(np.float32)
                self.v_slots[param] = np.zeros(param.shape).astype(np.float32)
            for k, v in kwarg.items():
                setattr(self, k, v)

        def __call__(self, ori_params, new_params, step):
            for param in new_params:
                grad = param.grad.numpy()
                if hasattr(self, "weight_decay") and self.weight_decay != 0.0:
                    grad = grad + ori_params[param] * self.weight_decay
                m = self.m_slots[param]
                v = self.v_slots[param]
                m *= self.betas[0]
                m += (1 - self.betas[0]) * grad
                v *= self.betas[1]
                v += (1 - self.betas[1]) * grad * grad
                delta = (m / (1 - self.betas[0] ** step)) / (
                    np.sqrt(v / (1 - self.betas[1] ** step)) + self.eps
                )
                np.testing.assert_almost_equal(
                    param.numpy(), ori_params[param] - self.lr * delta, decimal=6
                )

    with monkeypatch.context() as mk:
        mk.setenv("MEGENGINE_INPLACE_UPDATE", str(int(inplace_mode)))
        _test_optimizer("Adam", case, CheckValue, update_lr=update_lr)


@pytest.mark.parametrize(
    "case",
    [
        {"lr": 0.01, "eps": 1e-06, "lr_decay": 0.01},
        {"lr": 0.01, "eps": 1e-06, "lr_decay": 0.0},  # without lr_decay
        {
            "lr": 0.01,
            "eps": 1e-06,
            "lr_decay": 0.01,
            "weight_decay": 0.1,
        },  # with weight_decay
    ],
)
@pytest.mark.parametrize("update_lr", [False, True])
@pytest.mark.parametrize("inplace_mode", [False, True])
def test_adagrad(monkeypatch, case, update_lr, inplace_mode):
    class CheckValue:
        def __init__(self, net, **kwarg):
            self.s_slots = {}
            for param in net.parameters():
                self.s_slots[param] = np.zeros(param.shape).astype(np.float32)
            for k, v in kwarg.items():
                setattr(self, k, v)

        def __call__(self, ori_params, new_params, step):
            for param in new_params:
                grad = param.grad.numpy()
                if hasattr(self, "weight_decay") and self.weight_decay != 0.0:
                    grad = grad + ori_params[param] * self.weight_decay
                self.s_slots[param] += grad ** 2
                delta = grad / (self.s_slots[param] + self.eps) ** 0.5
                delta *= -(self.lr / (1 + (step - 1) * self.lr_decay))
                np.testing.assert_almost_equal(
                    param.numpy(), ori_params[param] + delta, decimal=6
                )

    with monkeypatch.context() as mk:
        mk.setenv("MEGENGINE_INPLACE_UPDATE", str(int(inplace_mode)))
        _test_optimizer("Adagrad", case, CheckValue, update_lr=update_lr)


@pytest.mark.parametrize(
    "case",
    [
        {"lr": 1.0, "eps": 1e-06, "rho": 0.9},
        {"lr": 1.0, "eps": 1e-06, "rho": 0.9, "weight_decay": 0.9},  # with weight_decay
    ],
)
@pytest.mark.parametrize("update_lr", [False, True])
@pytest.mark.parametrize("inplace_mode", [False, True])
def test_adadelta(monkeypatch, case, update_lr, inplace_mode):
    class CheckValue:
        def __init__(self, net, **kwarg):
            self.s_slots = {}
            self.a_slots = {}
            for param in net.parameters():
                self.s_slots[param] = np.zeros(param.shape).astype(np.float32)
                self.a_slots[param] = np.zeros(param.shape).astype(np.float32)
            for k, v in kwarg.items():
                setattr(self, k, v)

        def __call__(self, ori_params, new_params, step):
            for param in new_params:
                grad = param.grad.numpy()
                if hasattr(self, "weight_decay") and self.weight_decay != 0.0:
                    grad = grad + ori_params[param] * self.weight_decay
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
                np.testing.assert_almost_equal(
                    param.numpy(), ori_params[param] + delta, decimal=6
                )

    with monkeypatch.context() as mk:
        mk.setenv("MEGENGINE_INPLACE_UPDATE", str(int(inplace_mode)))
        _test_optimizer("Adadelta", case, CheckValue, update_lr=update_lr)


@pytest.mark.parametrize(
    "case",
    [
        {"betas": (0.8, 0.9), "eps": 1e-08, "lr": 0.01},
        {
            "betas": (0.8, 0.9),
            "eps": 1e-08,
            "lr": 0.01,
            "weight_decay": 0.1,
        },  # with weight_decay
    ],
)
@pytest.mark.parametrize("update_lr", [False, True])
@pytest.mark.parametrize("inplace_mode", [False, True])
def test_adamw(monkeypatch, case, update_lr, inplace_mode):
    class CheckValue:
        def __init__(self, net, **kwarg):
            self.m_slots = {}
            self.v_slots = {}
            for param in net.parameters():
                self.m_slots[param] = np.zeros(param.shape).astype(np.float32)
                self.v_slots[param] = np.zeros(param.shape).astype(np.float32)
            self.weight_decay = 0.01
            for k, v in kwarg.items():
                setattr(self, k, v)

        def __call__(self, ori_params, new_params, step):
            step = np.array(step).astype(np.float32)
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
                delta += ori_params[param] * self.weight_decay
                np.testing.assert_almost_equal(
                    param.numpy(), ori_params[param] - self.lr * delta, decimal=6
                )

    with monkeypatch.context() as mk:
        mk.setenv("MEGENGINE_INPLACE_UPDATE", str(int(inplace_mode)))
        _test_optimizer("AdamW", case, CheckValue, update_lr=update_lr)
