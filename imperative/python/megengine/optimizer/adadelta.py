# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Iterable, Union

import numpy as np

from ..tensor import Parameter, tensor
from .optimizer import Optimizer


class Adadelta(Optimizer):
    r"""
    Implements Adadelta algorithm.

    It has been proposed in `"ADADELTA: An Adaptive Learning Rate Method" <https://arxiv.org/abs/1212.5701>`_.

    :param params: iterable of parameters to optimize or dicts defining
        parameter groups.
    :param lr: coefficient that scales delta before it is applied
        to the parameters. Default: 1.0
    :param rho: coefficient used for computing a running average
        of squared gradients. Default: 0.9
    :param eps: term added to the denominator to improve
        numerical stability. Default: 1e-6
    :param weight_decay: weight decay (L2 penalty). Default: 0
    """

    def __init__(
        self,
        params: Union[Iterable[Parameter], dict],
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ):
        assert lr >= 0.0, "Invalid learning rate: {}".format(lr)
        assert rho >= 0.0 and rho <= 1.0, "Invalid rho value: {}".format(rho)
        assert eps >= 0.0, "Invalid epsilon value: {}".format(eps)
        assert weight_decay >= 0.0, "Invalid weight_decay value: {}".format(
            weight_decay
        )

        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _create_state(self, param_group):
        for param in param_group["params"]:
            self._add_state(param, "square_avg")
            self._add_state(param, "acc_delta")
            self._add_state(param, "step", initializer=0.0)

    def _updates(self, param_group):
        lr = param_group["lr"]
        weight_decay = param_group["weight_decay"]
        rho = param_group["rho"]
        eps = param_group["eps"]

        # since `conver_inputs` is disabled for param updates,
        # scalar should be explicitly tansforred to tensor
        _lr = tensor([lr])
        _weight_decay = tensor([weight_decay])
        _rho = tensor([rho])
        _eps = tensor([eps])

        c05 = tensor([0.5])
        c1 = tensor([1.0])
        c2 = tensor([2.0])
        for param in param_group["params"]:

            if param.grad is None:
                continue

            states = self._state[param]
            step = states["step"]
            step += c1
            grad = param.grad
            if weight_decay != 0.0:
                grad += param * _weight_decay

            square_avg = states["square_avg"]
            acc_delta = states["acc_delta"]
            square_avg = _rho * square_avg + (c1 - _rho) * grad ** c2
            std = (square_avg + _eps) ** c05
            delta = (acc_delta + _eps) ** c05 / std * grad
            param -= _lr * delta
            acc_delta = _rho * acc_delta + (c1 - _rho) * delta ** c2
            states["square_avg"]._reset(square_avg)
            states["acc_delta"]._reset(acc_delta)
