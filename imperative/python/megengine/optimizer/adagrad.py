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

from ..functional import sqrt
from ..tensor_nn import Parameter
from .optimizer import Optimizer


class Adagrad(Optimizer):
    r"""Implements Adagrad algorithm.

    It has been proposed in `"Adaptive Subgradient Methods for Online Learning
    and Stochastic Optimization" <http://jmlr.org/papers/v12/duchi11a.html>`_.

    :param params: iterable of parameters to optimize or dicts defining
        parameter groups.
    :param lr: coefficient that scale delta before it is applied
        to the parameters (default: 1e-2).
    :param lr_decay: learning rate decay (default: 0)
    :param eps: term added to the denominator to improve
        numerical stability (default: 1e-10).
    :param weight_decay: weight decay (L2 penalty) (default: 0).
    """

    def __init__(
        self,
        params: Union[Iterable[Parameter], dict],
        lr: float = 1e-2,
        lr_decay: float = 0.0,
        eps: float = 1e-10,
        weight_decay: float = 0.0,
    ):
        assert lr >= 0.0, "Invalid learning rate: {}".format(lr)
        assert lr_decay >= 0, "Invalid learning rate decay: {}".format(lr_decay)
        assert eps >= 0.0, "Invalid epsilon value: {}".format(eps)
        assert weight_decay >= 0.0, "Invalid weight_decay value: {}".format(
            weight_decay
        )

        defaults = dict(lr=lr, lr_decay=lr_decay, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _create_state(self, param_group):
        for param in param_group["params"]:
            self._add_state(param, "square_avg")
            self._add_state(param, "step", initializer=0.0)

    def _updates(self, param_group):
        lr = param_group["lr"]
        lr_decay = param_group["lr_decay"]
        weight_decay = param_group["weight_decay"]
        eps = param_group["eps"]

        for param in param_group["params"]:

            if not param.requires_grad or "grad" not in param.__dict__:
                continue

            states = self._state[param]
            step = states["step"]
            step += 1.0
            grad = param.grad
            if weight_decay != 0.0:
                grad += param * weight_decay

            square_avg = states["square_avg"]
            square_avg += grad ** 2
            delta = grad / sqrt(square_avg + eps)
            clr = lr / (1 + (step - 1) * lr_decay)

            param -= clr * delta
