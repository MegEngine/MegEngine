# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Iterable, Tuple, Union

from ..core import Buffer, Parameter
from .internal import add_update_fastpath as add_update
from .optimizer import Optimizer


class Adam(Optimizer):
    r"""Implements Adam algorithm.

    :param params: iterable of parameters to optimize or dicts defining
            parameter groups.
    :param lr: learning rate.
    :param betas: coefficients used for computing running averages of gradient
        and its square. Default: (0.9, 0.999)
    :param eps: term added to the denominator to improve numerical stability
        Default: 1e-8
    :param weight_decay: weight decay (L2 penalty). Default: 0
    """

    def __init__(
        self,
        params: Union[Iterable[Parameter], dict],
        lr: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        super().__init__(params, defaults)

    def _create_state(self, param_group):
        for param in param_group["params"]:
            self._add_state(param, "exp_avg")
            self._add_state(param, "exp_avg_sq")
            self._add_state(param, "step", initializer=0.0)

    def _updates(self, param_group):
        lr = param_group["lr"]
        weight_decay = param_group["weight_decay"]
        eps = param_group["eps"]
        beta0, beta1 = param_group["betas"]

        for param in param_group["params"]:
            if not param.requires_grad:
                continue

            step = self._state[param]["step"]
            step = add_update(step, 1)
            if not isinstance(param.grad, Buffer):
                raise TypeError(
                    "grad must be a Buffer, maybe you forget to call backward()?"
                )
            grad = param.grad
            if weight_decay != 0.0:
                grad = add_update(grad, param, beta=weight_decay)
            exp_avg = self._state[param]["exp_avg"]
            exp_avg_sq = self._state[param]["exp_avg_sq"]
            exp_avg = add_update(exp_avg, grad, alpha=beta0, beta=1 - beta0)
            exp_avg_sq = add_update(
                exp_avg_sq, grad * grad, alpha=beta1, beta=1 - beta1
            )
            add_update(
                param,
                exp_avg
                / (1 - beta0 ** step)
                / (exp_avg_sq.sqrt() / (1 - beta1 ** step).sqrt() + eps),
                beta=-lr,
            )
