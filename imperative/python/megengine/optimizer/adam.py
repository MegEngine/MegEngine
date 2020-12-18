# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Iterable, Tuple, Union

from ..tensor import Parameter, tensor
from .optimizer import Optimizer


class Adam(Optimizer):
    r"""
    Implements Adam algorithm proposed in `"Adam: A Method for Stochastic Optimization" <https://arxiv.org/abs/1412.6980>`_.

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

        # since `conver_inputs` is disabled for param updates,
        # scalar should be explicitly tansforred to tensor
        _lr = tensor([lr])
        _weight_decay = tensor([weight_decay])
        _eps = tensor([eps])
        _beta0, _beta1 = tensor([beta0]), tensor([beta1])

        c1 = tensor([1.0])
        c05 = tensor([0.5])
        for param in param_group["params"]:

            if param.grad is None:
                continue

            grad = param.grad
            if weight_decay != 0.0:
                grad += param * _weight_decay

            states = self._state[param]
            step = states["step"]
            step += c1
            exp_avg = states["exp_avg"]
            exp_avg_sq = states["exp_avg_sq"]
            exp_avg = _beta0 * exp_avg + grad * (c1 - _beta0)
            exp_avg_sq = _beta1 * exp_avg_sq + (c1 - _beta1) * (grad * grad)

            delta = (exp_avg / (c1 - _beta0 ** step)) / (
                (exp_avg_sq / (c1 - _beta1 ** step)) ** c05 + _eps
            )
            param -= _lr * delta

            # not inplace change, need to update underlying tensor handler in state
            states["exp_avg"]._reset(exp_avg)
            states["exp_avg_sq"]._reset(exp_avg_sq)
