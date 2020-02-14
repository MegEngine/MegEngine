# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Iterable, Union

from ..core import Buffer, Parameter
from .internal import add_update_fastpath as add_update
from .optimizer import Optimizer


class SGD(Optimizer):
    r"""Implements stochastic gradient descent.

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`.

    :param params: iterable of parameters to optimize or dicts defining
            parameter groups.
    :param lr: learning rate.
    :param momentum: momentum factor. Default: 0.0
    :param weight_decay: weight decay (L2 penalty). Default: 0.0
    """

    def __init__(
        self,
        params: Union[Iterable[Parameter], dict],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        assert lr >= 0.0, "Invalid learning rate: {}".format(lr)
        assert momentum >= 0.0, "Invalid momentum value: {}".format(momentum)
        assert weight_decay >= 0.0, "Invalid weight_decay value: {}".format(
            weight_decay
        )

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _create_state(self, param_group):
        if param_group["momentum"] != 0.0:
            for param in param_group["params"]:
                self._add_state(param, "momentum_buffer")

    def _updates(self, param_group):
        lr = param_group["lr"]
        weight_decay = param_group["weight_decay"]
        momentum = param_group["momentum"]

        for param in param_group["params"]:
            if not isinstance(param.grad, Buffer):
                raise TypeError(
                    "grad must be a Buffer, maybe you forget to call backward()?"
                )

            if not param.requires_grad:
                continue

            grad = param.grad
            if weight_decay != 0.0:
                grad = add_update(grad, param, beta=weight_decay)

            if momentum:
                v = self._state[param]["momentum_buffer"]
                update_v = add_update(v, grad, alpha=momentum)
                add_update(param, update_v, beta=-lr)
            else:
                add_update(param, grad, beta=-lr)
