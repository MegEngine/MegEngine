# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import os
from typing import Iterable, Union

from ..functional.inplace import _inplace_add_
from ..tensor import Parameter, tensor
from .optimizer import Optimizer


class SGD(Optimizer):
    r"""
    Implements stochastic gradient descent.

    Nesterov momentum is based on the formula from
    `"On the importance of initialization and momentum in deep learning" <http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf>`_ .

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

        # since `conver_inputs` is disabled for param updates,
        # scalar should be explicitly tansforred to tensor

        _lr = tensor([lr])
        _weight_decay = tensor([weight_decay])
        _momentum = tensor([momentum])

        inplace_mode = int(os.getenv("MEGENGINE_INPLACE_UPDATE", "0"))
        if inplace_mode:
            _neg_lr = tensor([-lr])
            c1 = tensor([1.0])

        for param in param_group["params"]:
            if param.grad is None:
                continue

            grad = param.grad
            if weight_decay != 0.0:
                grad += param * _weight_decay

            if inplace_mode:
                if momentum:
                    v = self._state[param]["momentum_buffer"]
                    _inplace_add_(v, grad, alpha=_momentum, beta=c1)
                    _inplace_add_(param, v, alpha=c1, beta=_neg_lr)
                else:
                    _inplace_add_(param, grad, alpha=c1, beta=_neg_lr)
                continue

            if momentum:
                v = self._state[param]["momentum_buffer"]
                # v = v * _momentum + grad
                v *= _momentum
                v += grad

                param -= _lr * v
            else:
                param -= _lr * grad
