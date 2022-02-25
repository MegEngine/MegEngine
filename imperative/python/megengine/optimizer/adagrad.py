# -*- coding: utf-8 -*-
from typing import Iterable, Union

import numpy as np

from ..tensor import Parameter, tensor
from .optimizer import Optimizer


class Adagrad(Optimizer):
    r"""Implements Adagrad algorithm.
    
    It has been proposed in `"Adaptive Subgradient Methods for Online Learning
    and Stochastic Optimization" <http://jmlr.org/papers/v12/duchi11a.html>`_.

    Args:
        params: iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: coefficient that scales delta before it is applied
            to the parameters. Default: 1e-2
        lr_decay: learning rate decay. Default: 0
        eps: term added to the denominator to improve
            numerical stability. Default: 1e-10
        weight_decay: weight decay (L2 penalty). Default: 0
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
        self._disable_type_convert = True

    def _create_state(self, param_group):
        for param in param_group["params"]:
            self._add_state(param, "square_avg")
            self._add_state(param, "step", initializer=0.0)

    def _updates(self, param_group):
        lr = param_group["lr"]
        lr_decay = param_group["lr_decay"]
        weight_decay = param_group["weight_decay"]
        eps = param_group["eps"]

        def make_scalar(val):
            return tensor(val, dtype="float32")

        # since `conver_inputs` is disabled for param updates,
        # scalar should be explicitly tansforred to tensor

        _lr, _lr_decay = map(make_scalar, (lr, lr_decay))
        _weight_decay = make_scalar(weight_decay)
        _eps = make_scalar(eps)

        c1, c2, c05 = map(make_scalar, (1.0, 2.0, 0.5))

        for param in param_group["params"]:

            if param.grad is None:
                continue

            states = self._state[param]
            step = states["step"]
            step += c1
            grad = param.grad
            if weight_decay != 0.0:
                grad = grad + param * _weight_decay

            square_avg = states["square_avg"]
            square_avg += grad ** c2
            delta = grad / (square_avg + _eps) ** c05
            clr = _lr / (c1 + (step - c1) * _lr_decay)

            param -= clr * delta
