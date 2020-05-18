from typing import Iterable, Union

import numpy as np

from ..core import Buffer, Parameter
from ..functional import sqrt
from .internal import add_update_fastpath as add_update
from .optimizer import Optimizer


class Adadelta(Optimizer):
    r"""Implements Adadelta algorithm.

    It has been proposed in `"ADADELTA: An Adaptive Learning Rate Method" <https://arxiv.org/abs/1212.5701>`_.

    :param params: iterable of parameters to optimize or dicts defining
        parameter groups.
    :param lr: coefficient that scale delta before it is applied
        to the parameters (default: 1.0).
    :param rho: coefficient used for computing a running average
        of squared gradients (default: 0.9).
    :param eps: term added to the denominator to improve
        numerical stability (default: 1e-6).
    :param weight_decay: weight decay (L2 penalty) (default: 0).
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

        for param in param_group["params"]:
            if not isinstance(param.grad, Buffer):
                raise TypeError(
                    "grad must be a Buffer, maybe you forget to call backward()?"
                )

            if not param.requires_grad:
                continue

            step = self._state[param]["step"]
            step = add_update(step, 1)
            grad = param.grad
            if weight_decay != 0.0:
                grad = add_update(grad, param, beta=weight_decay)

            square_avg = self._state[param]["square_avg"]
            acc_delta = self._state[param]["acc_delta"]
            square_avg = add_update(square_avg, grad ** 2, alpha=rho, beta=1 - rho)
            std = sqrt(square_avg + eps)
            delta = sqrt(acc_delta + eps) / std * grad
            add_update(param, delta, beta=-lr)
            acc_delta = add_update(acc_delta, delta ** 2, alpha=rho, beta=1 - rho)
