# -*- coding: utf-8 -*-
import os
from typing import Iterable, Tuple, Union

from ..functional.inplace import _inplace_add_
from ..tensor import Parameter, tensor
from .optimizer import Optimizer


class AdamW(Optimizer):
    r"""Implements AdamW algorithm proposed in `"Decoupled Weight Decay Regularization" <https://arxiv.org/abs/1711.05101>`_.

    Args:
        params: iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: learning rate.
            betas: coefficients used for computing running averages of gradient
            and its square. Default: (0.9, 0.999)
        eps: term added to the denominator to improve numerical stability. Default: 1e-8
        weight_decay: weight decay (L2 penalty). Default: 1e-2
    """

    def __init__(
        self,
        params: Union[Iterable[Parameter], dict],
        lr: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
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
        self._disable_type_convert = True

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

        def make_scalar(val):
            return tensor(val, dtype="float32")

        # since `conver_inputs` is disabled for param updates,
        # scalar should be explicitly tansforred to tensor

        _lr, _neg_lr = map(make_scalar, (lr, -lr))
        _weight_decay = make_scalar(weight_decay)
        _eps = make_scalar(eps)
        _beta0, _beta1 = map(make_scalar, (beta0, beta1))

        c1, c05 = map(make_scalar, (1.0, 0.5))

        inplace_mode = int(os.getenv("MEGENGINE_INPLACE_UPDATE", "0"))
        if inplace_mode:
            # reduce device sync
            c1_sub_beta0, c1_sub_beta1 = map(make_scalar, (1 - beta0, 1 - beta1))

        for param in param_group["params"]:

            if param.grad is None:
                continue

            grad = param.grad

            states = self._state[param]

            step, exp_avg, exp_avg_sq = (
                states["step"],
                states["exp_avg"],
                states["exp_avg_sq"],
            )

            if inplace_mode:
                _inplace_add_(step, c1, alpha=c1, beta=c1)
                _inplace_add_(exp_avg, grad, alpha=_beta0, beta=c1_sub_beta0)
                _inplace_add_(
                    exp_avg_sq, grad * grad, alpha=_beta1, beta=c1_sub_beta1,
                )

                delta = (exp_avg / (c1 - _beta0 ** step)) / (
                    (exp_avg_sq / (c1 - _beta1 ** step)) ** c05 + _eps
                )
                if weight_decay != 0.0:
                    delta += param * _weight_decay
                _inplace_add_(param, delta, alpha=c1, beta=_neg_lr)
                continue

            # step = step + c1
            step += c1

            # exp_avg = _beta0 * exp_avg + grad * (c1 - _beta0)
            exp_avg *= _beta0
            exp_avg += grad * (c1 - _beta0)

            # exp_avg_sq = _beta1 * exp_avg_sq + (c1 - _beta1) * (grad * grad)
            exp_avg_sq *= _beta1
            exp_avg_sq += (c1 - _beta1) * (grad * grad)

            delta = (exp_avg / (c1 - _beta0 ** step)) / (
                (exp_avg_sq / (c1 - _beta1 ** step)) ** c05 + _eps
            )
            if weight_decay != 0.0:
                delta += param * _weight_decay

            param -= _lr * delta
