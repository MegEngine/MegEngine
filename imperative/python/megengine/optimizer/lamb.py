# Copyright (c) 2020 Ross Wightman
# This file has been modified by Megvii ("Megvii Modifications").
# All Megvii Modifications are Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
"""LAMB optimizer

References: https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/lamb.py
"""
import os
from typing import Iterable, Tuple, Union

from megengine.core._imperative_rt.core2 import apply
from megengine.core.ops.builtin import LAMBUpdate

from .. import Parameter, tensor
from ..functional import sum
from ..functional.inplace import _inplace_add_
from .optimizer import Optimizer


class LAMB(Optimizer):
    r"""Implements LAMB algorithm.

    LAMB is proposed in `"Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"
    <https://arxiv.org/abs/1904.00962>`_.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr: learning rate.
        betas: coefficients used for computing running averages of gradient and its square.
            Default: ``(0.9, 0.999)``
        eps: term added to the denominator to improve numerical stability. Default: ``1e-8``
        bias_correction: enables bias correction by ``1 - beta ** step``. Default: ``True``
        weight_decay: weight decay (L2 penalty). Default: ``0.0``
        always_adapt: apply adaptive lr to ``0.0`` weight decay parameter. Default: ``False``
    """

    def __init__(
        self,
        params: Union[Iterable[Parameter], dict],
        lr: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        bias_correction: bool = True,
        weight_decay: float = 0.0,
        always_adapt: bool = False,
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
        self.bias_correction = bias_correction
        self.always_adapt = always_adapt
        self._disable_type_convert = True

    def _create_state(self, param_group):
        for param in param_group["params"]:
            self._add_state(param, "exp_avg")
            self._add_state(param, "exp_avg_sq")
            self._add_state(param, "step", initializer=0.0, dtype="float32")

    def _updates(self, param_group):
        lr = param_group["lr"]
        weight_decay = param_group["weight_decay"]
        eps = param_group["eps"]
        beta0, beta1 = param_group["betas"]

        # since `conver_inputs` is disabled for param updates,
        # scalar should be explicitly tansforred to tensor
        c1 = tensor(1.0)

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
            step += c1

            op = LAMBUpdate(
                beta0,
                beta1,
                int(step),
                lr,
                weight_decay,
                eps,
                self.bias_correction,
                self.always_adapt,
            )

            new_exp_avg, new_exp_avg_sq, new_param = apply(
                op, exp_avg, exp_avg_sq, param, grad
            )
            param._reset(new_param)
            exp_avg._reset(new_exp_avg)
            exp_avg_sq._reset(new_exp_avg_sq)


class LAMBFp16(LAMB):
    def _create_state(self, param_group):
        for param in param_group["params"]:
            self._add_state(param, "exp_avg", dtype="float32")
            self._add_state(param, "exp_avg_sq", dtype="float32")
            self._add_state(param, "step", initializer=0.0, dtype="float32")
            self._state[param]["param_fp32"] = param.astype("float32")

    def _updates(self, param_group):
        lr = param_group["lr"]
        weight_decay = param_group["weight_decay"]
        eps = param_group["eps"]
        beta0, beta1 = param_group["betas"]
        c1 = tensor(1.0)
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
            step += c1
            fp32_param = states["param_fp32"]
            op = LAMBUpdate(
                beta0,
                beta1,
                step,
                lr,
                weight_decay,
                eps,
                self.bias_correction,
                self.always_adapt,
            )

            new_exp_avg, new_exp_avg_sq, new_param = apply(
                op, exp_avg, exp_avg_sq, fp32_param, grad
            )
            fp32_param._reset(new_param)
            param._reset(new_param.astype("float16"))
            exp_avg._reset(new_exp_avg)
            exp_avg_sq._reset(new_exp_avg_sq)
