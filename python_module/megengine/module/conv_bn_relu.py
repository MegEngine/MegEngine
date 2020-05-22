# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Tuple, Union

from ..core import ones, zeros
from ..functional import flatten, relu, sqrt, sum
from .batchnorm import BatchNorm2d
from .conv import Conv2d
from .module import QATModule


class _ConvBn2d(QATModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        conv_mode: str = "CROSS_CORRELATION",
        compute_mode: str = "DEFAULT",
        eps=1e-5,
        momentum=0.9,
        affine=True,
        track_running_stats=True,
        freeze_bn=False,
    ):
        super().__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            conv_mode,
            compute_mode,
        )
        self.bn = BatchNorm2d(out_channels, eps, momentum, affine, track_running_stats)
        self.freeze_bn = freeze_bn

    def update_bn_stats(self):
        self.freeze_bn = False
        return self

    def freeze_bn_stats(self):
        self.freeze_bn = True
        return self

    def get_bn_gamma_beta(self):
        if self.bn.weight is None:
            gamma = ones((self.bn.num_features), dtype="float32")
        else:
            gamma = self.bn.weight

        if self.bn.bias is None:
            beta = zeros((self.bn.num_features), dtype="float32")
        else:
            beta = self.bn.bias

        return gamma, beta

    def get_batch_mean_var(self, inp):
        def _sum_channel(inp, axis=0, keepdims=True):
            if isinstance(axis, int):
                out = sum(inp, axis=axis, keepdims=keepdims)
            elif isinstance(axis, tuple):
                for idx, elem in enumerate(axis):
                    out = sum(inp if idx == 0 else out, axis=elem, keepdims=keepdims)
            return out

        sum1 = _sum_channel(inp, (0, 2, 3))
        sum2 = _sum_channel(inp ** 2, (0, 2, 3))
        reduce_size = inp.shapeof().prod() / inp.shapeof(1)
        batch_mean = sum1 / reduce_size
        batch_var = (sum2 - sum1 ** 2 / reduce_size) / (reduce_size - 1)

        return batch_mean, batch_var

    def fold_weight_bias(self, bn_mean, bn_var):
        # get fold bn conv param
        # bn_istd = 1 / bn_std
        # w_fold = gamma / bn_std * W
        # b_fold = gamma * (b - bn_mean) / bn_std + beta
        gamma, beta = self.get_bn_gamma_beta()
        b = self.conv.bias
        if b is None:
            b = zeros(self.conv._infer_bias_shape(), dtype="float32")
        if bn_mean is None:
            bn_mean = zeros((1, self.bn.num_features, 1, 1), dtype="float32")
        if bn_var is None:
            bn_var = ones((1, self.bn.num_features, 1, 1), dtype="float32")

        bn_istd = 1.0 / sqrt(bn_var + self.bn.eps)
        if self.conv.groups == 1:
            w_fold = (
                self.conv.weight
                * gamma.reshape(-1, 1, 1, 1)
                * bn_istd.reshape(-1, 1, 1, 1)
            )
        else:
            w_fold = (
                self.conv.weight
                * gamma.reshape(self.conv.groups, -1, 1, 1, 1)
                * bn_istd.reshape(self.conv.groups, -1, 1, 1, 1)
            )
        b_fold = flatten(beta) + (
            flatten(gamma) * (flatten(b) - flatten(bn_mean)) * flatten(bn_istd)
        )
        b_fold = b_fold.reshape(self.conv._infer_bias_shape())

        return w_fold, b_fold

    def calc_conv_bn_qat(self, inp):
        # TODO: use pytorch method as
        conv = self.conv(inp)
        self.bn(conv)

        if self.training:
            bn_mean, bn_var = self.get_batch_mean_var(conv)
        else:
            bn_mean, bn_var = self.bn.running_mean, self.bn.running_var

        w_fold, b_fold = self.fold_weight_bias(bn_mean, bn_var)
        w_qat = self.apply_fakequant_with_observer(
            w_fold, self.weight_fake_quant, self.weight_observer
        )
        return self.conv.calc_conv(inp, w_qat, b_fold)


class ConvBn2d(_ConvBn2d):
    r"""
    A fused :class:`~.QATModule` including Conv2d and BatchNorm2d, supporting ``qat`` mode
    and ``normal`` mode.
    """

    def forward_qat(self, inp):
        return self.apply_fakequant_with_observer(
            self.calc_conv_bn_qat(inp), self.act_fake_quant, self.act_observer
        )

    def forward(self, inp):
        return self.bn(self.conv(inp))


class ConvBnRelu2d(_ConvBn2d):
    r"""
    A fused :class:`~.QATModule` including Conv2d, BatchNorm2d and relu, supporting ``qat``
    mode and ``normal`` mode.
    """

    def forward_qat(self, inp):
        return self.apply_fakequant_with_observer(
            relu(self.calc_conv_bn_qat(inp)), self.act_fake_quant, self.act_observer
        )

    def forward(self, inp):
        return relu(self.bn(self.conv(inp)))
