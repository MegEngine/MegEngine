# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Tuple, Union

from ..core import ones, zeros
from ..functional import add_update, flatten, relu, sqrt, sum, zero_grad
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
        batch_var = (sum2 - sum1 ** 2 / reduce_size) / reduce_size
        return batch_mean, batch_var

    def fold_weight_bias(self, bn_mean, bn_var):
        # get fold bn conv param
        # bn_istd = 1 / bn_std
        # w_fold = gamma / bn_std * W
        # b_fold = gamma * (b - bn_mean) / bn_std + beta
        gamma = self.bn.weight
        if gamma is None:
            gamma = ones((self.bn.num_features), dtype="float32")
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = self.bn.bias
        if beta is None:
            beta = zeros((self.bn.num_features), dtype="float32")
        beta = beta.reshape(1, -1, 1, 1)

        if bn_mean is None:
            bn_mean = zeros((1, self.bn.num_features, 1, 1), dtype="float32")
        if bn_var is None:
            bn_var = ones((1, self.bn.num_features, 1, 1), dtype="float32")

        conv_bias = self.conv.bias
        if conv_bias is None:
            conv_bias = zeros(self.conv._infer_bias_shape(), dtype="float32")

        bn_istd = 1.0 / sqrt(bn_var + self.bn.eps)
        # bn_istd = 1 / bn_std
        # w_fold = gamma / bn_std * W
        scale_factor = gamma * bn_istd
        if self.conv.groups == 1:
            w_fold = self.conv.weight * scale_factor.reshape(-1, 1, 1, 1)
        else:
            w_fold = self.conv.weight * scale_factor.reshape(
                self.conv.groups, -1, 1, 1, 1
            )

        # b_fold = gamma * (b - bn_mean) / bn_std + beta
        b_fold = beta + gamma * (conv_bias - bn_mean) * bn_istd
        return w_fold, b_fold

    def update_running_mean_and_running_var(
        self, bn_mean, bn_var, num_elements_per_channel
    ):
        # update running mean and running var. no grad, use unbiased bn var
        bn_mean = zero_grad(bn_mean)
        bn_var = (
            zero_grad(bn_var)
            * num_elements_per_channel
            / (num_elements_per_channel - 1)
        )
        exponential_average_factor = 1 - self.bn.momentum
        add_update(
            self.bn.running_mean,
            delta=bn_mean,
            alpha=1 - exponential_average_factor,
            beta=exponential_average_factor,
        )
        add_update(
            self.bn.running_var,
            delta=bn_var,
            alpha=1 - exponential_average_factor,
            beta=exponential_average_factor,
        )

    def calc_conv_bn_qat(self, inp, approx=True):
        if self.training and not approx:
            conv = self.conv(inp)
            bn_mean, bn_var = self.get_batch_mean_var(conv)
            num_elements_per_channel = conv.shapeof().prod() / conv.shapeof(1)
            self.update_running_mean_and_running_var(
                bn_mean, bn_var, num_elements_per_channel
            )
        else:
            bn_mean, bn_var = self.bn.running_mean, self.bn.running_var

        # get gamma and beta in BatchNorm
        gamma = self.bn.weight
        if gamma is None:
            gamma = ones((self.bn.num_features), dtype="float32")
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = self.bn.bias
        if beta is None:
            beta = zeros((self.bn.num_features), dtype="float32")
        beta = beta.reshape(1, -1, 1, 1)
        # conv_bias
        conv_bias = self.conv.bias
        if conv_bias is None:
            conv_bias = zeros(self.conv._infer_bias_shape(), dtype="float32")

        bn_istd = 1.0 / sqrt(bn_var + self.bn.eps)
        # bn_istd = 1 / bn_std
        # w_fold = gamma / bn_std * W
        scale_factor = gamma * bn_istd
        if self.conv.groups == 1:
            w_fold = self.conv.weight * scale_factor.reshape(-1, 1, 1, 1)
        else:
            w_fold = self.conv.weight * scale_factor.reshape(
                self.conv.groups, -1, 1, 1, 1
            )
        b_fold = None
        if not (self.training and approx):
            # b_fold = gamma * (conv_bias - bn_mean) / bn_std + beta
            b_fold = beta + gamma * (conv_bias - bn_mean) * bn_istd

        w_qat = self.apply_fakequant_with_observer(
            w_fold, self.weight_fake_quant, self.weight_observer
        )
        conv = self.conv.calc_conv(inp, w_qat, b_fold)
        if not (self.training and approx):
            return conv

        # rescale conv to get original conv output
        orig_conv = conv / scale_factor.reshape(1, -1, 1, 1)
        if self.conv.bias is not None:
            orig_conv = orig_conv + self.conv.bias
        # calculate batch norm
        bn_mean, bn_var = self.get_batch_mean_var(orig_conv)
        bn_istd = 1.0 / sqrt(bn_var + self.bn.eps)
        conv = gamma * bn_istd * (orig_conv - bn_mean) + beta
        num_elements_per_channel = conv.shapeof().prod() / conv.shapeof(1)
        self.update_running_mean_and_running_var(
            bn_mean, bn_var, num_elements_per_channel
        )
        return conv


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
