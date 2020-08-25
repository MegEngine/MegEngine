# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ...core import ones, zeros
from ...functional import add_update, relu, sqrt, sum, zero_grad
from ...quantization.utils import fake_quant_bias
from .. import conv_bn as Float
from .module import QATModule


class _ConvBnActivation2d(Float._ConvBnActivation2d, QATModule):
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

        w_fold = self.apply_quant_weight(w_fold)
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

        w_qat = self.apply_quant_weight(w_fold)
        b_qat = fake_quant_bias(b_fold, inp, w_qat)
        conv = self.conv.calc_conv(inp, w_qat, b_qat)
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

    @classmethod
    def from_float_module(cls, float_module: Float._ConvBnActivation2d):
        r"""
        Return a :class:`~.QATModule` instance converted from
        a float :class:`~.Module` instance.
        """
        qat_module = cls(
            float_module.conv.in_channels,
            float_module.conv.out_channels,
            float_module.conv.kernel_size,
            float_module.conv.stride,
            float_module.conv.padding,
            float_module.conv.dilation,
            float_module.conv.groups,
            float_module.conv.bias is not None,
            float_module.conv.conv_mode.name,
            float_module.conv.compute_mode.name,
        )
        qat_module.conv.weight = float_module.conv.weight
        qat_module.conv.bias = float_module.conv.bias
        qat_module.bn = float_module.bn
        return qat_module


class ConvBn2d(_ConvBnActivation2d):
    r"""
    A fused :class:`~.QATModule` including Conv2d, BatchNorm2d with QAT support.
    Could be applied with :class:`~.Observer` and :class:`~.FakeQuantize`.
    """

    def forward(self, inp):
        return self.apply_quant_activation(self.calc_conv_bn_qat(inp))


class ConvBnRelu2d(_ConvBnActivation2d):
    r"""
    A fused :class:`~.QATModule` including Conv2d, BatchNorm2d and relu with QAT support.
    Could be applied with :class:`~.Observer` and :class:`~.FakeQuantize`.
    """

    def forward(self, inp):
        return self.apply_quant_activation(relu(self.calc_conv_bn_qat(inp)))
