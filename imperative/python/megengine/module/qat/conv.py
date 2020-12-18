# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ... import functional as F
from ...quantization.utils import fake_quant_bias
from .. import conv as Float
from .module import QATModule


class Conv2d(Float.Conv2d, QATModule):
    r"""
    A :class:`~.QATModule` Conv2d with QAT support.
    Could be applied with :class:`~.Observer` and :class:`~.FakeQuantize`.
    """

    def calc_conv_qat(self, inp):
        w_qat = self.apply_quant_weight(self.weight)
        if self.weight_fake_quant and self.weight_fake_quant.enabled:
            b_qat = fake_quant_bias(self.bias, inp, w_qat)
        else:
            b_qat = self.bias
        conv = self.calc_conv(inp, w_qat, b_qat)
        return conv

    @classmethod
    def from_float_module(cls, float_module: Float.Conv2d):
        r"""
        Return a :class:`~.QATModule` instance converted from
        a float :class:`~.Module` instance.
        """
        qat_module = cls(
            float_module.in_channels,
            float_module.out_channels,
            float_module.kernel_size,
            float_module.stride,
            float_module.padding,
            float_module.dilation,
            float_module.groups,
            float_module.bias is not None,
            float_module.conv_mode,
            float_module.compute_mode,
        )
        qat_module.weight = float_module.weight
        qat_module.bias = float_module.bias
        return qat_module

    def forward(self, inp):
        return self.apply_quant_activation(self.calc_conv_qat(inp))


class ConvRelu2d(Conv2d):
    r"""
    A :class:`~.QATModule` include Conv2d and Relu with QAT support.
    Could be applied with :class:`~.Observer` and :class:`~.FakeQuantize`.
    """

    def forward(self, inp):
        return self.apply_quant_activation(F.relu(self.calc_conv_qat(inp)))
