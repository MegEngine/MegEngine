# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ...tensor import Parameter
from ..qat import conv_bn as QAT
from .conv import Conv2d


class _ConvBnActivation2d(Conv2d):
    r"""
    Applies a 2D convolution over a quantized input tensor, used for inference only.

    The parameter is same with :class: `~.Conv2d`.
    """

    @classmethod
    def from_qat_module(cls, qat_module: QAT._ConvBnActivation2d):
        r"""
        Return a :class:`~.QuantizedModule` instance converted from a
        :class:`~.QATModule` instance.
        """
        output_dtype = qat_module.get_activation_dtype()
        qconv = cls(
            qat_module.conv.in_channels,
            qat_module.conv.out_channels,
            qat_module.conv.kernel_size,
            qat_module.conv.stride,
            qat_module.conv.padding,
            qat_module.conv.dilation,
            qat_module.conv.groups,
            dtype=output_dtype,
        )
        w_fold, b_fold = qat_module.fold_weight_bias(
            qat_module.bn.running_mean, qat_module.bn.running_var
        )
        weight = w_fold.astype(qat_module.get_weight_dtype())
        qconv.weight = Parameter(weight.numpy())
        qconv.bias = Parameter(b_fold.numpy())
        return qconv


class ConvBn2d(_ConvBnActivation2d):
    r"""Quantized version of :class:`~.qat.conv_bn.ConvBn2d`."""

    def forward(self, inp):
        return self.calc_conv_quantized(inp, nonlinear_mode="IDENTITY")


class ConvBnRelu2d(_ConvBnActivation2d):
    r"""Quantized version of :class:`~.qat.conv_bn.ConvBnRelu2d`."""

    def forward(self, inp):
        return self.calc_conv_quantized(inp, nonlinear_mode="RELU")
