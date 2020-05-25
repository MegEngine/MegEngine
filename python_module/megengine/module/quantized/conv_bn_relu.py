# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Tuple, Union

import megengine._internal as mgb

from ... import module as Float
from ...core import Parameter
from ...functional import conv_bias_activation
from ..qat import conv_bn_relu as QAT
from .module import QuantizedModule


class _ConvBnActivation2d(Float.Conv2d, QuantizedModule):
    r"""Applies a 2D convolution over an quantized input tensor, inference only.

    The parameter is same with :class: `~.Conv2d`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        conv_mode: str = "CROSS_CORRELATION",
        compute_mode: str = "DEFAULT",
        dtype=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            True,
            conv_mode,
            compute_mode,
        )
        self.output_dtype = dtype

    def calc_conv_quantized(self, inp, nonlinear_mode="IDENTITY"):
        inp_scale = mgb.dtype.get_scale(inp.dtype)
        w_scale = mgb.dtype.get_scale(self.weight.dtype)
        bias_scale = inp_scale * w_scale
        return conv_bias_activation(
            inp,
            self.weight,
            self.bias.astype(mgb.dtype.qint32(bias_scale)),
            self.output_dtype,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            conv_mode=self.conv_mode,
            compute_mode=self.compute_mode,
            nonlinear_mode=nonlinear_mode,
        )

    @classmethod
    def from_qat_module(cls, qat_module: QAT._ConvBnActivation2d):
        r"""
        return a :class:`~.QuantizedModule` instance converted from a
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
    r"""quantized version of :class:`~.qat.conv_bn_relu.ConvBn2d`."""

    def forward(self, inp):
        return self.calc_conv_quantized(inp, nonlinear_mode="IDENTITY")


class ConvBnRelu2d(_ConvBnActivation2d):
    r"""quantized version of :class:`~.qat.conv_bn_relu.ConvBnRelu2d`."""

    def forward(self, inp):
        return self.calc_conv_quantized(inp, nonlinear_mode="RELU")
