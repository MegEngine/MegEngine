# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from functools import partial
from typing import Tuple, Union

import megengine._internal as mgb

from ... import module as Float
from ...core import Parameter
from ...functional import conv_bias_activation
from ...module import Conv2d
from ...quantization.utils import register_method_to_class


class _ConvBnActivation2d(Conv2d):
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
        self.scale = 1.0
        self.zero_point = 0.0
        self.output_dtype = mgb.dtype.qint8(self.scale)
        self.weight = self.weight.astype(self.output_dtype)
        self.bias = self.bias.astype(mgb.dtype.qint32(self.scale))

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


class ConvBn2d(_ConvBnActivation2d):
    def forward(self, inp):
        if self.training:
            raise ValueError("quantized module only support inference.")
        return self.calc_conv_quantized(inp, nonlinear_mode="IDENTITY")


class ConvBnRelu2d(_ConvBnActivation2d):
    def forward(self, inp):
        if self.training:
            raise ValueError("quantized module only support inference.")
        return self.calc_conv_quantized(inp, nonlinear_mode="RELU")


def to_quantized(quantized_class, float_module):
    qconv = quantized_class(
        float_module.conv.in_channels,
        float_module.conv.out_channels,
        float_module.conv.kernel_size,
        float_module.conv.stride,
        float_module.conv.padding,
        float_module.conv.dilation,
        float_module.conv.groups,
    )
    w_fold, b_fold = float_module.fold_weight_bias(
        float_module.bn.running_mean, float_module.bn.running_var
    )
    weight = w_fold.astype(float_module.weight_observer.get_dtype())
    qconv.output_dtype = float_module.act_observer.get_dtype()
    qconv.weight = Parameter(weight.numpy())
    qconv.bias = Parameter(b_fold.numpy())
    qconv.scale, qconv.zero_point = float_module.act_observer.get_qparams()

    return qconv


# replace :class:`~.module.QATModule`'s ``to_quantized`` method.
# implemented here to avoid circular import.
register_method_to_class(Float.ConvBn2d)(partial(to_quantized, ConvBn2d))
register_method_to_class(Float.ConvBnRelu2d)(partial(to_quantized, ConvBnRelu2d))
