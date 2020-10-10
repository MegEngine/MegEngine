# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Tuple, Union

import numpy as np

from ... import module as Float
from ...core.tensor import dtype
from ...functional.nn import conv_bias_activation
from ...tensor import Parameter
from ..qat import conv as QAT
from .module import QuantizedModule


class Conv2d(Float.Conv2d, QuantizedModule):
    r"""Quantized version of :class:`~.qat.conv.Conv2d`."""
    r"""Applies a 2D convolution over a quantized input tensor, used for inference only.

    The parameter is same with :class: `~.Conv2d`.
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
        inp_scale = dtype.get_scale(inp.dtype)
        w_scale = dtype.get_scale(self.weight.dtype)
        bias_scale = inp_scale * w_scale
        return conv_bias_activation(
            inp,
            self.weight,
            self.bias.astype(dtype.qint32(bias_scale)),
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
    def from_qat_module(cls, qat_module: QAT.Conv2d):
        r"""
        return a :class:`~.QuantizedModule` instance converted from a
        :class:`~.QATModule` instance.
        """
        output_dtype = qat_module.get_activation_dtype()
        qconv = cls(
            qat_module.in_channels,
            qat_module.out_channels,
            qat_module.kernel_size,
            qat_module.stride,
            qat_module.padding,
            qat_module.dilation,
            qat_module.groups,
            dtype=output_dtype,
        )
        weight = qat_module.weight.astype(qat_module.get_weight_dtype())
        qconv.weight = Parameter(weight.numpy())
        if qat_module.bias is not None:
            qconv.bias = Parameter(qat_module.bias.numpy())
        else:
            qconv.bias = Parameter(
                np.zeros(qat_module._infer_bias_shape(), dtype=np.float32)
            )
        return qconv

    def forward(self, inp):
        return self.calc_conv_quantized(inp, nonlinear_mode="IDENTITY")


class ConvRelu2d(Conv2d):
    r"""Quantized version of :class:`~.qat.conv.ConvRelu2d`."""

    def forward(self, inp):
        return self.calc_conv_quantized(inp, nonlinear_mode="RELU")
