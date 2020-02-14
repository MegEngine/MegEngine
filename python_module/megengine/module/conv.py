# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from abc import abstractmethod
from typing import Tuple, Union

import numpy as np

import megengine._internal as mgb

from ..core import Parameter
from ..functional import conv2d
from ..utils.types import _pair, _pair_nonzero
from . import init
from .module import Module


class _ConvNd(Module):
    """base class for convolution modules, including transposed conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        dilation: Union[int, Tuple[int, int]],
        output_padding: Union[int, Tuple[int, int]],
        groups: int,
        bias: bool = True,
    ):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups

        self.weight = Parameter(np.zeros(self._infer_weight_shape(), dtype=np.float32))
        self.bias = None
        if bias:
            self.bias = Parameter(np.zeros(self._infer_bias_shape(), dtype=np.float32))
        self.reset_parameters()

    @abstractmethod
    def _get_fanin(self):
        pass

    def reset_parameters(self) -> None:
        fanin = self._get_fanin()
        std = np.sqrt(1 / fanin)
        init.normal_(self.weight, 0.0, std)
        if self.bias is not None:
            init.zeros_(self.bias)

    @abstractmethod
    def _infer_weight_shape(self):
        pass

    @abstractmethod
    def _infer_bias_shape(self):
        pass


class Conv2d(_ConvNd):
    r"""Applies a 2D convolution over an input tensor.

    For instance, given an input of the size :math:`(N, C_{\text{in}}, H, W)`,
    this layer generates an output of the size
    :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})` through the
    process described as below:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)

    where :math:`\star` is the valid 2D cross-correlation operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.

    When ``groups == in_channels`` and ``out_channels == K * in_channels``,
    where `K` is a positive integer, this operation is also known as depthwise
    convolution.

    In other words, for an input of size :math:`(N, C_{in}, H_{in}, W_{in})`,
    a depthwise convolution with a depthwise multiplier `K`, can be constructed
    by arguments :math:`(in\_channels=C_{in}, out\_channels=C_{in} \times K, ..., groups=C_{in})`.

    :param in_channels: number of input channels.
    :param out_channels: number of output channels.
    :param kernel_size: size of weight on spatial dimensions. If ``kernel_size`` is
        an :class:`int`, the actual kernel size would be
        ``(kernel_size, kernel_size)``. Default: 1
    :param stride: stride of the 2D convolution operation. Default: 1
    :param padding: size of the paddings added to the input on both sides of its
        spatial dimensions. Only zero-padding is supported. Default: 0
    :param dilation: dilation of the 2D convolution operation. Default: 1
    :param groups: number of groups to divide input and output channels into,
        so as to perform a "grouped convolution". When ``groups`` is not 1,
        ``in_channels`` and ``out_channels`` must be divisible by ``groups``,
        and there would be an extra dimension at the beginning of the weight's
        shape. Specifically, the shape of weight would be ``(groups,
        out_channel // groups, in_channels // groups, *kernel_size)``.
    :param bias: wether to add a bias onto the result of convolution. Default:
        True
    :param conv_mode: Supports `CROSS_CORRELATION` or `CONVOLUTION`. Default:
        `CROSS_CORRELATION`.
    :param compute_mode: When set to `DEFAULT`, no special requirements will be
        placed on the precision of intermediate results. When set to `FLOAT32`,
        float32 would be used for accumulator and intermediate result, but only
        effective when input and output are of float16 dtype.
    """

    _conv_mode_type = mgb.opr_param_defs.Convolution.Mode
    _compute_mode_type = mgb.opr_param_defs.Convolution.ComputeMode

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
    ):
        kernel_size = _pair_nonzero(kernel_size)
        stride = _pair_nonzero(stride)
        padding = _pair(padding)
        dilation = _pair_nonzero(dilation)
        self.conv_mode = self._conv_mode_type.convert(conv_mode)
        self.compute_mode = self._compute_mode_type.convert(compute_mode)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            (0, 0),
            groups,
            bias,
        )

    def _get_fanin(self):
        kh, kw = self.kernel_size
        ic = self.in_channels
        return kh * kw * ic

    def _infer_weight_shape(self):
        group = self.groups
        ichl = self.in_channels
        ochl = self.out_channels
        kh, kw = self.kernel_size
        if group == 1:
            # Assume format is NCHW
            return (ochl, ichl, kh, kw)

        assert (
            ichl % group == 0 and ochl % group == 0
        ), "invalid config: input_channels={} output_channels={} group={}".format(
            ichl, ochl, group
        )
        # Assume format is NCHW
        return (group, ochl // group, ichl // group, kh, kw)

    def _infer_bias_shape(self):
        # Assume format is NCHW
        return (1, self.out_channels, 1, 1)

    def forward(self, inp):
        return conv2d(
            inp,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.conv_mode,
            self.compute_mode,
        )
