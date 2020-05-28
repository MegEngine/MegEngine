# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Tuple, Union

from ..functional import relu
from .batchnorm import BatchNorm2d
from .conv import Conv2d
from .module import Module


class _ConvBnActivation2d(Module):
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


class ConvBn2d(_ConvBnActivation2d):
    r"""
    A fused :class:`~.Module` including Conv2d, BatchNorm2d. Could be replaced
    with :class:`~.QATModule` version :class:`~.qat.conv_bn.ConvBn2d` using
    :func:`~.quantize.quantize_qat`.
    """

    def forward(self, inp):
        return self.bn(self.conv(inp))


class ConvBnRelu2d(_ConvBnActivation2d):
    r"""
    A fused :class:`~.Module` including Conv2d, BatchNorm2d and relu. Could be replaced
    with :class:`~.QATModule` version :class:`~.qat.conv_bn.ConvBnRelu2d` using
    :func:`~.quantize.quantize_qat`.
    """

    def forward(self, inp):
        return relu(self.bn(self.conv(inp)))
