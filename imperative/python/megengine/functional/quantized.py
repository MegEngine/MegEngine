# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# pylint: disable=too-many-lines
from typing import Tuple, Union

from ..core._imperative_rt.core2 import apply
from ..core.ops import builtin
from ..tensor import Tensor
from ..utils.tuple_function import _pair, _pair_nonzero
from .debug_param import get_execution_strategy


def conv_bias_activation(
    inp: Tensor,
    weight: Tensor,
    bias: Tensor,
    dtype=None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    nonlinear_mode="identity",
    conv_mode="cross_correlation",
    compute_mode="default",
) -> Tensor:
    r"""Convolution bias with activation operation, only for inference.

    Args:
        inp: feature map of the convolution operation.
        weight: convolution kernel.
        bias: bias added to the result of convolution
        stride: stride of the 2D convolution operation. Default: 1
        padding: size of the paddings added to the input on both sides
            of its spatial dimensions. Only zero-padding is supported. Default: 0
        dilation: dilation of the 2D convolution operation. Default: 1
        groups: number of groups into which the input and output channels are divided,
            so as to perform a "grouped convolution". When ``groups`` is not 1,
            ``in_channels`` and ``out_channels`` must be divisible by ``groups``,
            and the shape of weight should be `(groups, out_channel // groups,
            in_channels // groups, height, width)`.
        conv_mode: supports 'cross_correlation' or 'convolution'. Default:
            'cross_correlation'
        dtype: support for ``np.dtype``, Default: np.int8
        compute_mode: when set to "default", no special requirements will be
            placed on the precision of intermediate results. When set to "float32",
            "float32" would be used for accumulator and intermediate result,
            but only effective when input and output are of float16 dtype.
    """
    ph, pw = _pair(padding)
    sh, sw = _pair_nonzero(stride)
    dh, dw = _pair_nonzero(dilation)
    sparse_type = "dense" if groups == 1 else "group"
    op = builtin.ConvBias(
        stride_h=sh,
        stride_w=sw,
        pad_h=ph,
        pad_w=pw,
        dilate_h=dh,
        dilate_w=dw,
        dtype=dtype,
        format="NCHW",
        strategy=get_execution_strategy(),
        nonlineMode=nonlinear_mode,
        mode=conv_mode,
        compute_mode=compute_mode,
        sparse=sparse_type,
    )
    (outputs,) = apply(op, inp, weight, bias)
    return outputs


def batch_conv_bias_activation(
    inp: Tensor,
    weight: Tensor,
    bias: Tensor,
    dtype=None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    nonlinear_mode="identity",
    conv_mode="cross_correlation",
    compute_mode="default",
) -> Tensor:
    r"""Batch convolution bias with activation operation, only for inference.

    Args:
        inp: feature map of the convolution operation.
        weight: convolution kernel in batched way.
        bias: bias added to the result of convolution
        stride: stride of the 2D convolution operation. Default: 1
        padding: size of the paddings added to the input on both sides
            of its spatial dimensions. Only zero-padding is supported. Default: 0
        dilation: dilation of the 2D convolution operation. Default: 1
        groups: number of groups into which the input and output channels are divided,
            so as to perform a "grouped convolution". When ``groups`` is not 1,
            ``in_channels`` and ``out_channels`` must be divisible by ``groups``,
            and the shape of weight should be `(groups, out_channel // groups,
            in_channels // groups, height, width)`.
        conv_mode: supports 'cross_correlation' or 'convolution'. Default:
            'cross_correlation'
        dtype: support for ``np.dtype``, Default: np.int8
        compute_mode: when set to "default", no special requirements will be
            placed on the precision of intermediate results. When set to "float32",
            "float32" would be used for accumulator and intermediate result,
            but only effective when input and output are of float16 dtype.
    """
    ph, pw = _pair(padding)
    sh, sw = _pair_nonzero(stride)
    dh, dw = _pair_nonzero(dilation)
    sparse_type = "dense" if groups == 1 else "group"
    op = builtin.BatchConvBias(
        stride_h=sh,
        stride_w=sw,
        pad_h=ph,
        pad_w=pw,
        dilate_h=dh,
        dilate_w=dw,
        dtype=dtype,
        format="NCHW",
        strategy=get_execution_strategy(),
        nonlineMode=nonlinear_mode,
        mode=conv_mode,
        compute_mode=compute_mode,
        sparse=sparse_type,
    )
    (outputs,) = apply(op, inp, weight, bias)
    return outputs


def conv_transpose2d(
    inp: Tensor,
    weight: Tensor,
    bias: Tensor = None,
    dtype=None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    conv_mode="cross_correlation",
    compute_mode="default",
) -> Tensor:

    assert (
        conv_mode.lower() == "cross_correlation"
        or conv_mode.name == "CROSS_CORRELATION"
    )
    assert compute_mode.lower() == "default" or compute_mode.name == "DEFAULT"

    if groups != 1:
        raise NotImplementedError(
            "group quantized transposed conv2d is not supported yet."
        )
    if bias is not None:
        raise NotImplementedError(
            "bias of quantized transposed conv2d is not supported yet."
        )

    pad_h, pad_w = _pair(padding)
    stride_h, stride_w = _pair_nonzero(stride)
    dilate_h, dilate_w = _pair_nonzero(dilation)

    # should be replaced by Op with bias such as ConvolutionBackwardDataBias
    op = builtin.ConvolutionBackwardData(
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dilate_h=dilate_h,
        dilate_w=dilate_w,
        strategy=get_execution_strategy(),
        dtype=dtype,
        compute_mode=compute_mode,
        mode=conv_mode,
    )
    (output,) = apply(op, weight, inp)
    return output
