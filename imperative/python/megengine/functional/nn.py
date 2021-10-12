# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# pylint: disable=too-many-lines
from functools import lru_cache
from typing import NamedTuple, Optional, Sequence, Tuple, Union

from ..core._imperative_rt.core2 import apply, dtype_promotion
from ..core._imperative_rt.ops import SubgraphBuilder as _SubgraphBuilder
from ..core.ops import builtin
from ..core.ops.builtin import (
    BatchNorm,
    Dimshuffle,
    Elemwise,
    GetVarShape,
    Identity,
    Reduce,
    Reshape,
    TypeCvt,
)
from ..core.ops.special import Const
from ..core.tensor import amp, megbrain_graph
from ..core.tensor.array_method import _elwise_apply
from ..core.tensor.utils import (
    astensor1d,
    astype,
    cast_tensors,
    convert_single_value,
    make_shape_tuple,
    setscalar,
    subgraph,
)
from ..device import get_default_device
from ..distributed import WORLD, is_distributed
from ..jit import exclude_from_trace
from ..random import uniform
from ..tensor import Tensor
from ..utils.deprecation import deprecated_func
from ..utils.tuple_function import _pair, _pair_nonzero, _triple, _triple_nonzero
from .debug_param import get_execution_strategy
from .distributed import all_reduce_sum
from .elemwise import _elwise, exp, log, log1p, maximum, minimum
from .math import matmul, max, sum
from .tensor import broadcast_to, concat, expand_dims, ones, squeeze, zeros

__all__ = [
    "adaptive_avg_pool2d",
    "adaptive_max_pool2d",
    "avg_pool2d",
    "batch_norm",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose2d",
    "conv_transpose3d",
    "deformable_conv2d",
    "deformable_psroi_pooling",
    "dropout",
    "embedding",
    "gelu",
    "hsigmoid",
    "hswish",
    "indexing_one_hot",
    "leaky_relu",
    "linear",
    "local_conv2d",
    "local_response_norm",
    "logsigmoid",
    "logsumexp",
    "logsoftmax",
    "max_pool2d",
    "one_hot",
    "prelu",
    "relu",
    "relu6",
    "remap",
    "sigmoid",
    "sliding_window",
    "sliding_window_transpose",
    "silu",
    "softmax",
    "softplus",
    "sync_batch_norm",
    "warp_affine",
    "warp_perspective",
    "pixel_shuffle",
]


def expand_hw(x):
    # NOTE: >1d array is accepted, as long as 1 <= size <= 2
    try:
        x = int(x)
        return [x, x]
    except (TypeError, ValueError):
        pass
    h, w = x
    return int(h), int(w)


def linear(
    inp: Tensor, weight: Tensor, bias: Optional[Tensor] = None, compute_mode="default",
) -> Tensor:
    r"""Applies a linear transformation to the input tensor.

    Refer to :class:`~.module.linear.Linear` for more information.

    Args:
        inp: input tensor with shape `(N, in_features)`.
        weight: weight with shape `(out_features, in_features)`.
        bias: bias with shape `(out_features,)`. Default: None
    """
    ret = matmul(inp, weight, transpose_b=True, compute_mode=compute_mode)
    if bias is not None:
        if amp._enabled:
            bias = bias.astype("float16")
        ret += bias
    return ret


def conv1d(
    inp: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    conv_mode="cross_correlation",
    compute_mode="default",
) -> Tensor:
    r"""1D convolution operation.

    Refer to :class:`~.Conv1d` for more information.

    Args:
        inp: The feature map of the convolution operation
        weight: The convolution kernel.
        bias: The bias added to the result of convolution (if given)
        stride: Stride of the 1D convolution operation. Default: 1
        padding: Size of the paddings added to the input on both sides of its
            spatial dimensions. Only zero-padding is supported. Default: 0
        dilation: Dilation of the 1D convolution operation. Default: 1
        groups: number of groups to divide input and output channels into,
            so as to perform a "grouped convolution". When ``groups`` is not 1,
            ``in_channels`` and ``out_channels`` must be divisible by ``groups``,
            and the shape of weight should be ``(groups, out_channel // groups,
            in_channels // groups, kernel_size)``. Default: 1
        conv_mode: Supports 'cross_correlation'. Default:
            'cross_correlation'.
        compute_mode: When set to 'default', no special requirements will be
            placed on the precision of intermediate results. When set to 'float32',
            float32 would be used for accumulator and intermediate result, but only
            effective when input and output are of float16 dtype.
    """
    assert (
        conv_mode.lower() == "cross_correlation"
        or conv_mode.name == "CROSS_CORRELATION"
    )
    assert compute_mode.lower() == "default" or compute_mode.name == "DEFAULT"
    assert inp.ndim == 3, "the input dimension of conv1d should be 3"
    assert weight.ndim == 3, "the weight dimension of conv1d should be 3"
    if amp._enabled:
        compute_mode = "float32"
        inp, weight, bias = cast_tensors(inp, weight, bias)
    else:
        dtype = dtype_promotion(inp, weight)
        if inp.dtype != dtype:
            inp = inp.astype(dtype)
        if weight.dtype != dtype:
            weight = weight.astype(dtype)

    inp = expand_dims(inp, 3)
    weight = expand_dims(weight, 3)
    if bias is not None:
        assert bias.ndim == 3, "the bias dimension of conv1d should be 3"
        bias = expand_dims(bias, 3)

    stride_h = stride
    pad_h = padding
    dilate_h = dilation

    sparse_type = "dense" if groups == 1 else "group"
    op = builtin.Convolution(
        stride_h=stride_h,
        stride_w=1,
        pad_h=pad_h,
        pad_w=0,
        dilate_h=dilate_h,
        dilate_w=1,
        strategy=get_execution_strategy(),
        mode=conv_mode,
        compute_mode=compute_mode,
        sparse=sparse_type,
    )
    (output,) = apply(op, inp, weight)
    if bias is not None:
        output += bias
    output = squeeze(output, 3)
    return output


def conv2d(
    inp: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    conv_mode="cross_correlation",
    compute_mode="default",
) -> Tensor:
    r"""2D convolution operation.

    Refer to :class:`~.module.Conv2d` for more information.

    Args:
        inp: feature map of the convolution operation.
        weight: convolution kernel.
        bias: bias added to the result of convolution (if given).
        stride: stride of the 2D convolution operation. Default: 1
        padding: size of the paddings added to the input on both sides of its
            spatial dimensions. Only zero-padding is supported. Default: 0
        dilation: dilation of the 2D convolution operation. Default: 1
        groups: number of groups into which the input and output channels are divided,
            so as to perform a ``grouped convolution``. When ``groups`` is not 1,
            ``in_channels`` and ``out_channels`` must be divisible by ``groups``,
            and the shape of weight should be ``(groups, out_channel // groups,
            in_channels // groups, height, width)``. Default: 1
        conv_mode: supports "cross_correlation". Default: "cross_correlation"
        compute_mode: when set to "default", no special requirements will be
            placed on the precision of intermediate results. When set to "float32",
            "float32" would be used for accumulator and intermediate result, but only
            effective when input and output are of float16 dtype.

    Returns:
        output tensor.
    """
    assert (
        conv_mode.lower() == "cross_correlation"
        or conv_mode.name == "CROSS_CORRELATION"
    )
    if amp._enabled:
        compute_mode = "float32"
        inp, weight, bias = cast_tensors(inp, weight, bias)
    else:
        dtype = dtype_promotion(inp, weight)
        if inp.dtype != dtype:
            inp = inp.astype(dtype)
        if weight.dtype != dtype:
            weight = weight.astype(dtype)

    stride_h, stride_w = expand_hw(stride)
    pad_h, pad_w = expand_hw(padding)
    dilate_h, dilate_w = expand_hw(dilation)

    sparse_type = "dense" if groups == 1 else "group"
    op = builtin.Convolution(
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dilate_h=dilate_h,
        dilate_w=dilate_w,
        strategy=get_execution_strategy(),
        mode=conv_mode,
        compute_mode=compute_mode,
        sparse=sparse_type,
    )
    (output,) = apply(op, inp, weight)
    if bias is not None:
        output += bias
    return output


def conv3d(
    inp: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int, int]] = 1,
    padding: Union[int, Tuple[int, int, int]] = 0,
    dilation: Union[int, Tuple[int, int, int]] = 1,
    groups: int = 1,
    conv_mode: str = "cross_correlation",
) -> Tensor:
    r"""3D convolution operation.

    Refer to :class:`~.Conv3d` for more information.

    Args:
        inp: feature map of the convolution operation.
        weight: convolution kernel.
        bias: bias added to the result of convolution (if given).
        stride: stride of the 3D convolution operation. Default: 1
        padding: size of the paddings added to the input on both sides of its
            spatial dimensions. Only zero-padding is supported. Default: 0
        dilation: dilation of the 3D convolution operation. Default: 1
        groups: number of groups into which the input and output channels are divided,
            so as to perform a ``grouped convolution``. When ``groups`` is not 1,
            ``in_channels`` and ``out_channels`` must be divisible by ``groups``,
            and the shape of weight should be ``(groups, out_channel // groups,
            in_channels // groups, depth, height, width)``. Default: 1
        conv_mode: supports "cross_correlation". Default: "cross_correlation"

    Returns:
        output tensor.
    """
    assert conv_mode.lower() == "cross_correlation"

    D, H, W = 0, 1, 2

    pad = _triple(padding)
    stride = _triple_nonzero(stride)
    dilate = _triple_nonzero(dilation)

    dtype = dtype_promotion(inp, weight)
    if inp.dtype != dtype:
        inp = inp.astype(dtype)
    if weight.dtype != dtype:
        weight = weight.astype(dtype)

    sparse_type = "dense" if groups == 1 else "group"
    op = builtin.Convolution3D(
        pad_d=pad[D],
        pad_h=pad[H],
        pad_w=pad[W],
        stride_d=stride[D],
        stride_h=stride[H],
        stride_w=stride[W],
        dilate_d=dilate[D],
        dilate_h=dilate[H],
        dilate_w=dilate[W],
        strategy=get_execution_strategy(),
        mode=conv_mode,
        sparse=sparse_type,
    )
    (output,) = apply(op, inp, weight)
    if bias is not None:
        output += bias
    return output


def conv_transpose2d(
    inp: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    conv_mode="cross_correlation",
    compute_mode="default",
) -> Tensor:
    r"""2D transposed convolution operation.

    Refer to :class:`~.ConvTranspose2d` for more information.

    Args:
        inp: feature map of the convolution operation.
        weight: convolution kernel.
        bias: bias added to the result of convolution (if given).
        stride: stride of the 2D convolution operation. Default: 1
        padding: size of the paddings added to the input on both sides of its
            spatial dimensions. Only zero-padding is supported. Default: 0
        dilation: dilation of the 2D convolution operation. Default: 1
        groups: number of groups into which the input and output channels are divided,
            so as to perform a ``grouped convolution``. When ``groups`` is not 1,
            ``in_channels`` and ``out_channels`` must be divisible by groups,
            and the shape of weight should be ``(groups, in_channels // groups,
            out_channels // groups, height, width)``. Default: 1
        conv_mode: supports "cross_correlation". Default: "cross_correlation"
        compute_mode: when set to "default", no special requirements will be
            placed on the precision of intermediate results. When set to "float32",
            "float32" would be used for accumulator and intermediate result, but only
            effective when input and output are of float16 dtype.

    Returns:
        output tensor.
    """
    assert (
        conv_mode.lower() == "cross_correlation"
        or conv_mode.name == "CROSS_CORRELATION"
    )
    if amp._enabled:
        compute_mode = "float32"
        inp, weight, bias = cast_tensors(inp, weight, bias)
    else:
        dtype = dtype_promotion(inp, weight)
        if inp.dtype != dtype:
            inp = inp.astype(dtype)
        if weight.dtype != dtype:
            weight = weight.astype(dtype)

    if groups != 1:
        raise NotImplementedError("group transposed conv2d is not supported yet.")

    stride_h, stride_w = expand_hw(stride)
    pad_h, pad_w = expand_hw(padding)
    dilate_h, dilate_w = expand_hw(dilation)

    op = builtin.ConvolutionBackwardData(
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dilate_h=dilate_h,
        dilate_w=dilate_w,
        strategy=get_execution_strategy(),
        compute_mode=compute_mode,
    )
    (output,) = apply(op, weight, inp)
    if bias is not None:
        output += bias
    return output


def deformable_conv2d(
    inp: Tensor,
    weight: Tensor,
    offset: Tensor,
    mask: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    conv_mode="cross_correlation",
    compute_mode="default",
) -> Tensor:
    r"""Deformable Convolution.

    Args:
        inp: input feature map.
        weight: convolution kernel.
        offset: input offset to kernel, channel of this tensor should match the deformable settings.
        mask: input mask to kernel, channel of this tensor should match the deformable settings.
        bias: bias added to the result of convolution (if given).
        stride: stride of the 2D convolution operation. Default: 1
        padding: size of the paddings added to the input on both sides of its
            spatial dimensions. Only zero-padding is supported. Default: 0
        dilation: dilation of the 2D convolution operation. Default: 1
        groups: number of groups into which the input and output channels are divided,
            so as to perform a ``grouped convolution``. When ``groups`` is not 1,
            ``in_channels`` and ``out_channels`` must be divisible by groups,
            and the shape of weight should be ``(groups, out_channel // groups,
            in_channels // groups, height, width)``. Default: 1
        conv_mode: supports "cross_correlation". Default: "cross_correlation"
        compute_mode: when set to "default", no special requirements will be
            placed on the precision of intermediate results. When set to "float32",
            "float32" would be used for accumulator and intermediate result, but only
            effective when input and output are of float16 dtype.

    Returns:
        output tensor.
    """
    assert (
        conv_mode.lower() == "cross_correlation"
        or conv_mode.name == "CROSS_CORRELATION"
    )
    if amp._enabled:
        compute_mode = "float32"
        inp, weight, offset, mask, bias = cast_tensors(inp, weight, offset, mask, bias)
    else:
        offset = offset.astype("float32")
        mask = mask.astype("float32")

    stride_h, stride_w = expand_hw(stride)
    pad_h, pad_w = expand_hw(padding)
    dilate_h, dilate_w = expand_hw(dilation)

    sparse_type = "dense" if groups == 1 else "group"
    op = builtin.DeformableConv(
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dilate_h=dilate_h,
        dilate_w=dilate_w,
        strategy=get_execution_strategy(),
        mode=conv_mode,
        compute_mode=compute_mode,
        sparse=sparse_type,
    )
    (output,) = apply(op, inp, weight, offset, mask)
    if bias is not None:
        output += bias
    return output


def local_conv2d(
    inp: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    conv_mode="cross_correlation",
):
    r"""Applies spatial 2D convolution over an groupped channeled image with untied kernels."""
    assert (
        conv_mode.lower() == "cross_correlation"
        or conv_mode.name == "CROSS_CORRELATION"
    )

    stride_h, stride_w = expand_hw(stride)
    pad_h, pad_w = expand_hw(padding)
    dilate_h, dilate_w = expand_hw(dilation)

    dtype = dtype_promotion(inp, weight)
    if inp.dtype != dtype:
        inp = inp.astype(dtype)
    if weight.dtype != dtype:
        weight = weight.astype(dtype)

    op = builtin.GroupLocal(
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dilate_h=dilate_h,
        dilate_w=dilate_w,
        mode=conv_mode,
        sparse="dense",
    )
    (output,) = apply(op, inp, weight)
    if bias is not None:
        output += bias
    return output


def conv_transpose3d(
    inp: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int, int]] = 1,
    padding: Union[int, Tuple[int, int, int]] = 0,
    dilation: Union[int, Tuple[int, int, int]] = 1,
) -> Tensor:
    r"""3D transposed convolution operation. Only support the case that groups = 1
    and conv_mode = "cross_correlation".

    Refer to :class:`~.ConvTranspose3d` for more information.

    Args:
        inp: feature map of the convolution operation.
        weight: convolution kernel.
            weight usually has shape ``(in_channels, out_channels, depth, height, width)``.
        bias: bias added to the result of convolution (if given).
        stride: stride of the 3D convolution operation. Default: 1
        padding: size of the paddings added to the input on all sides of its
            spatial dimensions. Only zero-padding is supported. Default: 0
        dilation: dilation of the 3D convolution operation. Default: 1

    Returns:
        output tensor.
    """
    D, H, W = 0, 1, 2
    pad = _triple(padding)
    stride = _triple_nonzero(stride)
    dilate = _triple_nonzero(dilation)

    dtype = dtype_promotion(inp, weight)
    if inp.dtype != dtype:
        inp = inp.astype(dtype)
    if weight.dtype != dtype:
        weight = weight.astype(dtype)

    op = builtin.Convolution3DBackwardData(
        pad_d=pad[D],
        pad_h=pad[H],
        pad_w=pad[W],
        stride_d=stride[D],
        stride_h=stride[H],
        stride_w=stride[W],
        dilate_d=dilate[D],
        dilate_h=dilate[H],
        dilate_w=dilate[W],
        strategy=get_execution_strategy(),
    )
    (output,) = apply(op, weight, inp)
    if bias is not None:
        output += bias
    return output


def max_pool2d(
    inp: Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
) -> Tensor:
    r"""Applies a 2D max pooling over an input tensor.

    Refer to :class:`~.MaxPool2d` for more information.

    Args:
        inp: input tensor.
        kernel_size: size of the window.
        stride: stride of the window. If not provided, its value is set to kernel_size.
            Default: None
        padding: implicit zero padding added on both sides. Default: 0

    Returns:
        output tensor.
    """
    if stride is None:
        stride = kernel_size
    window_h, window_w = _pair_nonzero(kernel_size)
    stride_h, stride_w = _pair_nonzero(stride)
    padding_h, padding_w = _pair(padding)

    op = builtin.Pooling(
        window_h=window_h,
        window_w=window_w,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=padding_h,
        pad_w=padding_w,
        mode="max",
    )
    (output,) = apply(op, inp)
    return output


def avg_pool2d(
    inp: Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    mode: str = "average_count_exclude_padding",
) -> Tensor:
    r"""Applies 2D average pooling over an input tensor.

    Refer to :class:`~.AvgPool2d` for more information.

    Args:
        inp: input tensor.
        kernel_size: size of the window.
        stride: stride of the window. If not provided, its value is set to ``kernel_size``.
            Default: None
        padding: implicit zero padding added on both sides. Default: 0
        mode: whether to count padding values, set to "average" will do counting.
            Default: "average_count_exclude_padding"

    Returns:
        output tensor.
    """
    if stride is None:
        stride = kernel_size
    window_h, window_w = _pair_nonzero(kernel_size)
    stride_h, stride_w = _pair_nonzero(stride)
    padding_h, padding_w = _pair(padding)

    op = builtin.Pooling(
        window_h=window_h,
        window_w=window_w,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=padding_h,
        pad_w=padding_w,
        mode=mode,
    )
    (output,) = apply(op, inp)
    return output


def adaptive_max_pool2d(
    inp: Tensor, oshp: Union[Tuple[int, int], int, Tensor],
) -> Tensor:
    r"""Applies a 2D max adaptive pooling over an input.

    Refer to :class:`~.MaxAdaptivePool2d` for more information.

    Args:
        inp: input tensor.
        oshp: OH, OW)` size of the output shape.

    Returns:
        output tensor.
    """
    if isinstance(oshp, int):
        oshp = (oshp, oshp)

    op = builtin.AdaptivePooling(mode="max", format="NCHW",)
    oshp = astensor1d(oshp, inp, dtype="int32", device=inp.device)
    (output,) = apply(op, inp, oshp)
    return output


def adaptive_avg_pool2d(
    inp: Tensor, oshp: Union[Tuple[int, int], int, Tensor],
) -> Tensor:
    r"""Applies a 2D average adaptive pooling over an input.

    Refer to :class:`~.AvgAdaptivePool2d` for more information.

    Args:
        inp: input tensor.
        oshp: OH, OW)` size of the output shape.

    Returns:
        output tensor.
    """
    if isinstance(oshp, int):
        oshp = (oshp, oshp)

    op = builtin.AdaptivePooling(mode="average", format="NCHW",)
    oshp = astensor1d(oshp, inp, dtype="int32", device=inp.device)
    (output,) = apply(op, inp, oshp)
    return output


def deformable_psroi_pooling(
    inp: Tensor,
    rois: Tensor,
    trans: Tensor,
    no_trans: bool,
    part_size: int,
    pooled_h: int,
    pooled_w: int,
    sample_per_part: int,
    spatial_scale: float,
    trans_std: float = 0.1,
):
    r"""Deformable PSROI(Position Sensitive Region of Interest) Pooling.

    Args:
        inp: input feature map.
        rois: the rois for feature pooling.
        trans: input offset to psroi_pooling.
        no_trans: check the phase of DeformablePSROIPooling. False to the
            1st phase, True to the 2nd phase.
        part_size: part size.
        sample_per_part: sample points of each part.
        pooled_shape: kernel shape of convolution.
        spatial_scale: the spatial_scale w.r.t input image.
        trans_std: multiplier used in 2nd phase.
    """
    op = builtin.DeformablePSROIPooling(
        no_trans=no_trans,
        part_size=part_size,
        pooled_h=pooled_h,
        pooled_w=pooled_w,
        sample_per_part=sample_per_part,
        spatial_scale=spatial_scale,
        trans_std=trans_std,
    )
    output, _ = apply(op, inp, rois, trans)
    return output


def hswish(x):
    r"""Element-wise `x * relu6(x + 3) / 6`.

    Example:

        .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            x = tensor(np.arange(5).astype(np.float32))
            out = F.hswish(x)
            print(out.numpy().round(decimals=4))

        .. testoutput::

            [0.     0.6667 1.6667 3.     4.    ]

    """
    return _elwise(x, mode=Elemwise.Mode.H_SWISH)


def sigmoid(x):
    r"""Element-wise `1 / ( 1 + exp( -x ) )`."""
    return _elwise(x, mode=Elemwise.Mode.SIGMOID)


def hsigmoid(x):
    r"""Element-wise `relu6(x + 3) / 6`."""
    return relu6(x + 3) / 6


def relu(x):
    r"""Element-wise `max(x, 0)`."""
    return _elwise(x, mode=Elemwise.Mode.RELU)


def relu6(x):
    r"""Element-wise `min(max(x, 0), 6)`."""
    return minimum(maximum(x, 0), 6)


def prelu(inp: Tensor, weight: Tensor) -> Tensor:
    r"""Elememt-wise PReLU function.

    Refer to :class:`~.PReLU` for more information.
    """
    return maximum(inp, 0) + weight * minimum(inp, 0)


def leaky_relu(inp: Tensor, negative_slope: float = 0.01) -> Tensor:
    r"""Element-wose LeakyReLU function

    Refer to :class:`~.LeakyReLU` for more information.
    """
    return maximum(inp, 0) + negative_slope * minimum(inp, 0)


def silu(x):
    r"""Applies the element-wise Sigmoid Linear Unit function, i.e. `x * sigmoid(x)`."""
    return _elwise(x, mode=Elemwise.Mode.SILU)


def gelu(x):
    r"""Applies the element-wise function:

    .. math::
        \text{gelu}(x) = x\Phi(x)

    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.
    """
    return _elwise(x, mode=Elemwise.Mode.GELU)


def softplus(inp: Tensor) -> Tensor:
    r"""Applies the element-wise function:

    .. math::
        \text{softplus}(x) = \log(1 + \exp(x))

    softplus is a smooth approximation to the ReLU function and can be used
    to constrain the output to be always positive.
    For numerical stability the implementation follows this transformation:

    .. math::
        \text{softplus}(x) = \log(1 + \exp(x))
                           = \log(1 + \exp(-\text{abs}(x))) + \max(x, 0)
                           = \log1p(\exp(-\text{abs}(x))) + \text{relu}(x)

   Examples:

        .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            x = tensor(np.arange(-3, 3, dtype=np.float32))
            y = F.softplus(x)
            print(y.numpy().round(decimals=4))

        Outputs:

        .. testoutput::

            [0.0486 0.1269 0.3133 0.6931 1.3133 2.1269]
    """
    return log1p(exp(-abs(inp))) + relu(inp)


def logsoftmax(inp: Tensor, axis: Union[int, Sequence[int]]) -> Tensor:
    r"""Applies the :math:`\log(\text{softmax}(x))` function to an n-dimensional
    input tensor. The :math:`\text{logsoftmax}(x)` formulation can be simplified as:

    .. math::
        \text{logsoftmax}(x_{i}) = \log(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} )

    For numerical stability the implementation follows this transformation:

    .. math::
        \text{logsoftmax}(x)
        = \log (\frac{\exp (x)}{\sum_{i}(\exp (x_{i}))})
        = x - \log (\sum_{i}(\exp (x_{i})))
        = x - \text{logsumexp}(x)

    Examples:

        .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            x = tensor(np.arange(-5, 5, dtype=np.float32)).reshape(2,5)
            y = F.logsoftmax(x, axis=1)
            print(y.numpy().round(decimals=4))

        Outputs:

        .. testoutput::

            [[-4.4519 -3.4519 -2.4519 -1.4519 -0.4519]
             [-4.4519 -3.4519 -2.4519 -1.4519 -0.4519]]
    """
    return inp - logsumexp(inp, axis, keepdims=True)


def logsigmoid(inp: Tensor) -> Tensor:
    r"""Applies the element-wise function:

    .. math::
        \text{logsigmoid}(x) = \log(\frac{ 1 }{ 1 + \exp(-x)})
        = \log(1/(1 + \exp(-x)))
        = - \log(1 + \exp(-x))
        = - \text{softplus}(-x)

    Examples:

        .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            x = tensor(np.arange(-5, 5, dtype=np.float32))
            y = F.logsigmoid(x)
            print(y.numpy().round(decimals=4))

        Outputs:

        .. testoutput::

            [-5.0067 -4.0182 -3.0486 -2.1269 -1.3133 -0.6931 -0.3133 -0.1269 -0.0486
            -0.0181]
    """
    return -softplus(-inp)


def logsumexp(
    inp: Tensor, axis: Union[int, Sequence[int]], keepdims: bool = False
) -> Tensor:
    r"""Calculates the logarithm of the inputs' exponential sum along the given :attr:`axis`.

    .. math::

        \text{logsumexp}(x)= \log \sum_{j=1}^{n} \exp \left(x_{j}\right)

    For numerical stability, the implementation follows this transformation:

    .. math::

        \text{logsumexp}(x)= \log \sum_{j=1}^{n} \exp \left(x_{j}\right)
        = \text{logsumexp}(x)=b+\log \sum_{j=1}^{n} \exp \left(x_{j}-b\right)

    where

    .. math::
        b = \max(x_j)

    Examples:

        .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            x = tensor(np.arange(-5, 5, dtype=np.float32)).reshape(2,5)
            y = F.logsumexp(x, axis=1, keepdims=False)
            print(y.numpy().round(decimals=4))

        Outputs:

        .. testoutput::

            [-0.5481  4.4519]
    """
    max_value = max(inp.detach(), axis, keepdims=True)
    if keepdims:
        return max_value + log(sum(exp(inp - max_value), axis, keepdims))
    else:
        return squeeze(max_value, axis=None) + log(
            sum(exp(inp - max_value), axis, keepdims)
        )


def _get_softmax_axis(ndim: int) -> int:
    if ndim in (0, 1, 3):
        return 0
    return 1


def softmax(inp: Tensor, axis: Optional[int] = None) -> Tensor:
    r"""Applies a :math:`\text{softmax}(x)` function. :math:`\text{softmax}(x)` is defined as:

    .. math::
            \text{softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    It is applied to all elements along axis, and rescales elements so that
    they stay in the range `[0, 1]` and sum to 1.

    See :class:`~.module.Softmax` for more details.

    Examples:

        .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            x = tensor(np.arange(-5, 5, dtype=np.float32)).reshape(2,5)
            out = F.softmax(x)
            print(out.numpy().round(decimals=4))

        Outputs:

        .. testoutput::

            [[0.0117 0.0317 0.0861 0.2341 0.6364]
            [0.0117 0.0317 0.0861 0.2341 0.6364]]
    """
    if axis is None:
        axis = _get_softmax_axis(len(inp.shape))
    offset = inp.max(axis=axis, keepdims=True).detach()
    cached = exp(inp - offset)
    down = sum(cached, axis=axis, keepdims=True)
    return cached / down


@lru_cache(maxsize=None)
def _get_layerNorm(device, dtype, dim, gopt_level=2):
    @subgraph("LayerNormAffine", dtype, device, 5, gopt_level=gopt_level)
    def layerNormAffine(inputs, f, c):
        inp, eps, _flatten_shape, weight, bias = inputs
        inp_shape = f(GetVarShape(), inp)

        inp = f(Reshape(axis=dim), inp, _flatten_shape)
        mean = f(Reduce(mode="mean", axis=-1), inp)
        x2s = f(Reduce(mode="sum_sqr", axis=-1), inp)
        reduce_shape = f(GetVarShape(), x2s)
        reduce_size = f(
            "//",
            f(Reduce(mode="product", axis=0), inp_shape),
            f(Reduce(mode="product", axis=0), reduce_shape),
        )
        reduce_size_f = f(TypeCvt(dtype=dtype), reduce_size)
        var = f("-", f("/", x2s, reduce_size_f), f("**", mean, c(2)))
        inv_sqrt_var = f("**", f("+", var, eps), c(-0.5))
        oup = f("fma3", inp, inv_sqrt_var, f("*", f("-", mean), inv_sqrt_var))
        affine_oup = f(Reshape(), oup, inp_shape)
        affine_oup = f("fma3", affine_oup, weight, bias)

        # NOTE: return oup make backward faster but take more memory
        return (affine_oup, oup, mean, x2s), (True, False, False, False)

    @subgraph("LayerNorm", dtype, device, 3, gopt_level=gopt_level)
    def layerNorm(inputs, f, c):
        inp, eps, _flatten_shape = inputs
        inp_shape = f(GetVarShape(), inp)

        inp = f(Reshape(axis=dim), inp, _flatten_shape)
        mean = f(Reduce(mode="mean", axis=-1), inp)
        x2s = f(Reduce(mode="sum_sqr", axis=-1), inp)
        reduce_shape = f(GetVarShape(), x2s)
        reduce_size = f(
            "//",
            f(Reduce(mode="product", axis=0), inp_shape),
            f(Reduce(mode="product", axis=0), reduce_shape),
        )
        reduce_size_f = f(TypeCvt(dtype=dtype), reduce_size)
        var = f("-", f("/", x2s, reduce_size_f), f("**", mean, c(2)))
        inv_sqrt_var = f("**", f("+", var, eps), c(-0.5))
        oup = f("fma3", inp, inv_sqrt_var, f("*", f("-", mean), inv_sqrt_var))
        oup = f(Reshape(), oup, inp_shape)

        return (oup,), (True,)

    return (layerNorm, layerNormAffine)


def layer_norm(
    inp: Tensor,
    normalized_shape: tuple,
    affine: bool,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
):
    r"""Applies layer normalization to the input. Support tensor of any shape as input.
    Reference: https://arxiv.org/pdf/1803.08494.pdf.
    
    Args:
        inp: input tensor.
        normalized_shape: the shape that you want to be normalizated 
        affine: whether to use weight and bias
        weight: must not be None when the affine is true
        bias: must not be None when the bias is true
        eps: a value added to the denominator for numerical stability. Default: 1e-5
    """

    if amp._enabled:
        inp, weight, bias = cast_tensors(inp, weight, bias, promote=True)

    _device = inp.device
    _dtype = inp.dtype
    _dim = len(inp.shape) - len(normalized_shape)

    _flatten_shape = concat(
        (
            convert_single_value(inp.shape[:_dim], dtype="int32", device=inp.device),
            convert_single_value(-1, dtype="int32", device=inp.device),
        )
    )
    (layerNorm, layerNormAffine) = _get_layerNorm(_device, _dtype, _dim)

    eps = convert_single_value(eps, dtype=inp.dtype, device=inp.device)
    if affine:
        outvar, *_ = apply(layerNormAffine(), inp, eps, _flatten_shape, weight, bias)
    else:
        outvar, *_ = apply(layerNorm(), inp, eps, _flatten_shape)

    return outvar


def batch_norm(
    inp: Tensor,
    running_mean: Tensor = None,
    running_var: Tensor = None,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    *,
    training: bool = False,
    momentum: float = 0.9,
    eps: float = 1e-5,
    inplace: bool = True,
    compute_mode="default",
    param_dim="dim_1c11"
):
    r"""Applies batch normalization to the input.

    Refer to :class:`~.BatchNorm2d` and :class:`~.BatchNorm1d` for more information.

    Args:
        inp: input tensor.
        running_mean: tensor to store running mean.
        running_var: tensor to store running variance.
        weight: scaling tensor in the learnable affine parameters.
            See :math:`\gamma` in :class:`~.BatchNorm2d`.
        bias: bias tensor in the learnable affine parameters.
            See :math:`\beta` in :class:`~.BatchNorm2d`.
        training: a boolean value to indicate whether batch norm is performed
            in training mode. Default: False
        momentum: value used for the ``running_mean`` and ``running_var``
            computation. Default: 0.9
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        inplace: whether to update ``running_mean`` and ``running_var``
            inplace or return new tensors. Default: True
    """
    if inp.ndim != 4:
        raise NotImplementedError("batch_norm for ndim != 4")

    if param_dim == "dim_1c11":
        C = inp.shape[1]
        pshape = (1, C, 1, 1)
    elif param_dim == "dim_111c":
        C = inp.shape[3]
        pshape = (1, 1, 1, C)
    else:
        raise ValueError("Invalid param_dim {}".format(param_dim))

    def make_full_if_none(x, value):
        if x is None:
            (x,) = Const(value, dtype=inp.dtype, device=inp.device)()
            shape = astensor1d(pshape, inp, dtype="int32", device=inp.device)
            (result,) = apply(builtin.Broadcast(), x, shape)
            return result
        elif x.ndim == 1:
            shape = astensor1d(pshape, inp, dtype="int32", device=inp.device)
            (result,) = apply(builtin.Reshape(), x, shape)
            return result
        return x

    has_mean = running_mean is not None
    has_var = running_var is not None

    if not training:
        assert has_mean, "running_mean must be provided in inference mode"
        assert has_var, "running_var must be provided in inference mode"

    if has_mean and running_mean.ndim != 4:
        raise ValueError
    if has_var and running_var.ndim != 4:
        raise ValueError

    if amp._enabled:
        inp = inp.astype("float16")
        weight, bias, running_mean, running_var = cast_tensors(
            weight, bias, running_mean, running_var, promote=True
        )
    weight = make_full_if_none(weight, 1)
    bias = make_full_if_none(bias, 0)

    if not training:
        op = builtin.BatchNorm(
            fwd_mode=BatchNorm.FwdMode.INFERENCE, epsilon=eps, param_dim=param_dim
        )
        ret = apply(op, inp, weight, bias, running_mean, running_var)[-1]
        return ret

    else:
        op = builtin.BatchNorm(
            avg_factor=1 - momentum, epsilon=eps, param_dim=param_dim
        )
        if has_mean or has_var:
            running_mean = make_full_if_none(running_mean, 0)
            running_var = make_full_if_none(running_var, 1)
            new_mean, new_var, *_, inp = apply(
                op, inp, weight, bias, running_mean, running_var
            )
            if not has_mean:
                new_mean = None
            if not has_var:
                new_var = None

            if inplace:
                if has_mean:
                    running_mean[...] = new_mean
                if has_var:
                    running_var[...] = new_var

                return inp
            else:
                return inp, new_mean, new_var
        else:
            inp = apply(op, inp, weight, bias)[-1]
            return inp


@lru_cache(maxsize=None)
def _get_sync_bn_ops(device, dtype, eps_mode, ndim, channels):
    # fmt: off
    @subgraph("SyncBnStage0", dtype, device, 1)
    def syncbn_stage0(inputs, f, c):
        input = inputs[0]
        reduce_shape = c((1, channels) + (1,) * (ndim - 2), dtype="int32", device=device)
        input_shape = f(GetVarShape(), input)
        input_elems = f(Reduce(mode="product", axis=0), input_shape)
        reduce_elems = f(Reduce(mode="product", axis=0), reduce_shape)
        reduce_size = f("//", input_elems, reduce_elems)
        channel_x1s = f(Reduce(mode="sum"), input, reduce_shape)
        channel_x2s = f(Reduce(mode="sum_sqr"), input, reduce_shape)
        reduce_size_f = f(TypeCvt(dtype=dtype), reduce_size)
        return (reduce_shape, reduce_size_f, channel_x1s, channel_x2s), (False, False, True, True)

    @subgraph("SyncBnStage1", dtype, device, 7)
    def syncbn_stage1(inputs, f, c):
        input, reduce_size, channel_x1s, channel_x2s, eps = inputs[0:5]
        weight, bias = inputs[5:7]
        channel_mean = f("/", channel_x1s, reduce_size)
        channel_var =\
            f("+",  f("/",  f("**", channel_x1s, c(2)),
                            f("-",  f("*", reduce_size, reduce_size))),
                    f("/", channel_x2s, reduce_size))
        invsqrt_channel_var = f("**", f(eps_mode, channel_var, eps), c(-0.5))
        inv_var_wt = f("*", invsqrt_channel_var, weight)
        neg_channel_mean = f("-", channel_mean)
        outvar =\
            f("fma3",  input, inv_var_wt,
                    f("+",  f("*", neg_channel_mean, inv_var_wt),
                            bias))
        return (outvar, channel_mean, channel_var, inv_var_wt), (True, False, False, False)

    @subgraph("SyncBnStage1Inference", dtype, device, 6)
    def syncbn_stage1_inference(inputs, f, c):
        input, channel_mean, channel_var, eps = inputs[0:4]
        weight, bias = inputs[4:6]
        invsqrt_channel_var = f("**", f(eps_mode, channel_var, eps), c(-0.5))
        inv_var_wt = f("*", invsqrt_channel_var, weight)
        neg_channel_mean = f("-", channel_mean)
        outvar =\
            f("+",  f("*", input, inv_var_wt),
                    f("+",  f("*", neg_channel_mean, inv_var_wt),
                            bias))
        return (outvar,), (True,)

    @subgraph("SyncBnStage2", dtype, device, 7)
    def syncbn_stage2(inputs, f, c):
        running_mean, running_var, momentum = inputs[0:3]
        reduce_size, channel_x1s, channel_x2s, channel_mean = inputs[3:7]
        c1_minus_momentum = f("-", c(1), momentum)
        reduce_size_minus_c1 = f("-", reduce_size, c(1))
        running_mean = f("fma4",
            running_mean, momentum,
            c1_minus_momentum, channel_mean,
        )
        channel_variance_unbiased =\
            f("+",  f("/",  f("**", channel_x1s, c(2)),
                            f("*",  f("-", reduce_size),
                                    reduce_size_minus_c1)),
                    f("/",  channel_x2s,
                            reduce_size_minus_c1))
        running_var = f("fma4",
            running_var, momentum,
            c1_minus_momentum, channel_variance_unbiased
        )
        return (running_mean, running_var), (True, True)

    @subgraph("SyncBnConcatStats", dtype, device, 3)
    def syncbn_concat_stats(inputs, f, c):
        reduce_size, channel_x1s, channel_x2s = inputs[0:3]
        reduce_size = f(builtin.Broadcast(), reduce_size, c([1]*ndim, dtype="int32"))
        stats = f(builtin.Concat(axis=1, comp_node=device), reduce_size, channel_x1s, channel_x2s)
        return (stats,), (True,)

    @subgraph("SyncBnSplitStats", dtype, device, 1)
    def syncbn_split_stats(inputs, f, c):
        stats = inputs[0]
        c_1 = c(1, dtype="int32")
        channel_x1s_end = c(channels+1, dtype="int32")
        def _subtensor(src, axis, begin, end):
            items = (axis, (begin is not None), (end is not None), False, False),
            args = ()
            if begin is not None:
                args += begin,
            if end is not None:
                args += end,
            return f(builtin.Subtensor(items=items), src, *args)
        reduce_size = _subtensor(stats, 1, None, c_1)
        channel_x1s = _subtensor(stats, 1, c_1, channel_x1s_end)
        channel_x2s = _subtensor(stats, 1, channel_x1s_end, None)
        reduce_size = f(builtin.Reshape(), reduce_size, c_1)
        return (reduce_size, channel_x1s, channel_x2s), (False, True, True)
    # fmt: on
    return (
        syncbn_stage0,
        syncbn_stage1,
        syncbn_stage1_inference,
        syncbn_stage2,
        syncbn_concat_stats,
        syncbn_split_stats,
    )


def sync_batch_norm(
    inp: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    training: bool = False,
    momentum: Union[float, Tensor] = 0.9,
    eps: float = 1e-5,
    eps_mode="additive",
    group=WORLD,
) -> Tensor:
    r"""Applies synchronized batch normalization to the input.

    Refer to :class:`~.BatchNorm2d` and :class:`~.BatchNorm1d` for more information.

    Args:
        inp: input tensor.
        running_mean: tensor to store running mean.
        running_var: tensor to store running variance.
        weight: scaling tensor in the learnable affine parameters.
            See :math:`\gamma` in :class:`~.BatchNorm2d`.
        bias: bias tensor in the learnable affine parameters.
            See :math:`\beta` in :class:`~.BatchNorm2d`.
        training: a boolean value to indicate whether batch norm is performed
            in traning mode. Default: False
        momentum: value used for the ``running_mean`` and ``running_var``
            computation. Default: 0.9
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        eps_mode: mode of calculation for eps, "max" or "additive".
            Default: "additive"
        group: communication group, caculate mean and variance between this group.
            Default: :obj:`~megengine.distributed.WORLD`
    """
    _eps_mode = eps_mode.lower()
    assert _eps_mode in {"max", "additive"}, "unknown eps_mode: {}".format(eps_mode)
    if _eps_mode == "additive" and not (is_distributed() and training):
        return batch_norm(
            inp,
            running_mean,
            running_var,
            weight,
            bias,
            training=training,
            momentum=momentum,
            eps=eps,
        )
    _channels = make_shape_tuple(inp.shape)[1]
    _ndim = inp.ndim
    _device = inp.device
    _dtype = inp.dtype

    if _ndim != 4:
        raise NotImplementedError("sync_batch_norm for ndim != 4")

    def _make_full_if_none(x, value):
        if x is None:
            (x,) = Const(value, dtype=inp.dtype, device=_device)()
            (result,) = apply(builtin.Broadcast(), x, reduce_shape)
            return result
        elif x.ndim == 1:
            (result,) = apply(builtin.Reshape(), x, reduce_shape)
            return result
        return x

    (
        syncbn_stage0,
        syncbn_stage1,
        syncbn_stage1_inference,
        syncbn_stage2,
        syncbn_concat_stats,
        syncbn_split_stats,
    ) = _get_sync_bn_ops(_device, _dtype, eps_mode, _ndim, _channels)

    reduce_shape, reduce_size, channel_x1s, channel_x2s = apply(syncbn_stage0(), inp)

    eps = convert_single_value(eps, dtype=inp.dtype, device=inp.device)

    weight = _make_full_if_none(weight, 1)
    bias = _make_full_if_none(bias, 0)

    if training:
        if is_distributed():
            # reduce all nodes' data to calculate mean and variance
            (stat,) = apply(
                syncbn_concat_stats(), reduce_size, channel_x1s, channel_x2s
            )
            stat = all_reduce_sum(stat, group)
            reduce_size, channel_x1s, channel_x2s = apply(syncbn_split_stats(), stat)

        outvar, channel_mean, *_ = apply(
            syncbn_stage1(),
            inp,
            reduce_size,
            channel_x1s,
            channel_x2s,
            eps,
            weight,
            bias,
        )
    else:
        assert running_var is not None and running_mean is not None
        channel_mean = running_mean
        channel_var = running_var
        outvar, *_ = apply(
            syncbn_stage1_inference(), inp, channel_mean, channel_var, eps, weight, bias
        )

    # outvar = output * weight + bias
    # where output = inp * invsqrt_channel_variance + (
    #    -channel_mean * invsqrt_channel_variance
    # )
    # Manually expand output for gopt

    if training and running_var is not None and running_mean is not None:
        momentum = convert_single_value(momentum, dtype=inp.dtype, device=inp.device)
        running_mean[...], running_var[...] = apply(
            syncbn_stage2(),
            running_mean,
            running_var,
            momentum,
            reduce_size,
            channel_x1s,
            channel_x2s,
            channel_mean,
        )

    return outvar


def dropout(inp: Tensor, drop_prob: float, training: bool = True) -> Tensor:
    r"""Returns a new tensor where each of the elements are randomly set to zero
    with probability P = ``drop_prob``. Optionally rescale the output tensor if ``training`` is True.

    Args:
        inp: input tensor.
        drop_prob: probability to drop (set to zero) a single element.
        training: the default behavior of ``dropout`` during training is to rescale the output,
            then it can be replaced by an :class:`~.Identity` during inference. Default: True
    Returns:
        the ouput tensor

    Examples:

        .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            # test training mode
            data = tensor(np.ones(10000000, dtype=np.float32))
            out = F.nn.dropout(data, 1.0 / 3.0, training=True)
            assert not out.numpy().all()

            # test eval mode
            out = F.nn.dropout(data, 1.0 / 3.0, training=False)
            assert out.numpy().all()

        Outputs:

        .. testoutput::
            :options: +SKIP

            [1.5 1.5 0.  1.5 1.5 1.5 1.5 1.5 1.5 1.5]
    """
    assert 0 <= drop_prob < 1
    if not training or drop_prob == 0:
        return inp

    # model in training mode, e.g. model.train()
    rv = uniform(size=inp.shape)
    mask = rv > drop_prob
    ret = inp * mask.astype(inp.dtype)
    ret *= 1 / (1 - drop_prob)

    return ret


def one_hot(inp: Tensor, num_classes: int) -> Tensor:
    r"""Performs one-hot encoding for the input tensor.

    Args:
        inp: input tensor.
        num_classes: number of classes denotes the last dimension of the output tensor.

    Examples:

        .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            x = tensor(np.arange(1, 4, dtype=np.int32))
            out = F.one_hot(x, num_classes=4)
            print(out.numpy())

        Outputs:

        .. testoutput::

            [[0 1 0 0]
             [0 0 1 0]
             [0 0 0 1]]
    """
    zeros_tensor = zeros(list(inp.shape) + [num_classes], inp.dtype, inp.device)
    ones_tensor = ones(list(inp.shape) + [1], inp.dtype, inp.device)

    op = builtin.IndexingSetOneHot(axis=inp.ndim)
    (result,) = apply(op, zeros_tensor, inp, ones_tensor)
    return result


def embedding(
    inp: Tensor,
    weight: Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: Optional[float] = None,
):
    r"""Applies lookup table for embedding.

    Args:
        inp: tensor with indices.
        weight: learnable weights which embeds from.
        padding_idx: should be set to None, not supported now.
        max_norm: should be set to None, not supported now.
        norm_type: should be set to None, not supported now.

    Refer to :class:`~.module.Embedding` for more information.
    """
    if padding_idx is not None:
        raise ValueError("Not support padding_idx Now!")
    if max_norm is not None or norm_type is not None:
        raise ValueError("Not support weight normlization Now!")

    dest_shp = list(inp.shape) + [weight.shape[-1]]
    return weight[inp.reshape(-1)].reshape(dest_shp)


def indexing_one_hot(
    src: Tensor, index: Tensor, axis: int = 1, keepdims=False
) -> Tensor:
    r"""One-hot indexing for some axes.

    Args:
        src: input tensor.
        index: index tensor.
        axis: axis on src for which values in index index. Default: 1
        keepdims: whether not to remove the axis in result. Default: False

    Examples:

        .. testcode::

            import megengine.functional as F
            from megengine import tensor

            src = tensor([[1.0, 2.0]])
            index = tensor([0])
            val = F.indexing_one_hot(src, index)
            print(val.numpy())

        Outputs:

        .. testoutput::

            [1.]
    """
    assert isinstance(src, Tensor), "src must be of Tensor type"
    op = builtin.IndexingOneHot(axis=axis)
    index = convert_single_value(index, dtype="int32", device=src.device)
    (result,) = apply(op, src, index)
    if not keepdims:
        result = squeeze(result, axis)
    return result


def sliding_window(
    inp: Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int]] = 0,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
) -> Tensor:
    r"""Extracts sliding local blocks from a batched input tensor.

    Refer to :class:`~.SlidingWindow` for more information.

    Args:
        inp: input tensor.
        kernel_size: size of the window.
        padding: implicit zero padding added on both sides of input. Default: 0
        stride: stride of the window. Default: 1
        dilation: dilation of the window. Default: 1
    """
    padding_h, padding_w = _pair(padding)
    stride_h, stride_w = _pair_nonzero(stride)
    dilation_h, dilation_w = _pair_nonzero(dilation)
    window_h, window_w = _pair_nonzero(kernel_size)

    op = builtin.Images2Neibs(
        pad_h=padding_h,
        pad_w=padding_w,
        stride_h=stride_h,
        stride_w=stride_w,
        dilate_h=dilation_h,
        dilate_w=dilation_w,
        window_h=window_h,
        window_w=window_w,
    )
    (output,) = apply(op, inp)
    return output


def sliding_window_transpose(
    inp: Tensor,
    output_size: Union[int, Tuple[int, int]],
    kernel_size: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int]] = 0,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
) -> Tensor:
    r"""Sum over the sliding windows on the corresponding input location.

    Refer to :class:`~.SlidingWindowTranspose` for more information.

    Args:
        inp: input tensor.
        output_size: shape of output tensor.
        kernel_size: size of the window.
        padding: implicit zero padding added on both sides of input. Default: 0
        stride: stride of the window. Default: 1
        dilation: dilation of the window. Default: 1
    """
    output_h, output_w = _pair_nonzero(output_size)
    padding_h, padding_w = _pair(padding)
    stride_h, stride_w = _pair_nonzero(stride)
    dilation_h, dilation_w = _pair_nonzero(dilation)
    window_h, window_w = _pair_nonzero(kernel_size)

    expected_h = (
        output_h + 2 * padding_h - dilation_h * (window_h - 1) - 1
    ) // stride_h + 1
    expected_w = (
        output_w + 2 * padding_w - dilation_w * (window_w - 1) - 1
    ) // stride_w + 1
    assert inp.ndim == 6, "the input dimension of sliding_window_transpose should be 6"
    assert (
        inp.shape[2] == expected_h and inp.shape[3] == expected_w
    ), "the input shape and output size do not match"

    op = builtin.SlidingWindowTranspose(
        out_h=output_h,
        out_w=output_w,
        pad_h=padding_h,
        pad_w=padding_w,
        stride_h=stride_h,
        stride_w=stride_w,
        dilate_h=dilation_h,
        dilate_w=dilation_w,
        window_h=window_h,
        window_w=window_w,
    )
    (output,) = apply(op, inp)
    return output


def pad(
    src: Tensor,
    pad_witdth: Tuple[Tuple[int, int], ...],
    mode: str = "constant",
    constant_value: float = 0.0,
) -> Tensor:
    """
    Pad is python warpper for padding opr in megbrain, can padding in random one of the max 7 dimensions.
    Supported constant, edge(replicate) and reflect mode, constatnt is the default mode.
    """
    p_offsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    assert mode.lower() in ["constant", "edge", "replicate", "reflect"]

    if mode.lower() == "edge":
        mode = "replicate"

    for i in range(0, len(pad_witdth)):
        p_offsets[i * 2] = pad_witdth[i][0]
        p_offsets[i * 2 + 1] = pad_witdth[i][1]

    op = builtin.Padding(
        front_offset_dim0=p_offsets[0],
        front_offset_dim1=p_offsets[2],
        front_offset_dim2=p_offsets[4],
        front_offset_dim3=p_offsets[6],
        front_offset_dim4=p_offsets[8],
        front_offset_dim5=p_offsets[10],
        front_offset_dim6=p_offsets[12],
        back_offset_dim0=p_offsets[1],
        back_offset_dim1=p_offsets[3],
        back_offset_dim2=p_offsets[5],
        back_offset_dim3=p_offsets[7],
        back_offset_dim4=p_offsets[9],
        back_offset_dim5=p_offsets[11],
        back_offset_dim6=p_offsets[13],
        padding_val=constant_value,
        padding_mode=mode.upper(),
    )
    (output,) = apply(op, src)
    return output


def local_response_norm(
    inp: Tensor,
    kernel_size: int = 5,
    k: float = 2.0,
    alpha: float = 1e-4,
    beta: float = 0.75,
) -> Tensor:
    r"""
    Apply local response normalization to the input tensor.

    Args:
        kernel_size: the size of the kernel to apply LRN on.
        k: hyperparameter k. The default vaule is 2.0.
        alpha: hyperparameter alpha. The default value is 1e-4.
        beta: hyperparameter beta. The default value is 0.75.

    Example:

    .. testcode::

        from megengine import tensor
        import megengine.functional as f
        import numpy as np

        inp = tensor(np.arange(25, dtype=np.float32).reshape(1,1,5,5))
        GT = np.array([[[[ 0.,         0.999925,   1.9994003,  2.9979765,  3.9952066],
           [ 4.9906454,  5.983851,   6.974385,   7.961814,   8.945709 ],
           [ 9.925651,  10.90122,   11.872011,  12.837625,  13.7976675],
           [14.751757,  15.699524,  16.640602,  17.574642,  18.501305 ],
           [19.420258,  20.331186,  21.233786,  22.127764,  23.012836 ]]]])

        out = f.local_response_norm(inp, kernel_size=3, k=1.0, alpha=1e-4, beta=0.75)
        np.testing.assert_allclose(GT, out.numpy(), rtol=1e-6, atol=1e-6)
        print('pass')

    Outputs:

    .. testoutput::

        pass

    """
    op = builtin.LRN(n=kernel_size, k=k, alpha=alpha, beta=beta,)
    (output,) = apply(op, inp)
    return output


@lru_cache(maxsize=None)
def _get_layerPixelShuffle(device, dtype, dim_order):
    @subgraph("LayerPixelShuffle", dtype, device, 3)
    def layerPixelShuffle(inputs, f, c):
        inp, shape_0, shape_1 = inputs
        inp = f(Reshape(), inp, shape_0)
        inp = f(Dimshuffle(dim_order), inp)
        oup = f(Reshape(), inp, shape_1)
        return (oup,), (True,)

    return layerPixelShuffle


def pixel_shuffle(inp: Tensor, upscale_factor: int) -> Tensor:
    """
    Rearranges elements in a tensor of shape (*, C x r^2, H, W) to a tensor of
    shape (*, C, H x r, W x r), where r is an upscale factor, where * is zero
    or more batch dimensions.

    :param inp: input tensor.
    :param upscale_factor: upscale factor of pixel_shuffle.
    :return: output tensor.
    """
    assert upscale_factor > 0, "upscale_factor should larger than 0"
    assert inp.ndim >= 3, "the input dimension of pixel_shuffle should be larger than 3"
    assert (
        inp.shape[-3] % (upscale_factor ** 2) == 0
    ), "the -3 dimension should be divided by (upscale_factor ** 2)"

    _device = inp.device
    _dtype = inp.dtype
    shape_ori = inp.shape
    high_dim = shape_ori[:-3]
    square = upscale_factor ** 2
    n = 1
    for item in high_dim:
        n *= item
    shape_0 = (
        n,
        int(shape_ori[-3] / square),
        upscale_factor,
        upscale_factor,
        shape_ori[-2],
        shape_ori[-1],
    )
    shape_1 = (
        *high_dim,
        shape_ori[-3] / square,
        shape_ori[-2] * upscale_factor,
        shape_ori[-1] * upscale_factor,
    )

    dim_order = (0, 1, 4, 2, 5, 3)

    layerPixelShuffle = _get_layerPixelShuffle(_device, _dtype, dim_order)

    shape_0 = convert_single_value(shape_0, dtype=inp.dtype, device=inp.device)
    shape_1 = convert_single_value(shape_1, dtype=inp.dtype, device=inp.device)
    outvar, *_ = apply(layerPixelShuffle(), inp, shape_0, shape_1)

    return outvar


from .quantized import conv_bias_activation  # isort:skip
from .loss import *  # isort:skip
from .metric import *  # isort:skip
from .vision import *  # isort:skip
