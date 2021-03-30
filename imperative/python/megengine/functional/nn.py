# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# pylint: disable=too-many-lines
from typing import Optional, Sequence, Tuple, Union

from ..core._imperative_rt.core2 import apply
from ..core._imperative_rt.graph import VarNode
from ..core._trace_option import use_symbolic_shape
from ..core.ops import builtin
from ..core.ops.builtin import BatchNorm, Elemwise
from ..core.ops.special import Const
from ..core.tensor import megbrain_graph, utils
from ..core.tensor.array_method import _elwise_apply
from ..core.tensor.utils import astensor1d, astype, setscalar
from ..device import get_default_device
from ..distributed import WORLD, is_distributed
from ..random import uniform
from ..tensor import Tensor
from ..utils.deprecation import deprecated_func
from ..utils.tuple_function import _pair, _pair_nonzero, _triple, _triple_nonzero
from .debug_param import get_execution_strategy
from .distributed import all_reduce_sum
from .elemwise import _elwise, exp, floor, log, log1p, maximum, minimum
from .math import argsort, matmul, max, prod, sum
from .tensor import (
    broadcast_to,
    concat,
    expand_dims,
    full,
    ones,
    reshape,
    squeeze,
    zeros,
)

__all__ = [
    "adaptive_avg_pool2d",
    "adaptive_max_pool2d",
    "avg_pool2d",
    "batch_norm",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose2d",
    "deformable_conv2d",
    "deformable_psroi_pooling",
    "dropout",
    "embedding",
    "hsigmoid",
    "hswish",
    "indexing_one_hot",
    "leaky_relu",
    "linear",
    "local_conv2d",
    "logsigmoid",
    "logsumexp",
    "logsoftmax",
    "max_pool2d",
    "one_hot",
    "prelu",
    "relu",
    "relu6",
    "remap",
    "resize",
    "sigmoid",
    "softmax",
    "softplus",
    "sync_batch_norm",
    "warp_affine",
    "warp_perspective",
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


def linear(inp: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """
    Applies a linear transformation to the input tensor.

    Refer to :class:`~.module.linear.Linear` for more information.

    :param inp: input tensor with shape `(N, in_features)`.
    :param weight: weight with shape `(out_features, in_features)`.
    :param bias: bias with shape `(out_features,)`.
        Default: None
    """
    ret = matmul(inp, weight, transpose_b=True)
    if bias is not None:
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
    """1D convolution operation.

    Refer to :class:`~.Conv1d` for more information.

    :param inp: The feature map of the convolution operation
    :param weight: The convolution kernel
    :param bias: The bias added to the result of convolution (if given)
    :param stride: Stride of the 1D convolution operation. Default: 1
    :param padding: Size of the paddings added to the input on both sides of its
        spatial dimensions. Only zero-padding is supported. Default: 0
    :param dilation: Dilation of the 1D convolution operation. Default: 1
    :param groups: number of groups to divide input and output channels into,
        so as to perform a "grouped convolution". When ``groups`` is not 1,
        ``in_channels`` and ``out_channels`` must be divisible by ``groups``,
        and the shape of weight should be ``(groups, out_channel // groups,
        in_channels // groups, height, width)``.
    :type conv_mode: string or :class:`mgb.opr_param_defs.Convolution.Mode`
    :param conv_mode: Supports 'cross_correlation'. Default:
        'cross_correlation'.
    :type compute_mode: string or
        :class:`mgb.opr_param_defs.Convolution.ComputeMode`
    :param compute_mode: When set to 'default', no special requirements will be
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
    inp, weight = utils.convert_inputs(inp, weight)
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
    """
    2D convolution operation.

    Refer to :class:`~.module.Conv2d` for more information.

    :param inp: feature map of the convolution operation.
    :param weight: convolution kernel.
    :param bias: bias added to the result of convolution (if given).
    :param stride: stride of the 2D convolution operation. Default: 1
    :param padding: size of the paddings added to the input on both sides of its
        spatial dimensions. Only zero-padding is supported. Default: 0
    :param dilation: dilation of the 2D convolution operation. Default: 1
    :param groups: number of groups into which the input and output channels are divided, 
        so as to perform a ``grouped convolution``. When ``groups`` is not 1,
        ``in_channels`` and ``out_channels`` must be divisible by ``groups``,
        and the shape of weight should be `(groups, out_channel // groups,
        in_channels // groups, height, width)`.
    :type conv_mode: string or :class:`Convolution.Mode`
    :param conv_mode: supports "cross_correlation". Default:
        "cross_correlation"
    :type compute_mode: string or
        :class:`Convolution.ComputeMode`
    :param compute_mode: when set to "default", no special requirements will be
        placed on the precision of intermediate results. When set to "float32",
        "float32" would be used for accumulator and intermediate result, but only
        effective when input and output are of float16 dtype.
    :return: output tensor.
    """
    assert (
        conv_mode.lower() == "cross_correlation"
        or conv_mode.name == "CROSS_CORRELATION"
    )
    assert compute_mode.lower() == "default" or compute_mode.name == "DEFAULT"

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
    inp, weight = utils.convert_inputs(inp, weight)
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
    """
    3D convolution operation.

    Refer to :class:`~.Conv3d` for more information.

    :param inp: feature map of the convolution operation.
    :param weight: convolution kernel.
    :param bias: bias added to the result of convolution (if given).
    :param stride: stride of the 3D convolution operation. Default: 1
    :param padding: size of the paddings added to the input on both sides of its
        spatial dimensions. Only zero-padding is supported. Default: 0
    :param dilation: dilation of the 3D convolution operation. Default: 1
    :param groups: number of groups into which the input and output channels are divided,
        so as to perform a ``grouped convolution``. When ``groups`` is not 1,
        ``in_channels`` and ``out_channels`` must be divisible by ``groups``,
        and the shape of weight should be `(groups, out_channel // groups,
        in_channels // groups, t, height, width)`.
    :param conv_mode: supports "cross_correlation". Default:
        "cross_correlation"
    :return: output tensor.
    """
    assert conv_mode.lower() == "cross_correlation"

    D, H, W = 0, 1, 2

    pad = _triple(padding)
    stride = _triple_nonzero(stride)
    dilate = _triple_nonzero(dilation)

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
    inp, weight = utils.convert_inputs(inp, weight)
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
    """
    2D transposed convolution operation.

    Refer to :class:`~.ConvTranspose2d` for more information.

    :param inp: feature map of the convolution operation.
    :param weight: convolution kernel.
    :param bias: bias added to the result of convolution (if given).
    :param stride: stride of the 2D convolution operation. Default: 1
    :param padding: size of the paddings added to the input on both sides of its
        spatial dimensions. Only zero-padding is supported. Default: 0
    :param dilation: dilation of the 2D convolution operation. Default: 1
    :param groups: number of groups into which the input and output channels are divided, 
        so as to perform a ``grouped convolution``. When ``groups`` is not 1,
        ``in_channels`` and ``out_channels`` must be divisible by groups,
        and the shape of weight should be `(groups, out_channel // groups,
        in_channels // groups, height, width)`. Default: 1
    :type conv_mode: string or :class:`Convolution.Mode`
    :param conv_mode: supports "cross_correlation". Default:
        "cross_correlation"
    :type compute_mode: string or
        :class:`Convolution.ComputeMode`
    :param compute_mode: when set to "default", no special requirements will be
        placed on the precision of intermediate results. When set to "float32",
        "float32" would be used for accumulator and intermediate result, but only
        effective when input and output are of float16 dtype.
    :return: output tensor.
    """
    assert (
        conv_mode.lower() == "cross_correlation"
        or conv_mode.name == "CROSS_CORRELATION"
    )
    assert compute_mode.lower() == "default" or compute_mode.name == "DEFAULT"

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
    )
    weight, inp = utils.convert_inputs(weight, inp)
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
    """
    Deformable Convolution.

    :param inp: input feature map.
    :param weight: convolution kernel.
    :param offset: input offset to kernel, channel of this tensor should match the deformable settings.
    :param mask: input mask to kernel, channel of this tensor should match the deformable settings.
    :param bias: bias added to the result of convolution (if given).
    :param stride: stride of the 2D convolution operation. Default: 1
    :param padding: size of the paddings added to the input on both sides of its
        spatial dimensions. Only zero-padding is supported. Default: 0
    :param dilation: dilation of the 2D convolution operation. Default: 1
    :param groups: number of groups into which the input and output channels are divided, 
        so as to perform a ``grouped convolution``. When ``groups`` is not 1,
        ``in_channels`` and ``out_channels`` must be divisible by groups,
        and the shape of weight should be `(groups, out_channel // groups,
        in_channels // groups, height, width)`. Default: 1
    :type conv_mode: string or :class:`Convolution.Mode`
    :param conv_mode: supports "cross_correlation". Default:
        "cross_correlation"
    :type compute_mode: string or
        :class:`Convolution.ComputeMode`
    :param compute_mode: when set to "default", no special requirements will be
        placed on the precision of intermediate results. When set to "float32",
        "float32" would be used for accumulator and intermediate result, but only
        effective when input and output are of float16 dtype.
    :return: output tensor.
    """
    assert (
        conv_mode.lower() == "cross_correlation"
        or conv_mode.name == "CROSS_CORRELATION"
    )
    assert compute_mode.lower() == "default" or compute_mode.name == "DEFAULT"

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
    inp, weight, offset, mask = utils.convert_inputs(inp, weight, offset, mask)
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
    """Applies spatial 2D convolution over an groupped channeled image with untied kernels."""
    assert (
        conv_mode.lower() == "cross_correlation"
        or conv_mode.name == "CROSS_CORRELATION"
    )

    stride_h, stride_w = expand_hw(stride)
    pad_h, pad_w = expand_hw(padding)
    dilate_h, dilate_w = expand_hw(dilation)

    op = builtin.GroupLocal(
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dilate_h=dilate_h,
        dilate_w=dilate_w,
        mode=conv_mode,
        compute_mode="default",
        sparse="dense",
    )
    inp, weight = utils.convert_inputs(inp, weight)
    (output,) = apply(op, inp, weight)
    if bias is not None:
        output += bias
    return output


def max_pool2d(
    inp: Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
) -> Tensor:
    """
    Applies a 2D max pooling over an input tensor.

    Refer to :class:`~.MaxPool2d` for more information.

    :param inp: input tensor.
    :param kernel_size: size of the window.
    :param stride: stride of the window. If not provided, its value is set to kernel_size.
        Default: None
    :param padding: implicit zero padding added on both sides. Default: 0
    :return: output tensor.
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
    """
    Applies 2D average pooling over an input tensor.

    Refer to :class:`~.AvgPool2d` for more information.

    :param inp: input tensor.
    :param kernel_size: size of the window.
    :param stride: stride of the window. If not provided, its value is set to ``kernel_size``.
        Default: None
    :param padding: implicit zero padding added on both sides. Default: 0
    :param mode: whether to count padding values, set to "average" will do counting.
        Default: "average_count_exclude_padding"
    :return: output tensor.
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
    """
    Applies a 2D max adaptive pooling over an input.

    Refer to :class:`~.MaxAdaptivePool2d` for more information.

    :param inp: input tensor.
    :param oshp: `(OH, OW)` size of the output shape.
    :return: output tensor.
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
    """
    Applies a 2D average adaptive pooling over an input.

    Refer to :class:`~.AvgAdaptivePool2d` for more information.

    :param inp: input tensor.
    :param oshp: `(OH, OW)` size of the output shape.
    :return: output tensor.
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
    """
    Deformable PSROI(Position Sensitive Region of Interest) Pooling.

    :param inp: input feature map.
    :param rois: the rois for feature pooling.
    :param trans: input offset to psroi_pooling.
    :param no_trans: check the phase of DeformablePSROIPooling. False to the
                        1st phase, True to the 2nd phase.
    :param part_size: part size.
    :param sample_per_part: sample points of each part.
    :param pooled_shape: kernel shape of convolution.
    :param spatial_scale: the spatial_scale w.r.t input image.
    :param trans_std: multiplier used in 2nd phase.
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
    """
    Element-wise `x * relu6(x + 3) / 6`.

    :param x: input tensor.
    :return: computed tensor.

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
    """Element-wise `1 / ( 1 + exp( -x ) )`."""
    return _elwise(x, mode=Elemwise.Mode.SIGMOID)


def hsigmoid(x):
    """Element-wise `relu6(x + 3) / 6`."""
    return relu6(x + 3) / 6


def relu(x):
    """Element-wise `max(x, 0)`."""
    return _elwise(x, mode=Elemwise.Mode.RELU)


def relu6(x):
    """Element-wise `min(max(x, 0), 6)`."""
    return minimum(maximum(x, 0), 6)


def prelu(inp: Tensor, weight: Tensor) -> Tensor:
    r"""
    Applies the element-wise PReLU function.

    Refer to :class:`~.PReLU` for more information.
    """
    return maximum(inp, 0) + weight * minimum(inp, 0)


def leaky_relu(inp: Tensor, negative_slope: float = 0.01) -> Tensor:
    r"""
    Applies the element-wise leaky_relu function

    Refer to :class:`~.LeakyReLU` for more information.
    """
    return maximum(inp, 0) + negative_slope * minimum(inp, 0)


def softplus(inp: Tensor) -> Tensor:
    r"""
    Applies the element-wise function:

    .. math::
        \text{softplus}(x) = \log(1 + \exp(x))

    softplus is a smooth approximation to the ReLU function and can be used
    to constrain the output to be always positive.
    For numerical stability the implementation follows this transformation:

    .. math::
        \text{softplus}(x) = \log(1 + \exp(x))
                           = \log(1 + \exp(-\text{abs}(x))) + \max(x, 0)
                           = \log1p(\exp(-\text{abs}(x))) + \text{relu}(x)

    :param inp: input tensor.

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
    r"""
    Applies the :math:`\log(\text{softmax}(x))` function to an n-dimensional
    input tensor. The :math:`\text{logsoftmax}(x)` formulation can be simplified as:

    .. math::
        \text{logsoftmax}(x_{i}) = \log(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} )

    For numerical stability the implementation follows this transformation:

    .. math::
        \text{logsoftmax}(x)
        = \log (\frac{\exp (x)}{\sum_{i}(\exp (x_{i}))})
        = x - \log (\sum_{i}(\exp (x_{i})))
        = x - \text{logsumexp}(x)

    :param inp: input tensor.
    :param axis: axis along which :math:`\text{logsoftmax}(x)` will be applied.

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
    r"""
    Applies the element-wise function:

    .. math::
        \text{logsigmoid}(x) = \log(\frac{ 1 }{ 1 + \exp(-x)})
        = \log(1/(1 + \exp(-x)))
        = - \log(1 + \exp(-x))
        = - \text{softplus}(-x)

    :param inp: input tensor.

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
    r"""
    Calculates the logarithm of the inputs' exponential sum along the given :attr:`axis`.

    .. math::

        \text{logsumexp}(x)= \log \sum_{j=1}^{n} \exp \left(x_{j}\right)

    For numerical stability, the implementation follows this transformation:

    .. math::

        \text{logsumexp}(x)= \log \sum_{j=1}^{n} \exp \left(x_{j}\right)
        = \text{logsumexp}(x)=b+\log \sum_{j=1}^{n} \exp \left(x_{j}-b\right)

    where

    .. math::
        b = \max(x_j)

    :param inp: input tensor.
    :param axis: axis over which the sum is taken. It could be single axis or list of axes.
    :param keepdims: whether to retain :attr:`axis` or not for the output tensor.

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
    r"""
    Applies a :math:`\text{softmax}(x)` function. :math:`\text{softmax}(x)` is defined as:

    .. math::
            \text{softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    It is applied to all elements along axis, and rescales elements so that
    they stay in the range `[0, 1]` and sum to 1.

    See :class:`~megengine.module.activation.Softmax` for more details.

    :param inp: input tensor.
    :param axis: an axis along which :math:`\text{softmax}(x)` will be applied. By default,
        :math:`\text{softmax}(x)` will apply along the highest ranked axis.

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
    inplace: bool = True
):
    r"""
    Applies batch normalization to the input.

    Refer to :class:`~.BatchNorm2d` and :class:`~.BatchNorm1d` for more information.

    :param inp: input tensor.
    :param running_mean: tensor to store running mean.
    :param running_var: tensor to store running variance.
    :param weight: scaling tensor in the learnable affine parameters.
        See :math:`\gamma` in :class:`~.BatchNorm2d`.
    :param bias: bias tensor in the learnable affine parameters.
        See :math:`\beta` in :class:`~.BatchNorm2d`.
    :param training: a boolean value to indicate whether batch norm is performed
        in training mode. Default: False
    :param momentum: value used for the ``running_mean`` and ``running_var``
        computation.
        Default: 0.9
    :param eps: a value added to the denominator for numerical stability.
        Default: 1e-5
    :param inplace: whether to update ``running_mean`` and ``running_var`` inplace or return new tensors
        Default: True
    :return: output tensor.
    """
    if inp.ndim != 4:
        raise NotImplementedError("batch_norm for ndim != 4")

    C = inp.shape[1]

    def make_full_if_none(x, value):
        if x is None:
            (x,) = Const(value, dtype=inp.dtype, device=inp.device)()
            shape = utils.astensor1d(
                (1, C, 1, 1), inp, dtype="int32", device=inp.device
            )
            (result,) = apply(builtin.Broadcast(), x, shape)
            return result
        elif x.ndim == 1:
            shape = utils.astensor1d(
                (1, C, 1, 1), inp, dtype="int32", device=inp.device
            )
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

    inp, weight, bias, running_mean, running_var = utils.convert_inputs(
        inp, weight, bias, running_mean, running_var
    )

    weight = make_full_if_none(weight, 1)
    bias = make_full_if_none(bias, 0)

    if not training:
        op = builtin.BatchNorm(
            fwd_mode=BatchNorm.FwdMode.INFERENCE, epsilon=eps, param_dim="dim_1c11"
        )
        ret = apply(op, inp, weight, bias, running_mean, running_var)[-1]
        return ret

    else:
        op = builtin.BatchNorm(
            avg_factor=1 - momentum, epsilon=eps, param_dim="dim_1c11"
        )
        if has_mean or has_var:
            running_mean = make_full_if_none(running_mean, 0)
            running_var = make_full_if_none(running_var, 1)
            new_mean, new_var, _, _, inp = apply(
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
            (_, _, inp,) = apply(op, inp, weight, bias)
            return inp


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
    r"""
    Applies synchronized batch normalization to the input.

    Refer to :class:`~.BatchNorm2d` and :class:`~.BatchNorm1d` for more information.

    :param inp: input tensor.
    :param running_mean: tensor to store running mean.
    :param running_var: tensor to store running variance.
    :param weight: scaling tensor in the learnable affine parameters.
        See :math:`\gamma` in :class:`~.BatchNorm2d`.
    :param bias: bias tensor in the learnable affine parameters.
        See :math:`\beta` in :class:`~.BatchNorm2d`.
    :param training: a boolean value to indicate whether batch norm is performed
        in traning mode. Default: False
    :param momentum: value used for the ``running_mean`` and ``running_var``
        computation.
        Default: 0.9
    :param eps: a value added to the denominator for numerical stability.
        Default: 1e-5
    :return: output tensor.
    """
    assert eps_mode.lower() in {"max", "additive"}, "unknown eps_mode: {}".format(
        eps_mode
    )
    _channels = inp.shape[1]
    _ndim = inp.ndim
    _device = inp.device
    _dtype = inp.dtype
    _param_shape = (1, _channels) + (1,) * (_ndim - 2)
    _reduce_axis = [0] + [i for i in range(2, _ndim)]

    if training:

        def _sum_on_channel(inp):
            return inp.sum(axis=_reduce_axis, keepdims=True)

        reduce_size = inp.shape[0]
        for i in range(2, _ndim):
            reduce_size = reduce_size * inp.shape[i]
        channel_x1s = _sum_on_channel(inp)
        channel_x2s = _sum_on_channel(inp ** 2)

        if is_distributed():
            # reduce all nodes' data to calculate mean and variance
            reduce_size = broadcast_to(
                Tensor(reduce_size).astype(dtype=_dtype), [1] * _ndim
            )
            stat = concat([reduce_size, channel_x1s, channel_x2s], axis=1)
            stat = all_reduce_sum(stat, group)
            reduce_size = stat[:, :1].reshape(1)
            channel_x1s = stat[:, 1 : 1 + _channels]
            channel_x2s = stat[:, 1 + _channels :]

        channel_mean = channel_x1s / reduce_size
        channel_variance = (
            channel_x1s ** 2 / (-reduce_size * reduce_size) + channel_x2s / reduce_size
        )
    else:
        assert running_var is not None and running_mean is not None
        channel_variance = running_var.reshape(*_param_shape)
        channel_mean = running_mean.reshape(*_param_shape)

    invsqrt_channel_variance = (
        maximum(channel_variance, eps) if eps_mode == "max" else channel_variance + eps
    ) ** -0.5

    if weight is not None:
        weight = weight.reshape(*_param_shape)
    if bias is not None:
        bias = bias.reshape(*_param_shape)

    # outvar = output * weight + bias
    # where output = inp * invsqrt_channel_variance + (
    #    -channel_mean * invsqrt_channel_variance
    # )
    # Manually expand output for gopt

    if weight is not None:
        inv_var_wt = invsqrt_channel_variance * weight
        neg_channel_mean = -channel_mean
        if bias is not None:
            outvar = inp * inv_var_wt + (neg_channel_mean * inv_var_wt + bias)
        else:
            outvar = inp * inv_var_wt + neg_channel_mean * inv_var_wt
    else:
        outvar = inp * invsqrt_channel_variance + (
            -channel_mean * invsqrt_channel_variance
        )
        if bias is not None:
            outvar = outvar + bias

    if training and running_var is not None and running_mean is not None:
        running_mean *= momentum
        running_mean += (1 - momentum) * channel_mean
        channel_variance_unbiased = channel_x1s ** 2 / (
            -reduce_size * (reduce_size - 1)
        ) + channel_x2s / (reduce_size - 1)
        running_var *= momentum
        running_var += (1 - momentum) * channel_variance_unbiased

    return outvar


def dropout(inp: Tensor, drop_prob: float, training: bool = True) -> Tensor:
    """
    Returns a new tensor where each of the elements are randomly set to zero
    with probability P = ``drop_prob``. Optionally rescale the output tensor if ``training`` is True.

    :param inp: input tensor.
    :param drop_prob: probability to drop (set to zero) a single element.
    :param training: the default behavior of ``dropout`` during training is to rescale the output,
        then it can be replaced by an :class:`~.Identity` during inference. Default: True
    :return: the output tensor

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = tensor(np.ones(10, dtype=np.float32))
        out = F.dropout(x, 1./3.)
        print(out.numpy())

    Outputs:

    .. testoutput::
        :options: +SKIP

        [1.5 1.5 0.  1.5 1.5 1.5 1.5 1.5 1.5 1.5]

    """
    assert 0 <= drop_prob < 1
    rv = uniform(size=inp.shape)
    mask = rv > drop_prob
    inp *= mask.astype(inp.dtype)
    if training:
        inp *= 1 / (1 - drop_prob)
    return inp


def one_hot(inp: Tensor, num_classes: int) -> Tensor:
    r"""
    Performs one-hot encoding for the input tensor.

    :param inp: input tensor.
    :param num_classes: number of classes denotes the last dimension of the output tensor.
    :return: output tensor.

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
    """
    Applies lookup table for embedding.

    :param inp: tensor with indices.
    :param weight: learnable weights which embeds from.
    :param padding_idx: should be set to None, not supported now.
    :param max_norm: should be set to None, not supported now.
    :param norm_type: should be set to None, not supported now.
    :return: output tensor.

    Refer to :class:`~.Embedding` for more information.
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
    r"""
    One-hot indexing for some axes.

    :param src: input tensor.
    :param index: index tensor.
    :param axis: axis on src for which values in index index. Default: 1
    :param keepdims: whether not to remove the axis in result. Default: False
    :return: output tensor.

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
    index = utils.convert_single_value(index, dtype="int32", device=src.device)
    (result,) = apply(op, src, index)
    if not keepdims:
        result = squeeze(result, axis)
    return result


interpolate = deprecated_func("1.3", "megengine.functional.vision", "interpolate", True)
roi_pooling = deprecated_func("1.3", "megengine.functional.vision", "roi_pooling", True)
roi_align = deprecated_func("1.3", "megengine.functional.vision", "roi_align", True)
nms = deprecated_func("1.3", "megengine.functional.vision", "nms", True)
resize = deprecated_func("1.3", "megengine.functional.vision", "resize", True)
remap = deprecated_func("1.3", "megengine.functional.vision", "remap", True)
nvof = deprecated_func("1.3", "megengine.functional.vision", "nvof", True)
warp_affine = deprecated_func("1.3", "megengine.functional.vision", "warp_affine", True)
warp_perspective = deprecated_func(
    "1.3", "megengine.functional.vision", "warp_perspective", True
)

from .loss import *  # isort:skip
from .quantized import conv_bias_activation  # isort:skip
