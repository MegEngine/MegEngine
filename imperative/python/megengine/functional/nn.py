# pylint : disable = too - many - lines
from builtins import max as _builtins_max
from builtins import min as _builtins_min
from functools import lru_cache
from typing import NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from ..core import _config
from ..core._imperative_rt.core2 import (
    Const,
    adaptive_pool2d_cpp,
    apply,
    dtype_promotion,
    pixel_shuffle_cpp,
)
from ..core._imperative_rt.ops import get_global_rng_seed as _get_global_rng_seed
from ..core.ops import builtin
from ..core.ops.builtin import (
    BatchNorm,
    Dimshuffle,
    Dropout,
    Elemwise,
    GetVarShape,
    Identity,
    Reduce,
    Reshape,
    TypeCvt,
)
from ..core.tensor import amp, megbrain_graph
from ..core.tensor.array_method import _matmul
from ..core.tensor.utils import (
    astensor1d,
    cast_tensors,
    convert_single_value,
    make_shape_tuple,
    subgraph,
    subgraph_fn,
)
from ..device import get_cudnn_version, get_default_device, is_cuda_available
from ..distributed import WORLD, is_distributed
from ..jit import exclude_from_trace
from ..logger import get_logger
from ..tensor import Tensor
from ..utils.deprecation import deprecated_func
from .debug_param import get_execution_strategy
from .distributed import all_reduce_sum
from .elemwise import _elwise, exp, log, log1p, maximum, minimum
from .math import max, normalize, sum
from .tensor import (
    broadcast_to,
    concat,
    expand_dims,
    ones,
    repeat,
    reshape,
    squeeze,
    transpose,
    zeros,
    zeros_like,
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
    "conv_transpose3d",
    "deformable_conv2d",
    "deformable_psroi_pooling",
    "dropout",
    "embedding",
    "gelu",
    "general_norm",
    "group_norm",
    "hsigmoid",
    "instance_norm",
    "hswish",
    "indexing_one_hot",
    "layer_norm",
    "leaky_relu",
    "linear",
    "local_conv2d",
    "local_response_norm",
    "logsigmoid",
    "logsumexp",
    "logsoftmax",
    "max_pool2d",
    "normalize",
    "one_hot",
    "prelu",
    "pad",
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
    "region_restricted_conv",
    "multi_head_attention",
]


def expand_hw(x):
    # judge int is 5 times faster than judge Sequence
    if isinstance(x, int):
        return x, x
    if isinstance(x, Sequence):
        return int(x[0]), int(x[1])
    return int(x), int(x)


def expand_dhw(x):
    if isinstance(x, int):
        return x, x, x
    if isinstance(x, Sequence):
        return int(x[0]), int(x[1]), int(x[2])
    return int(x), int(x), int(x)


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
    compute_mode = _config._get_actual_op_param(compute_mode, _config.__compute_mode)
    ret = _matmul(inp, weight, transpose_b=True, compute_mode=compute_mode)
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
    if bias is not None:
        assert bias.ndim == 3, "the bias dimension of conv1d should be 3"

    stride_h = stride
    pad_h = padding
    dilate_h = dilation

    compute_mode = _config._get_actual_op_param(compute_mode, _config.__compute_mode)
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
        if amp._enabled:
            (bias,) = cast_tensors(bias)
        output += bias
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

    stride_h, stride_w = expand_hw(stride)
    pad_h, pad_w = expand_hw(padding)
    dilate_h, dilate_w = expand_hw(dilation)

    sparse_type = "dense" if groups == 1 else "group"
    compute_mode = _config._get_actual_op_param(compute_mode, _config.__compute_mode)
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
        if amp._enabled:
            (bias,) = cast_tensors(bias)
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

    pad = expand_dhw(padding)
    stride = expand_dhw(stride)
    dilate = expand_dhw(dilation)

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
    output_padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    conv_mode="cross_correlation",
    compute_mode="default",
) -> Tensor:
    r"""2D transposed convolution operation.

    Refer to :class:`~.module.conv.ConvTranspose2d` for more information.

    Args:
        inp: feature map of the convolution operation.
        weight: convolution kernel.
            weight usually has shape ``(in_channels, out_channels, height, width)``.
        bias: bias added to the result of convolution (if given).
        stride: stride of the 2D convolution operation. Default: 1
        padding: size of the paddings added to the input on both sides of its
            spatial dimensions. Only zero-padding is supported. Default: 0
        output_padding: size of paddings appended to output. Default: 0
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

    stride_h, stride_w = expand_hw(stride)
    pad_h, pad_w = expand_hw(padding)
    output_pad_h, output_pad_w = expand_hw(output_padding)
    dilate_h, dilate_w = expand_hw(dilation)

    compute_mode = _config._get_actual_op_param(compute_mode, _config.__compute_mode)
    sparse_type = "dense" if groups == 1 else "group"
    op = builtin.ConvolutionBackwardData(
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dilate_h=dilate_h,
        dilate_w=dilate_w,
        strategy=get_execution_strategy(),
        compute_mode=compute_mode,
        sparse=sparse_type,
    )
    if output_pad_h != 0 or output_pad_h != 0:
        assert (
            output_pad_h < stride[0]
        ), "output_padding[0] shoule be less than stride[0]"
        assert (
            output_pad_w < stride[1]
        ), "output_padding[1] shoule be less than stride[1]"
        Hout = (
            (inp.shape[2] - 1) * stride[0]
            - 2 * padding[0]
            + dilation[0] * (weight.shape[2] - 1)
            + output_pad_h
            + 1
        )
        Wout = (
            (inp.shape[3] - 1) * stride[1]
            - 2 * padding[1]
            + dilation[1] * (weight.shape[3] - 1)
            + output_pad_w
            + 1
        )
        output_shape = [inp.shape[0], weight.shape[1], Hout, Wout]
        output_shape = astensor1d(output_shape)
        (output,) = apply(op, weight, inp, output_shape)
    else:
        (output,) = apply(op, weight, inp)
    if bias is not None:
        if amp._enabled:
            bias = cast_tensors(bias)
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
            weight usually has shape ``(out_channels, in_channels, height, width)``.
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
        inp, weight, offset, mask, bias = cast_tensors(inp, weight, offset, mask, bias)
    else:
        offset = offset.astype("float32")
        mask = mask.astype("float32")

    stride_h, stride_w = expand_hw(stride)
    pad_h, pad_w = expand_hw(padding)
    dilate_h, dilate_w = expand_hw(dilation)

    compute_mode = _config._get_actual_op_param(compute_mode, _config.__compute_mode)
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
    r"""Applies a spatial convolution with untied kernels over an groupped channeled input 4D tensor.
    It is also known as the locally connected layer.

    Args:
        inp: input feature map.
        weight: convolution kernel.
            weight usually has shape ``(out_channels, in_channels, height, width)``.
        bias: bias added to the result of convolution (if given).
        stride: stride of the 2D convolution operation. Default: 1
        padding: size of the paddings added to the input on both sides of its
            spatial dimensions. Only zero-padding is supported. Default: 0
        dilation: dilation of the 2D convolution operation. Default: 1

    Returns:
        output tensor.
    """
    assert (
        conv_mode.lower() == "cross_correlation"
        or conv_mode.name == "CROSS_CORRELATION"
    )

    stride_h, stride_w = expand_hw(stride)
    pad_h, pad_w = expand_hw(padding)
    dilate_h, dilate_w = expand_hw(dilation)

    # local conv only support "dense" mode, but weight could contain group dimension.
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
    output_padding: Union[int, Tuple[int, int, int]] = 0,
    dilation: Union[int, Tuple[int, int, int]] = 1,
    groups: int = 1,
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
        output_padding: size of paddings appended to output. Default: 0
        dilation: dilation of the 3D convolution operation. Default: 1
        groups: number of groups into which the input and output channels are divided,
            so as to perform a ``grouped convolution``. When ``groups`` is not 1,
            ``in_channels`` and ``out_channels`` must be divisible by groups,
            and the shape of weight should be ``(groups, in_channels // groups,
            out_channels // groups, depth, height, width)``. Default: 1

    Returns:
        output tensor.
    """
    D, H, W = 0, 1, 2
    pad = expand_dhw(padding)
    stride = expand_dhw(stride)
    dilate = expand_dhw(dilation)
    output_padding = expand_dhw(output_padding)

    sparse_type = "dense" if groups == 1 else "group"
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
        sparse=sparse_type,
    )
    if output_padding[0] != 0 or output_padding[1] != 0 or output_padding[2] != 0:
        assert (
            output_padding[0] < stride[0]
        ), "output_padding[0] shoule be less than stride[0]"
        assert (
            output_padding[1] < stride[1]
        ), "output_padding[1] shoule be less than stride[1]"
        assert (
            output_padding[2] < stride[2]
        ), "output_padding[2] shoule be less than stride[2]"
        Dout = (
            (inp.shape[2] - 1) * stride[0]
            - 2 * padding[0]
            + dilation[0] * (weight.shape[2] - 1)
            + output_padding[0]
            + 1
        )
        Hout = (
            (inp.shape[3] - 1) * stride[1]
            - 2 * padding[1]
            + dilation[1] * (weight.shape[3] - 1)
            + output_padding[1]
            + 1
        )
        Wout = (
            (inp.shape[4] - 1) * stride[2]
            - 2 * padding[2]
            + dilation[2] * (weight.shape[4] - 1)
            + output_padding[2]
            + 1
        )
        output_shape = [inp.shape[0], weight.shape[1], Dout, Hout, Wout]
        output_shape = astensor1d(output_shape)
        (output,) = apply(op, weight, inp, output_shape)
    else:
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
        inp: input tensor of shape :math:`(N, C, H_{\text{in}}, W_{\text{in}})`.
        kernel_size: size of the window used to calculate the max value.
        stride: stride of the window. Default value is ``kernel_size``.
        padding: implicit zero padding added on both sides. Default: 0.

    Returns:
        output tensor of shape `(N, C, H_{\text{out}}, W_{\text{out}})`.

    Examples:
        >>> import numpy as np
        >>> input = Tensor(np.arange(1 * 1 * 3 * 4).astype(np.float32).reshape(1, 1, 3, 4))
        >>> F.nn.max_pool2d(input, 2, 1, 0)
        Tensor([[[[ 5.  6.  7.]
           [ 9. 10. 11.]]]], device=xpux:0)
    """
    if stride is None:
        stride = kernel_size
    window_h, window_w = expand_hw(kernel_size)
    stride_h, stride_w = expand_hw(stride)
    padding_h, padding_w = expand_hw(padding)

    op = builtin.Pooling(
        window_h=window_h,
        window_w=window_w,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=padding_h,
        pad_w=padding_w,
        mode="max",
        strategy=get_execution_strategy(),
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
        inp: input tensor of shape :math:`(N, C, H_{\text{in}}, W_{\text{in}})` .
        kernel_size: size of the window used to calculate the average value.
        stride: stride of the window. Default value is ``kernel_size``.
        padding: implicit zero padding added on both sides. Default: 0.
        mode: whether to include the padding values while calculating the average, set
            to "average" will do counting.
            Default: "average_count_exclude_padding"

    Returns:
        output tensor of shape :math:`(N, C, H_{\text{out}}, W_{\text{out}})`.

    Examples:
        >>> import numpy as np
        >>> inp = Tensor(np.arange(1 * 1 * 3 * 4).astype(np.float32).reshape(1, 1, 3, 4))
        >>> F.avg_pool2d(inp, kernel_size=2, stride=2, padding=[1,0], mode="average")
            Tensor([[[[0.25 1.25]
             [6.5  8.5 ]]]], device=xpux:0)
    """
    if stride is None:
        stride = kernel_size
    window_h, window_w = expand_hw(kernel_size)
    stride_h, stride_w = expand_hw(stride)
    padding_h, padding_w = expand_hw(padding)

    op = builtin.Pooling(
        window_h=window_h,
        window_w=window_w,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=padding_h,
        pad_w=padding_w,
        mode=mode,
        strategy=get_execution_strategy(),
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
        oshp: `(OH, OW)` size of the output shape.

    Returns:
        output tensor.
    """
    return adaptive_pool2d_cpp(inp, oshp, "MAX")


def adaptive_avg_pool2d(
    inp: Tensor, oshp: Union[Tuple[int, int], int, Tensor],
) -> Tensor:
    r"""Applies a 2D average adaptive pooling over an input.

    Refer to :class:`~.AvgAdaptivePool2d` for more information.

    Args:
        inp: input tensor.
        oshp: `(OH, OW)` size of the output shape.

    Returns:
        output tensor.
    """
    return adaptive_pool2d_cpp(inp, oshp, "AVERAGE")


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
        >>> import numpy as np
        >>> x = Tensor(np.arange(5).astype(np.float32))
        >>> out = F.hswish(x)
        >>> out.numpy().round(decimals=4)
        array([0.    , 0.6667, 1.6667, 3.    , 4.    ], dtype=float32)
    """
    return _elwise(x, mode=Elemwise.Mode.H_SWISH)


def sigmoid(x):
    r"""Element-wise `1 / ( 1 + exp( -x ) )`."""
    return _elwise(x, mode=Elemwise.Mode.SIGMOID)


def hsigmoid(x):
    r"""Element-wise `relu6(x + 3) / 6`."""
    return _elwise(x, mode=Elemwise.Mode.HSIGMOID)


def relu(x):
    r"""Element-wise `max(x, 0)`."""
    return _elwise(x, mode=Elemwise.Mode.RELU)


def relu6(x):
    r"""Element-wise `min(max(x, 0), 6)`."""
    return _elwise(x, mode=Elemwise.Mode.RELU6)


def prelu(x, y):
    r"""Element-wise `max(x, 0) + y * min(x, 0)`."""
    return _elwise(x, y, mode=Elemwise.Mode.PRELU)


def leaky_relu(inp: Tensor, negative_slope: float = 0.01) -> Tensor:
    r"""Element-wise LeakyReLU function

    Refer to :class:`~.LeakyReLU` for more information.
    """
    return _elwise(inp, negative_slope, mode=Elemwise.Mode.PRELU)


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
         >>> import numpy as np
         >>> x = Tensor(np.arange(-3, 3, dtype=np.float32))
         >>> y = F.softplus(x)
         >>> y.numpy().round(decimals=4)
         array([0.0486, 0.1269, 0.3133, 0.6931, 1.3133, 2.1269], dtype=float32)
    """
    return _elwise(inp, mode=Elemwise.Mode.SOFTPLUS)


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
        >>> import numpy as np
        >>> x = Tensor(np.arange(-5, 5, dtype=np.float32)).reshape(2,5)
        >>> y = F.logsoftmax(x, axis=1)
        >>> y.numpy().round(decimals=4)
        array([[-4.4519, -3.4519, -2.4519, -1.4519, -0.4519],
               [-4.4519, -3.4519, -2.4519, -1.4519, -0.4519]], dtype=float32)
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
        >>> import numpy as np
        >>> x = Tensor(np.arange(-5, 5, dtype=np.float32))
        >>> y = F.logsigmoid(x)
        >>> y.numpy().round(decimals=4)
        array([-5.0067, -4.0182, -3.0486, -2.1269, -1.3133, -0.6931, -0.3133,
               -0.1269, -0.0486, -0.0181], dtype=float32)
    """
    return _elwise(inp, mode=Elemwise.Mode.LOGSIGMOID)


def logsumexp(
    inp: Tensor, axis: Union[int, Sequence[int]], keepdims: bool = False
) -> Tensor:
    r"""Calculates the logarithm of the inputs' exponential sum along the given :attr:`axis`.

    .. math::

        \text{logsumexp}(x)= \log \sum_{j=1}^{
    n} \exp \left(x_{
    j}\right)

    For numerical stability, the implementation follows this transformation:

    .. math::

        \text{logsumexp}(x)= \log \sum_{j=1}^{
    n} \exp \left(x_{
    j}\right)
        = \text{logsumexp}(x)=b+\log \sum_{j=1}^{
    n} \exp \left(x_{j}-b\right)

    where

    .. math::
        b = \max(x_j)

    Examples:
        >>> import numpy as np
        >>> x = Tensor(np.arange(-5, 5, dtype=np.float32)).reshape(2,5)
        >>> y = F.logsumexp(x, axis=1, keepdims=False)
        >>> y.numpy().round(decimals=4)
        array([-0.5481,  4.4519], dtype=float32)
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
        >>> import numpy as np
        >>> x = Tensor(np.arange(-5, 5, dtype=np.float32)).reshape(2,5)
        >>> out = F.softmax(x)
        >>> out.numpy().round(decimals=4)
        array([[0.0117, 0.0317, 0.0861, 0.2341, 0.6364],
               [0.0117, 0.0317, 0.0861, 0.2341, 0.6364]], dtype=float32)
    """
    if axis is None:
        axis = _get_softmax_axis(len(inp.shape))
    if isinstance(axis, list):
        offset = inp.max(axis=axis, keepdims=True).detach()
        cached = exp(inp - offset)
        down = sum(cached, axis=axis, keepdims=True)
        return cached / down
    else:
        op = builtin.Softmax(axis=axis,)
        (output,) = apply(op, inp)
        return output


def instance_norm(
    inp: Tensor,
    affine: bool,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
):
    r"""Applies instance normalization to the input.

    Refer to :class:`~.InstanceNorm` for more information.

    Args:
        inp: input tensor.
        affine: whether to use learnable affine parameters (weight, bias)
        weight: scaling tensor in the learnable affine parameters.
            See :math:`\gamma` in :class:`~.InstanceNorm`.
        bias: bias tensor in the learnable affine parameters.
            See :math:`\beta` in :class:`~.InstanceNorm`.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
    """
    op = builtin.InstanceNorm(affine=affine, eps=eps)
    if affine:
        assert weight is not None, "weight must be provided if affine is True"
        assert bias is not None, "bias must be provided if affine is True"
        return apply(op, inp, weight, bias)[0]
    else:
        return apply(op, inp)[0]


def group_norm(
    inp: Tensor,
    num_groups: int,
    affine: bool,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
):
    r"""Applies group normalization to the input.

    Refer to :class:`~.GroupNorm` for more information.

    Args:
        inp: input tensor.
        num_groups: number of groups to separate the channels into
            See :attr:`num_groups` in :class:`~.GroupNorm`.
        affine: whether to use learnable affine parameters (weight, bias)
        weight: scaling tensor in the learnable affine parameters.
            See :math:`\gamma` in :class:`~.GroupNorm`.
        bias: bias tensor in the learnable affine parameters.
            See :math:`\beta` in :class:`~.GroupNorm`.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
    """
    op = builtin.GroupNorm(affine=affine, eps=eps, group=num_groups,)
    if affine:
        assert weight is not None, "weight must be provided if affine is True"
        assert bias is not None, "bias must be provided if affine is True"
        return apply(op, inp, weight, bias)[0]
    else:
        return apply(op, inp)[0]


def layer_norm(
    inp: Tensor,
    normalized_shape: tuple,
    affine: bool,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
):
    r"""Applies layer normalization to the input.

    Refer to :class:`~.LayerNorm` for more information.

    Args:
        inp: input tensor.
        normalized_shape: the shape that you want to be normalizated
            See :attr:`normalized_shape` in :class:`~.LayerNorm`.
        affine: whether to use learnable affine parameters (weight, bias)
        weight: scaling tensor in the learnable affine parameters.
            See :math:`\gamma` in :class:`~.LayerNorm`.
        bias: bias tensor in the learnable affine parameters.
            See :math:`\beta` in :class:`~.LayerNorm`.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
    """
    if isinstance(normalized_shape, int):
        normalized_shape = [normalized_shape]

    normalized_dim = len(normalized_shape)
    assert normalized_dim > 0

    normalized_size = 1
    for i in range(normalized_dim):
        normalized_size = normalized_size * normalized_shape[i]

    op = builtin.LayerNorm(
        affine=affine,
        eps=eps,
        normalized_dim=normalized_dim,
        normalized_size=normalized_size,
    )
    if affine:
        assert weight is not None, "weight must be provided if affine is True"
        assert bias is not None, "bias must be provided if affine is True"
        return apply(op, inp, weight, bias)[0]
    else:
        return apply(op, inp)[0]


def general_norm(
    inp: Tensor,
    normalized_axis: tuple,
    affine: bool = True,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
):
    r"""Applies general normalization to the input.

    Refer to :class:`~.GeneralNorm` for more information.

    Args:
        inp: input tensor.
        normalized_axis: the axis that you want to be normalizated
            See :attr:`normalized_axis` in :class:`~.GeneralNorm`.
        affine: whether to use learnable affine parameters (weight, bias)
        weight: scaling tensor in the learnable affine parameters.
            See :math:`\gamma` in :class:`~.GeneralNorm`.
        bias: bias tensor in the learnable affine parameters.
            See :math:`\beta` in :class:`~.GeneralNorm`.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
    """
    if not isinstance(normalized_axis, Sequence):
        normalized_axis = [normalized_axis]
    assert isinstance(normalized_axis, (list, tuple))

    assert len(normalized_axis) > 0, "normalization axis not specified"

    normalized_axis = [num + inp.ndim if num < 0 else num for num in normalized_axis]
    assert normalized_axis == sorted(
        normalized_axis
    ), "The order of normalized_axis is incorrect, should be {}, but got {}. Please specify the values of axis in the correct order in normalized_axis".format(
        sorted(normalized_axis), normalized_axis
    )
    assert (
        normalized_axis[-1] < inp.ndim
    ), "the maximum axis in normalized_axis is greater than inp_shape.ndim"
    assert len(set(normalized_axis)) == len(
        normalized_axis
    ), "there are duplicate axis in normalized_axis"

    _reshape = []
    _rereshape = []
    _need_reshape = (
        _builtins_max(normalized_axis) - _builtins_min(normalized_axis)
    ) != (len(normalized_axis) - 1)
    if _need_reshape:
        get_logger().warning(
            "normalized_axis is discontinuous, and performance may be poor"
        )
        unnormalized_axis = list(set(range(inp.ndim)) - set(normalized_axis))
        unnormalized_axis.sort()
        _reshape = unnormalized_axis + normalized_axis
        inp = transpose(inp, _reshape)

        for i in range(inp.ndim):
            _rereshape.append(_reshape.index(i))

        normalized_axis = list(
            set(range(inp.ndim)) - set(range(len(unnormalized_axis)))
        )

    assert (_builtins_max(normalized_axis) - _builtins_min(normalized_axis)) == (
        len(normalized_axis) - 1
    )

    op = builtin.GeneralNorm(
        affine=affine,
        eps=eps,
        axis_start=_builtins_min(normalized_axis),
        axis_end=_builtins_max(normalized_axis) + 1,
    )
    if affine:
        assert weight is not None, "weight must be provided if affine is True"
        assert bias is not None, "bias must be provided if affine is True"
        out = apply(op, inp, weight, bias)[0]
    else:
        out = apply(op, inp)[0]

    if _need_reshape:
        out = transpose(out, _rereshape)
    return out


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
        compute_mode: When set to 'default', no special requirements will be
            placed on the precision of intermediate results. When set to 'float32',
            float32 would be used for accumulator and intermediate result, but only
            effective when input and output are of float16 dtype.
        param_dim: a value indicating in which format the parameters are.
            Default: 'dim_1c11', which means NCHW format.
            And 'dim_111c' means NHWC format.
    """

    def make_full_if_none(x, value):
        x_ndim = None if x is None else x.ndim
        # in general case, x will be returned here directly
        if x_ndim is not None and x_ndim != 1:
            return x

        C = inp.shape[1]
        pshape = (1, C, 1, 1)

        if x is None:
            x = Const(value, inp.dtype, inp.device)
            shape = astensor1d(pshape, inp, dtype="int32", device=inp.device)
            (result,) = apply(builtin.Broadcast(), x, shape)
            result.format = inp.format
            return result
        else:
            assert x_ndim == 1
            shape = astensor1d(pshape, inp, dtype="int32", device=inp.device)
            (result,) = apply(builtin.Reshape(), x, shape)
            return result

    has_mean = running_mean is not None
    has_var = running_var is not None

    if not training:
        assert has_mean, "running_mean must be provided in inference mode"
        assert has_var, "running_var must be provided in inference mode"

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
    # fmt : off
    @subgraph("SyncBnStage0", dtype, device, 1)
    def syncbn_stage0(inputs, f, c):
        input = inputs[0]
        reduce_shape = c(
            (1, channels) + (1,) * (ndim - 2), dtype="int32", device=device
        )
        input_shape = f(GetVarShape(), input)
        input_elems = f(Reduce(mode="product", axis=0), input_shape)
        reduce_elems = f(Reduce(mode="product", axis=0), reduce_shape)
        reduce_size = f("//", input_elems, reduce_elems)
        channel_x1s = f(Reduce(mode="sum"), input, reduce_shape)
        channel_x2s = f(Reduce(mode="sum_sqr"), input, reduce_shape)
        reduce_size_f = f(TypeCvt(dtype=dtype), reduce_size)
        return (
            (reduce_shape, reduce_size_f, channel_x1s, channel_x2s),
            (False, False, True, True),
        )

    @subgraph("SyncBnStage1", dtype, device, 7)
    def syncbn_stage1(inputs, f, c):
        input, reduce_size, channel_x1s, channel_x2s, eps = inputs[0:5]
        weight, bias = inputs[5:7]
        channel_mean = f("/", channel_x1s, reduce_size)
        channel_var = f(
            "+",
            f(
                "/",
                f("**", channel_x1s, c(2)),
                f("-", f("*", reduce_size, reduce_size)),
            ),
            f("/", channel_x2s, reduce_size),
        )
        invsqrt_channel_var = f("**", f(eps_mode, channel_var, eps), c(-0.5))
        inv_var_wt = f("*", invsqrt_channel_var, weight)
        neg_channel_mean = f("-", channel_mean)
        outvar = f(
            "fma3",
            input,
            inv_var_wt,
            f("+", f("*", neg_channel_mean, inv_var_wt), bias),
        )
        return (outvar, channel_mean, channel_var), (True, True, True)

    @subgraph("SyncBnStage1Inference", dtype, device, 6)
    def syncbn_stage1_inference(inputs, f, c):
        input, channel_mean, channel_var, eps = inputs[0:4]
        weight, bias = inputs[4:6]
        invsqrt_channel_var = f("**", f(eps_mode, channel_var, eps), c(-0.5))
        inv_var_wt = f("*", invsqrt_channel_var, weight)
        neg_channel_mean = f("-", channel_mean)
        outvar = f(
            "+",
            f("*", input, inv_var_wt),
            f("+", f("*", neg_channel_mean, inv_var_wt), bias),
        )
        return (outvar,), (True,)

    @subgraph("SyncBnStage2", dtype, device, 7)
    def syncbn_stage2(inputs, f, c):
        running_mean, running_var, momentum = inputs[0:3]
        reduce_size, channel_x1s, channel_x2s, channel_mean = inputs[3:7]
        c1_minus_momentum = f("-", c(1), momentum)
        reduce_size_minus_c1 = f("-", reduce_size, c(1))
        running_mean = f(
            "fma4", running_mean, momentum, c1_minus_momentum, channel_mean,
        )
        channel_variance_unbiased = f(
            "+",
            f(
                "/",
                f("**", channel_x1s, c(2)),
                f("*", f("-", reduce_size), reduce_size_minus_c1),
            ),
            f("/", channel_x2s, reduce_size_minus_c1),
        )
        running_var = f(
            "fma4", running_var, momentum, c1_minus_momentum, channel_variance_unbiased
        )
        return (running_mean, running_var), (True, True)

    @subgraph("SyncBnConcatStats", dtype, device, 3)
    def syncbn_concat_stats(inputs, f, c):
        reduce_size, channel_x1s, channel_x2s = inputs[0:3]
        reduce_size = f(builtin.Broadcast(), reduce_size, c([1] * ndim, dtype="int32"))
        stats = f(
            builtin.Concat(axis=1, comp_node=device),
            reduce_size,
            channel_x1s,
            channel_x2s,
        )
        return (stats,), (True,)

    @subgraph("SyncBnSplitStats", dtype, device, 1)
    def syncbn_split_stats(inputs, f, c):
        stats = inputs[0]
        c_1 = c(1, dtype="int32")
        channel_x1s_end = c(channels + 1, dtype="int32")

        def _subtensor(src, axis, begin, end):
            items = ((axis, (begin is not None), (end is not None), False, False),)
            args = ()
            if begin is not None:
                args += (begin,)
            if end is not None:
                args += (end,)
            return f(builtin.Subtensor(items=items), src, *args)

        reduce_size = _subtensor(stats, 1, None, c_1)
        channel_x1s = _subtensor(stats, 1, c_1, channel_x1s_end)
        channel_x2s = _subtensor(stats, 1, channel_x1s_end, None)
        reduce_size = f(builtin.Reshape(), reduce_size, c_1)
        return (reduce_size, channel_x1s, channel_x2s), (False, True, True)

    # fmt : on
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
    if amp._enabled:
        inp, weight, bias, running_mean, running_var = cast_tensors(
            inp, weight, bias, running_mean, running_var, promote=True
        )

    _channels = make_shape_tuple(inp.shape)[1]
    _ndim = inp.ndim
    _device = inp.device
    _dtype = inp.dtype

    if _ndim != 4:
        raise NotImplementedError("sync_batch_norm for ndim != 4")

    def _make_full_if_none(x, value):
        if x is None:
            x = Const(value, inp.dtype, _device)
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
    # - channel_mean * invsqrt_channel_variance
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
    if amp._enabled:
        outvar = outvar.astype("float16")
    return outvar


def dropout(inp: Tensor, drop_prob: float, training: bool = True) -> Tensor:
    r"""Returns a new tensor where each of the elements are randomly set to zero
    with probability P = ``drop_prob``. Optionally rescale the output tensor if ``training`` is True.

    Args:
        inp: input tensor.
        drop_prob: probability to drop (set to zero) a single element.
        training: the default behavior of ``dropout`` during training is to rescale the output,
            then it can be replaced by an :class:`~.module.identify.Identity` during inference. Default: True
    Returns:
        the ouput tensor

    Examples:
        >>> import numpy as np
        >>> data = Tensor(np.ones(10000000, dtype=np.float32))
        >>> out = F.nn.dropout(data, 1.0 / 3.0, training=True)
        >>> assert not out.numpy().all()
        >>> out = F.nn.dropout(data, 1.0 / 3.0, training=False)
        >>> assert out.numpy().all()
        >>> out.numpy()
        array([1., 1., 1., ..., 1., 1., 1.], dtype=float32)
    """
    assert 0 <= drop_prob < 1
    if not training or drop_prob == 0:
        return inp

    # model in training mode, e.g.model.train()
    op = Dropout(drop_prob=drop_prob, seed=_get_global_rng_seed(), handle=0)
    outputs = apply(op, inp)
    return outputs[0]


def one_hot(inp: Tensor, num_classes: int) -> Tensor:
    r"""Performs one-hot encoding for the input tensor.

    Args:
        inp: input tensor.
        num_classes: number of classes denotes the last dimension of the output tensor.

    Examples:
        >>> import numpy as np
        >>> x = Tensor(np.arange(1, 4, dtype=np.int32))
        >>> F.one_hot(x, num_classes=4)
        Tensor([[0 1 0 0]
         [0 0 1 0]
         [0 0 0 1]], dtype=int32, device=xpux:0)
    """
    zeros_tensor = zeros(
        list(inp.shape) + [num_classes], dtype=inp.dtype, device=inp.device
    )
    ones_tensor = ones(list(inp.shape) + [1], dtype=inp.dtype, device=inp.device)

    op = builtin.IndexingSetOneHot(axis=inp.ndim, ndim=inp.ndim)
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
        >>> src = Tensor([[1.0, 2.0]])
        >>> index = Tensor([0])
        >>> val = F.indexing_one_hot(src, index)
        >>> val.numpy()
        array([1.], dtype=float32)
    """
    assert isinstance(src, Tensor), "src must be of Tensor type"
    op = builtin.IndexingOneHot(axis=axis, ndim=src.ndim)
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

    Refer to :class:`~.module.sliding_window.SlidingWindow` for more information.

    Args:
        inp: input tensor.
        kernel_size: size of the window.
        padding: implicit zero padding added on both sides of input. Default: 0
        stride: stride of the window. Default: 1
        dilation: dilation of the window. Default: 1
    """
    padding_h, padding_w = expand_hw(padding)
    stride_h, stride_w = expand_hw(stride)
    dilation_h, dilation_w = expand_hw(dilation)
    window_h, window_w = expand_hw(kernel_size)

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

    Refer to :class:`~.module.sliding_window.SlidingWindowTranspose` for more information.

    Args:
        inp: input tensor.
        output_size: shape of output tensor.
        kernel_size: size of the window.
        padding: implicit zero padding added on both sides of input. Default: 0
        stride: stride of the window. Default: 1
        dilation: dilation of the window. Default: 1
    """
    output_h, output_w = expand_hw(output_size)
    padding_h, padding_w = expand_hw(padding)
    stride_h, stride_w = expand_hw(stride)
    dilation_h, dilation_w = expand_hw(dilation)
    window_h, window_w = expand_hw(kernel_size)

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
    pad_width: Tuple[Tuple[int, int], ...],
    mode: str = "constant",
    constant_value: float = 0.0,
) -> Tensor:
    r"""Pads the input tensor.

    Args:
        pad_width: A tuple. Each element in the tuple is the tuple of 2-elements,
            the 2 elements represent the padding size on both sides of the current dimension, ``(front_offset, back_offset)``
        mode: One of the following string values. Default: ``'constant'``

            * ``'constant'``: Pads with a constant value.
            * ``'reflect'``: Pads with the reflection of the tensor mirrored on the first and last values of the tensor along each axis.
            * ``'replicate'``: Pads with the edge values of tensor.
        constant_val: Fill value for ``'constant'`` padding. Default: 0

    Examples:
        >>> import numpy as np
        >>> inp = Tensor([[1., 2., 3.],[4., 5., 6.]])
        >>> inp
        Tensor([[1. 2. 3.]
         [4. 5. 6.]], device=xpux:0)
        >>> F.nn.pad(inp, pad_width=((1, 1),), mode="constant")
        Tensor([[0. 0. 0.]
         [1. 2. 3.]
         [4. 5. 6.]
         [0. 0. 0.]], device=xpux:0)
        >>> F.nn.pad(inp, pad_width=((1, 1),), mode="constant", constant_value=9)
        Tensor([[9. 9. 9.]
         [1. 2. 3.]
         [4. 5. 6.]
         [9. 9. 9.]], device=xpux:0)
        >>> F.nn.pad(inp, pad_width=((1, 1), (1, 2)), mode="reflect")
        Tensor([[5. 4. 5. 6. 5. 4.]
         [2. 1. 2. 3. 2. 1.]
         [5. 4. 5. 6. 5. 4.]
         [2. 1. 2. 3. 2. 1.]], device=xpux:0)
        >>> F.nn.pad(inp, pad_width=((1, 1), (1, 2)), mode="replicate")
        Tensor([[1. 1. 2. 3. 3. 3.]
         [1. 1. 2. 3. 3. 3.]
         [4. 4. 5. 6. 6. 6.]
         [4. 4. 5. 6. 6. 6.]], device=xpux:0)

    """
    p_offsets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    assert mode.lower() in ["constant", "edge", "replicate", "reflect"]

    if mode.lower() == "edge":
        mode = "replicate"

    for i in range(0, len(pad_width)):
        p_offsets[i * 2] = pad_width[i][0]
        p_offsets[i * 2 + 1] = pad_width[i][1]

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
        >>> import numpy as np
        >>> inp = Tensor(np.arange(25, dtype=np.float32).reshape(1,1,5,5))
        >>> GT = np.array([[[[ 0.,         0.999925,   1.9994003,  2.9979765,  3.9952066],
        ...                  [ 4.9906454,  5.983851,   6.974385,   7.961814,   8.945709 ],
        ...                  [ 9.925651,  10.90122,   11.872011,  12.837625,  13.7976675],
        ...                  [14.751757,  15.699524,  16.640602,  17.574642,  18.501305 ],
        ...                  [19.420258,  20.331186,  21.233786,  22.127764,  23.012836 ]]]])
        >>> out = F.local_response_norm(inp, kernel_size=3, k=1.0, alpha=1e-4, beta=0.75)
        >>> np.testing.assert_allclose(GT, out.numpy(), rtol=1e-6, atol=1e-6)
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


def layerPixelShuffle_traceable(inp, upscale_factor):
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
        int(shape_ori[-3] / square),
        shape_ori[-2] * upscale_factor,
        shape_ori[-1] * upscale_factor,
    )

    dim_order = (0, 1, 4, 2, 5, 3)

    layerPixelShuffle = _get_layerPixelShuffle(_device, _dtype, dim_order)

    shape_0 = convert_single_value(shape_0, device=inp.device)
    shape_1 = convert_single_value(shape_1, device=inp.device)
    outvar, *_ = apply(layerPixelShuffle(), inp, shape_0, shape_1)

    return outvar


def pixel_shuffle(inp: Tensor, upscale_factor: int) -> Tensor:
    """
    Rearranges elements in a tensor of shape `(..., C * r^2, H, W)` to a tensor of
    shape `(..., C, H * r, W * r)`, where `r` is an upscale factor, where `...` is
    zero or more batch dimensions.

    :param inp: input tensor.
    :param upscale_factor: upscale factor of pixel_shuffle.
    :return: output tensor.
    """
    return pixel_shuffle_cpp(inp, upscale_factor, layerPixelShuffle_traceable)


def region_restricted_conv(
    inp: Tensor,
    weight: Tensor,
    rin: Tensor,
    rout: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int, int]] = 1,
    padding: Union[int, Tuple[int, int, int]] = 0,
    dilation: Union[int, Tuple[int, int, int]] = 1,
    groups: int = 1,
    conv_mode: str = "cross_correlation",
    compute_mode="default",
) -> Tensor:
    r"""Region Restricted convolution operation.

    Refer to :class:`~.RegionRestrictedConv` for more information.

    Args:
        inp: feature map of the convolution operation.
        weight: convolution kernel.
        rin: input mask
        rout: output mask
        bias: bias added to the result of convolution (if given).
        stride: stride of the 2D region restricted convolution operation. Default: 1
        padding: size of the paddings added to the input on both sides of its
            spatial dimensions. Only zero-padding is supported. Default: 0
        dilation: dilation of the 2D convolution operation. Default: 1
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

    pad_h, pad_w = expand_hw(padding)
    stride_h, stride_w = expand_hw(stride)
    dilate_h, dilate_w = expand_hw(dilation)

    sparse_type = "group"
    assert groups > 0, (
        "RegionRestrictedConv expected grouped conv mode, \
    which requires groups > 0, but got groups=%d"
        % (groups)
    )
    op = builtin.RegionRestrictedConvolution(
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dilate_h=dilate_h,
        dilate_w=dilate_w,
        mode=conv_mode,
        compute_mode=compute_mode,
        sparse=sparse_type,
    )
    (output,) = apply(op, inp, weight, rin, rout)
    if bias is not None:
        output += bias
    return output


def _mha_shape_check(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_padding_mask: Optional[Tensor],
    attn_mask: Optional[Tensor],
    num_heads: int,
):
    # Verifies the expected shape for `query, `key`, `value`, `key_padding_mask` and `attn_mask`
    # and returns if the input is batched or not.
    # Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.
    q_dim = query.ndim
    k_dim = key.ndim
    v_dim = value.ndim
    kpm_dim = key_padding_mask.ndim if key_padding_mask is not None else 0
    kpm_shape = tuple(key_padding_mask.shape) if key_padding_mask is not None else None
    am_dim = attn_mask.ndim if attn_mask is not None else 0
    am_shape = tuple(attn_mask.shape) if attn_mask is not None else None
    # Shape check.
    if q_dim == 3:
        # Batched Inputs
        is_batched = True
        assert k_dim == 3 and v_dim == 3, (
            "For batched (3-D) `query`, expected `key` and `value` to be 3-D"
            f" but found {k_dim}-D and {v_dim}-D tensors respectively"
        )
        q_shape0, q_shape1, _ = query.shape
        k_shape0, k_shape1, _ = key.shape
        v_shape0, v_shape1, _ = value.shape
        assert q_shape0 == k_shape0 and k_shape0 == v_shape0, (
            "For batched (3-D) `query`, expected the batch sizes of `query`, `key` and `value` to be equal"
            f" but found query batch size is {q_shape0}, key batch size is {k_shape0} and value batch size is {v_shape0} respectively"
        )
        assert k_shape1 == v_shape1, (
            "For batched (3-D) `query`, expected the sequence length of `key` and `value` to be equal"
            f" but found key seqlen is {k_shape1} and value seqlen is {v_shape1} respectively"
        )
        if key_padding_mask is not None:
            assert kpm_dim == 2, (
                "For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
                f" but found {kpm_dim}-D tensor instead"
            )
            assert (kpm_shape[0] == k_shape0 and kpm_shape[1] == k_shape1) or (
                kpm_shape[0] == 2 and kpm_shape[1] == k_shape0
            ), (
                f"For batched (3-D) `query`, expected `key_padding_mask.shape` equal {k_shape0, k_shape1} or {2, k_shape0}"
                f" but found {kpm_shape} instead"
            )
        if attn_mask is not None:
            assert am_dim in (2, 3), (
                "For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                f" but found {am_dim}-D tensor instead"
            )
            if am_dim == 2:
                assert (am_shape[0] == q_shape1 and am_shape[1] == k_shape1) or (
                    am_shape[0] == 2 and am_shape[1] == q_shape1
                ), f"Expected `attn_mask` shape to be {q_shape1, k_shape1} or {2, q_shape1} but got {am_shape}"
            if am_dim == 3:
                assert (
                    am_shape[0] == q_shape0 * num_heads
                    and am_shape[1] == q_shape1
                    and am_shape[2] == k_shape1
                ), f"Expected `attn_mask` shape to be {q_shape0 * num_heads, q_shape1, k_shape1} but got {am_shape}"
    elif q_dim == 2:
        # Unbatched Inputs
        is_batched = False
        assert k_dim == 2 and v_dim == 2, (
            "For unbatched (2-D) `query`, expected `key` and `value` to be 2-D"
            f" but found {k_dim}-D and {v_dim}-D tensors respectively"
        )
        q_shape0, q_shape1 = query.shape
        k_shape0, k_shape1 = key.shape
        v_shape0, v_shape1 = value.shape
        assert k_shape0 == v_shape0, (
            "For unbatched (3-D) `query`, expected the sequence length of `key` and `value` to be equal"
            f" but found key seqlen is {k_shape0} and query seqlen is {v_shape0} respectively"
        )
        if key_padding_mask is not None:
            assert kpm_dim in (1, 2), (
                "For unbatched (2-D) `query`, expected `key_padding_mask` to be `None`, 1-D or 2-D"
                f" but found {kpm_dim}-D tensor instead"
            )
            assert (kpm_dim == 1 and kpm_shape[0] == k_shape0) or (
                kpm_dim == 2 and kpm_shape[0] == 2,
                kpm_shape[1] == 1,
            ), (
                f"For batched (3-D) `query`, expected `key_padding_mask.shape` equal {k_shape0} or {2,1}"
                f" but found {kpm_shape} tensor instead"
            )
        if attn_mask is not None:
            assert am_dim in (2, 3), (
                "For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                f" but found {am_dim}-D tensor instead"
            )
            if am_dim == 2:
                assert (am_shape[0] == q_shape0 and am_shape[1] == k_shape0) or (
                    am_shape[0] == 2 and am_shape[1] == q_shape0
                ), f"Expected `attn_mask` shape to be {q_shape0, k_shape0} or {2, q_shape0} but got {am_shape}"
            if am_dim == 3:
                assert (
                    am_shape[0] == num_heads
                    and am_shape[1] == q_shape0
                    and am_shape[2] == k_shape0
                ), f"Expected `attn_mask` shape to be {num_heads, q_shape0, k_shape0} but got {am_shape}"
    else:
        raise AssertionError(
            f"query should be unbatched 2D or batched 3D tensor but received {q_dim}-D query tensor"
        )

    return is_batched


def _canonical_mask(
    mask: Optional[Tensor],
    mask_name: str,
    other_type,
    other_name: str,
    target_type,
    check_other: bool = True,
    maybe_cudnn_style_mask=False,
) -> Optional[Tensor]:
    if mask is not None and not maybe_cudnn_style_mask:
        _mask_dtype = mask.dtype
        _mask_is_float = (
            _mask_dtype == np.float16
            or _mask_dtype == np.float32
            or _mask_dtype == np.float64
        )
        assert (
            _mask_dtype == bool or _mask_is_float
        ), f"only bool and floating types of {mask_name} are supported"
        if check_other and other_type is not None:
            if _mask_dtype != other_type:
                get_logger().warning(
                    f"Support for mismatched {mask_name} and {other_name} "
                    "is deprecated. Use same type for both instead."
                )
        if not _mask_is_float:
            mask_ = zeros_like(mask).astype(target_type)
            mask_[mask] = float("-inf")
            return mask_
    return mask


def _merge_masks(
    attn_mask: Tensor,
    key_padding_mask: Tensor,
    query: Tensor,
    key: Tensor,
    add_bias_kv: bool = False,
    add_zero_attn: bool = False,
    is_causal: bool = False,
    maybe_cudnn_style_mask: bool = False,
    num_heads: int = 0,
):
    r"""
    Determine mask type and combine masks if necessary.

    Note: This function will continue to improve with the iteration of MHA.

    Args:
        attn_mask: MHA's attention mask tensor, the shape is :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`
        key_padding_mask: MHA's padding mask tensor, the shape is :math:`(N, S)`
        query: MHA's query, the shape is :math:`(N, L, E_q)`
        key: MHA's key, the shape is :math:`(N, S, E_k)`
        add_bias_kv: used to determine whether pad is needed on the sequence dimension of attn_mask and key_padding_mask, from MHA's ``add_bias_kv``.
        add_zero_attn: used to determine whether pad is needed on the sequence dimension of attn_mask and key_padding_mask, from MHA's ``add_zero_attn``.
        is_causal: MHA's is_causal, is_causal provides a hint that attn_mask is the causal mask.
        maybe_cudnn_style_mask: MHA's maybe_cudnn_style_mask, like is_causal, maybe_cudnn_style_mask provides a hint that attn_mask and key_padding_mask is the cudnn style mask.
        num_heads: MHA's head number.
    Returns:
        merged_mask: merged mask, may be None, the shape is :math:`(L, S)`, :math:`(2\cdotL + 2\cdotN)` or :math:`(N\cdot\text{num\_heads}, L, S)`
        mask_type: merged mask type ``("no_mask", "default_mask", "cudnn_style_mask" or "user_defined_mask")``
    """
    mask_type = "no_mask"
    merged_mask = None
    batch_size = query.shape[0]
    seq_qlen = query.shape[1]
    seq_klen = key.shape[1]
    attn_mask_dim = attn_mask.ndim if attn_mask is not None else 0
    attn_mask_shape = attn_mask.shape if attn_mask is not None else 0
    key_padding_mask_shape = (
        key_padding_mask.shape if key_padding_mask is not None else 0
    )

    # is_causal is used to hint whether to use a causal mask, where the upper right triangle is all -inf,
    # and the diagonal and lower left triangle are all 0. But if attn_mask is given, attn_mask is used first.
    if (
        not maybe_cudnn_style_mask
        and is_causal
        and attn_mask is None
        and key_padding_mask is None
    ):
        # At this point, merged_mask = None
        mask_type = "default_mask"
    elif (
        not maybe_cudnn_style_mask
        and is_causal
        and attn_mask is not None
        and key_padding_mask is None
    ):
        # At this point, merged_mask = attn_mask
        mask_type = "default_mask"
        merged_mask = attn_mask
    elif maybe_cudnn_style_mask and not add_zero_attn and not add_bias_kv:
        # Please be careful, we only check if the shape is correct,
        # and we will not check if the values in attn_mask_tensor and key_padding_mask_tensor are correct.
        assert (
            attn_mask is not None and key_padding_mask is not None
        ), "if maybe_cudnn_style_mask, must given attn_mask and key_padding_mask."
        assert attn_mask_shape == (2, seq_qlen) and key_padding_mask_shape == (
            2,
            batch_size,
        )
        merged_mask = concat(
            (
                reshape(attn_mask, (2 * seq_qlen)),
                reshape(key_padding_mask, (2 * batch_size)),
            )
        ).astype("int32")
        mask_type = "cudnn_style_mask"
    else:
        if attn_mask is not None and key_padding_mask is None:
            # At this point, merged_mask = attn_mask
            mask_type = "user_defined_mask"
            merged_mask = attn_mask
        elif key_padding_mask is not None and attn_mask is None:
            mask_type = "user_defined_mask"
            # At this point, merged_mask.ndim = 4
            key_padding_mask_expanded = reshape(
                key_padding_mask, (batch_size, 1, 1, seq_klen)
            )
            key_padding_mask_expanded = broadcast_to(
                key_padding_mask_expanded, (None, num_heads, seq_qlen, None)
            )
            merged_mask = key_padding_mask_expanded
            merged_mask = reshape(
                merged_mask, (batch_size * num_heads, seq_qlen, seq_klen)
            )
        elif (attn_mask is not None) and (key_padding_mask is not None):
            # At this point, merged_mask.ndim = 3
            mask_type = "user_defined_mask"
            if attn_mask_dim == 2:
                key_padding_mask_expanded = reshape(
                    key_padding_mask, (batch_size, 1, seq_klen)
                )
                merged_mask = (attn_mask + key_padding_mask_expanded).reshape(
                    (batch_size, 1, seq_qlen, seq_klen)
                )
                attn_mask_expanded = broadcast_to(
                    merged_mask, (None, num_heads, None, None)
                )
                merged_mask = reshape(
                    attn_mask_expanded, (batch_size * num_heads, seq_qlen, seq_klen)
                )
            else:
                key_padding_mask_expanded = reshape(
                    key_padding_mask, (batch_size, 1, 1, seq_klen)
                )
                attn_mask_expanded = reshape(
                    attn_mask, (batch_size, num_heads, seq_qlen, seq_klen)
                )
                merged_mask = attn_mask_expanded + key_padding_mask_expanded
                merged_mask = reshape(
                    merged_mask, (batch_size * num_heads, seq_qlen, seq_klen)
                )
    return merged_mask, mask_type


def multi_head_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim: int,
    num_heads: int,
    attn_drop: float,
    out_drop: float,
    io_weight_bias: Optional[Tensor],
    qproj_size: Optional[int] = None,
    kproj_size: Optional[int] = None,
    vproj_size: Optional[int] = None,
    oproj_size: Optional[int] = None,
    qbias: bool = False,
    kbias: bool = False,
    vbias: bool = False,
    obias: bool = False,
    bias_k: Optional[Tensor] = None,
    bias_v: Optional[Tensor] = None,
    add_zero_attn: bool = False,
    key_padding_mask: Optional[Tensor] = None,
    attn_mask: Optional[Tensor] = None,
    need_weights: bool = False,
    average_attn_weights: bool = False,
    is_causal: bool = False,
    maybe_cudnn_style_mask: bool = False,
    reslink: bool = False,
    training: bool = True,
):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    .. math::
        \text{MultiHeadAttn}\big(q, k, v, W_Q, W_K, W_V, W_O\big) = \sum^{nHeads-1}_{i=0}W_{O,i}h_i

    where :math:`h_i=W_{V,i}v \text{Softmax}\Big( \text{smScaler} \cdot k^TW^T_{K,i}W_{Q,i}q \Big),\text{for }i\text{ = 0 ... nHeads-1}`.

    See :class:`~.module.MultiHeadAttn` for more details.

    Note: This API is experimental, and there is a possibility of subsequent changes.

    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        attn_drop: probability of an element to be zeroed, used in attention matrix.
        out_drop: probability of an element to be zeroed, used in final output.
        io_weight_bias: input/output projection weight/bias all in one.
            The order of arrangement is: query weight, key weight, value weight, out weight, query bias, key bias, value bias, out bias, the following parameters will be used to indicate whether these items exist: qproj_size, kproj_size, vproj_size, oproj_size, qbias, kbias, vbias, obias.
            Note: :math:`Y=X@W+B` is used here instead of :math:`Y=X@W^T+B` in pytorch.
        qproj_size: indicates the projection size of query weight in io_weight_bias, 0 indicates disabled query projection and no query projection weight.
        kproj_size: indicates the projection size of key weight in io_weight_bias, 0 indicates disabled key projection and no key projection weight.
        vproj_size: indicates the projection size of value weight in io_weight_bias, 0 indicates disabled value projection and no value projection weight.
        oproj_size: indicates the projection size of out weight in io_weight_bias, 0 indicates disabled output projection and no output projection weight.
        qbias: indicates whether there is a query bias in io_weight_bias, this parameter is only valid when qproj_size > 0.
        kbias: indicates whether there is a key bias in io_weight_bias, this parameter is only valid when kproj_size > 0.
        vbias: indicates whether there is a value bias in io_weight_bias, this parameter is only valid when vproj_size > 0.
        obias: indicates whether there is a out bias in io_weight_bias, this parameter is only valid when oproj_size > 0.
        bias_k, bias_v: the bias of the key and value sequences to be added at sequence dim. distinguished from kbias and vbias, bias_kv here is not kbias and vbias in the linear layer, and bias_kv here will be added to the K and V at sequence dimensions, where K and V are the matrices of key and value after projection, and K and V will be used to calculate the attention matrix.
            Note: Should be set to None, and configuration of this parameter is not supported now. The reason is that there is only cudnn implementation now, and we may try to loosen this option after submitting the commit that adds MHA proxy implementation.
        add_zero_attn: if specified, adds a new batch of zeros to the key and value sequences at sequence dim. Default: ``False``.
            Note: should be set to False, and configuration of this parameter is not supported now. The reason is that there is only cudnn implementation now, and we may try to loosen this option after submitting the commit that adds MHA proxy implementation.
        key_padding_mask: if specified, a mask of shape :math:`(N, S)` indicating which elements within ``key`` to ignore for the purpose of
            attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`. Binary and float masks are supported. For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        need_weights: indicates whether to return the attention weight, which is the output result of softmax. Default: `False`
        average_attn_weights: if true, indicates that the returned ``attn_weights`` should be averaged across
            heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
            effect when ``need_weights=True``. Default: ``False`` (i.e. average weights across heads)
        is_causal: if specified, applies a causal mask as attention mask. Default: ``False``
            Warning: ``is_causal`` provides a hint that ``attn_mask`` is the causal mask. Providing incorrect hints can result in incorrect execution, including forward and backward compatibility.
        maybe_cudnn_style_mask: if specified, applies a cudnn style mask as attention mask. Default: ``False``
            Note: In the cudnn style, the shape of the attn_mask is :math:`(2, L)`, and the shape of the key_padding_mask is :math:`(2, N)`.
            Warning: like is_causal, maybe_cudnn_style_mask provides a hint that attn_mask and key_padding_mask is a cudnn style mask. Providing incorrect hints can result in incorrect execution, including forward and backward compatibility. In addition, if the ``_merge_masks`` function returns ``merge_type=cudnn_style_mask``, please ensure that other conditions are correct so that it can run the implementation of cudnn, otherwise an error will be reported.
        reslink: add input query to final output.
            Note: It is only valid if the input query is the same as the shape of the output. Should be set to False, and configuration of this parameter is not supported now. The reason is that there is only cudnn implementation now, and we may try to loosen this option after submitting the commit that adds MHA proxy implementation.
        training: will apply dropout if is ``True``.

    Outputs:
        - **out[0]=attn_output** - Attention outputs of shape :math:`(N, L, E)`,
          where :math:`L` is the target sequence length, :math:`N` is
          the batch size, and :math:`E` is the embedding dimension ``embed_dim``.
        - **out[1]=attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
          returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N * \text{num\_heads}, L, S)`.
        - **out[2]=mask_reversespace** - Used to save the dropout mask needed for backward propagation.,
        - **out[3]=othr_reversespace** - Used to save the intermediate results that need to be used in backward propagation.,
    """
    qproj_size = embed_dim if qproj_size is None else qproj_size
    kproj_size = embed_dim if kproj_size is None else kproj_size
    vproj_size = embed_dim if vproj_size is None else vproj_size
    oproj_size = embed_dim if oproj_size is None else oproj_size
    if qbias:
        assert (
            qproj_size is not None and qproj_size > 0
        ), "when query projection bias is true, query projection weight must be given."
    if kbias:
        assert (
            kproj_size is not None and kproj_size > 0
        ), "when key projection bias is true, key projection weight must be given"
    if vbias:
        assert (
            vproj_size is not None and vproj_size > 0
        ), "when value projection bias is true, value projection weight must be given"
    if obias:
        assert (
            oproj_size is not None and oproj_size > 0
        ), "when output projection bias is true, output projection weight must be given"
    unsupport_reason = " The reason is that there is only cudnn implementation now, and we may try to loosen this option after submitting the commit that adds MHA proxy implementation."
    assert add_zero_attn is False, (
        "add_zero_attn should be False, and configuration of this parameter is not supported now."
        + unsupport_reason
    )
    assert bias_k is None, (
        "bias_k should be None, and configuration of this parameter is not supported now."
        + unsupport_reason
    )
    assert bias_v is None, (
        "bias_v should be None, and configuration of this parameter is not supported now."
        + unsupport_reason
    )
    assert reslink is False, (
        "reslink should be False, and configuration of this parameter is not supported now."
        + unsupport_reason
    )
    head_dim = embed_dim if qproj_size == 0 else embed_dim // num_heads
    smScaler = head_dim ** -0.5
    k_size = key.shape[2]
    v_size = value.shape[2]

    is_batched = _mha_shape_check(
        query, key, value, key_padding_mask, attn_mask, num_heads
    )
    if not is_batched:
        query = expand_dims(query, 0)
        key = expand_dims(key, 0)
        value = expand_dims(value, 0)
        if key_padding_mask is not None:
            key_padding_mask = expand_dims(key_padding_mask, 0)

    key_padding_mask = _canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=attn_mask,
        other_name="attn_mask",
        target_type=query.dtype,
        maybe_cudnn_style_mask=maybe_cudnn_style_mask,
    )
    attn_mask = _canonical_mask(
        mask=attn_mask,
        mask_name="attn_mask",
        other_type=None,
        other_name="",
        target_type=query.dtype,
        check_other=False,
        maybe_cudnn_style_mask=maybe_cudnn_style_mask,
    )
    attn_mask_tensor, attn_mask_type = _merge_masks(
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        query=query,
        key=key,
        add_bias_kv=bias_k is not None and bias_v is not None,
        add_zero_attn=add_zero_attn,
        is_causal=is_causal,
        maybe_cudnn_style_mask=maybe_cudnn_style_mask,
        num_heads=num_heads,
    )

    def get_tensor_combination_type(attn_mask_tensor, bias_k, bias_v):
        bias_kv = bias_k is not None and bias_v is not None
        if not bias_kv and attn_mask_tensor is None:
            return "none"
        elif not bias_kv and attn_mask_tensor is not None:
            return "only_mask"
        elif bias_kv and attn_mask_tensor is None:
            return "only_biaskv"
        else:
            return "all"

    tensor_combination_type = get_tensor_combination_type(
        attn_mask_tensor, bias_k, bias_v
    )

    op = builtin.MultiHeadAttn(
        num_heads=num_heads,
        sm_scaler=smScaler,
        attn_prob=attn_drop,
        out_prob=out_drop,
        reslink=reslink,
        training=training,
        input_order=0,
        seed=_get_global_rng_seed(),
        attn_mask_type=attn_mask_type,
        add_zero_attn=add_zero_attn,
        embeding_size=embed_dim,
        k_size=k_size,
        v_size=v_size,
        qproj_size=qproj_size,
        kproj_size=kproj_size,
        vproj_size=vproj_size,
        oproj_size=oproj_size,
        qbias=qbias,
        kbias=kbias,
        vbias=vbias,
        obias=obias,
        need_weights=need_weights,
        tensor_combination_type=tensor_combination_type,
    )
    if tensor_combination_type == "none":
        out = apply(op, query, key, value, io_weight_bias)
    elif tensor_combination_type == "only_mask":
        out = apply(op, query, key, value, io_weight_bias, attn_mask_tensor)
    elif tensor_combination_type == "only_biaskv":
        out = apply(op, query, key, value, io_weight_bias, bias_k, bias_v)
    else:
        out = apply(
            op, query, key, value, io_weight_bias, attn_mask_tensor, bias_k, bias_v
        )
    if need_weights:
        if average_attn_weights:
            shape = out[1].shape
            out_weight = out[1].reshape(-1, num_heads, shape[-2], shape[-1])
            return out[0], out_weight.mean(axis=1)
        else:
            return out[0], out[1]
    else:
        return out[0], None


from .loss import *  # isort:skip
from .metric import *  # isort:skip
from .vision import *  # isort:skip
from .quantized import conv_bias_activation  # isort:skip
