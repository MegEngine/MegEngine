# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# pylint: disable=too-many-lines
from typing import Optional, Sequence, Tuple, Union

from ..core._imperative_rt import CompNode
from ..core._imperative_rt.core2 import apply
from ..core._trace_option import use_symbolic_shape
from ..core.ops import builtin
from ..core.ops.builtin import BatchNorm
from ..core.ops.special import Const
from ..core.tensor import megbrain_graph, utils
from ..core.tensor.utils import astensor1d
from ..distributed import WORLD, is_distributed
from ..jit.tracing import is_tracing
from ..random import uniform
from ..tensor import Tensor
from .debug_param import get_conv_execution_strategy
from .distributed import all_reduce_sum
from .elemwise import exp, floor, log, log1p, maximum, minimum, relu
from .math import argsort, max, prod, sum
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
from .types import _pair, _pair_nonzero

__all__ = [
    "adaptive_avg_pool2d",
    "adaptive_max_pool2d",
    "avg_pool2d",
    "batch_norm",
    "conv2d",
    "conv_transpose2d",
    "dot",
    "dropout",
    "indexing_one_hot",
    "leaky_relu",
    "local_conv2d",
    "logsigmoid",
    "logsumexp",
    "logsoftmax",
    "matmul",
    "max_pool2d",
    "one_hot",
    "prelu",
    "remap",
    "softmax",
    "softplus",
    "svd",
    "warp_perspective",
    "conv1d",
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


def conv2d(
    inp: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    conv_mode="CROSS_CORRELATION",
    compute_mode="DEFAULT",
) -> Tensor:
    """
    2D convolution operation.

    Refer to :class:`~.Conv2d` for more information.

    :param inp: feature map of the convolution operation.
    :param weight: convolution kernel.
    :param bias: bias added to the result of convolution (if given).
    :param stride: stride of the 2D convolution operation. Default: 1
    :param padding: size of the paddings added to the input on both sides of its
        spatial dimensions. Only zero-padding is supported. Default: 0
    :param dilation: dilation of the 2D convolution operation. Default: 1
    :param groups: number of groups into which the input and output channels are divided, so as to perform a ``grouped convolution``. When ``groups`` is not 1,
        ``in_channels`` and ``out_channels`` must be divisible by ``groups``,
        and the shape of weight should be `(groups, out_channel // groups,
        in_channels // groups, height, width)`.
    :type conv_mode: string or :class:`Convolution.Mode`
    :param conv_mode: supports "CROSS_CORRELATION". Default:
        "CROSS_CORRELATION"
    :type compute_mode: string or
        :class:`Convolution.ComputeMode`
    :param compute_mode: when set to "DEFAULT", no special requirements will be
        placed on the precision of intermediate results. When set to "FLOAT32",
        "Float32" would be used for accumulator and intermediate result, but only
        effective when input and output are of Float16 dtype.
    :return: output tensor.
    """
    assert conv_mode == "CROSS_CORRELATION" or conv_mode.name == "CROSS_CORRELATION"
    assert compute_mode == "DEFAULT" or compute_mode.name == "DEFAULT"

    stride_h, stride_w = expand_hw(stride)
    pad_h, pad_w = expand_hw(padding)
    dilate_h, dilate_w = expand_hw(dilation)

    Sparse = builtin.Convolution.Sparse
    sparse_type = "DENSE" if groups == 1 else "GROUP"
    op = builtin.Convolution(
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dilate_h=dilate_h,
        dilate_w=dilate_w,
        strategy=get_conv_execution_strategy(),
        mode=conv_mode,
        compute_mode=compute_mode,
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
    conv_mode="CROSS_CORRELATION",
    compute_mode="DEFAULT",
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
    :param groups: number of groups into which the input and output channels are divided, so as to perform a ``grouped convolution``. When ``groups`` is not 1,
        ``in_channels`` and ``out_channels`` must be divisible by groups,
        and the shape of weight should be `(groups, out_channel // groups,
        in_channels // groups, height, width)`. Default: 1
    :type conv_mode: string or :class:`Convolution.Mode`
    :param conv_mode: supports "CROSS_CORRELATION". Default:
        "CROSS_CORRELATION"
    :type compute_mode: string or
        :class:`Convolution.ComputeMode`
    :param compute_mode: when set to "DEFAULT", no special requirements will be
        placed on the precision of intermediate results. When set to "FLOAT32",
        "Float32" would be used for accumulator and intermediate result, but only
        effective when input and output are of Float16 dtype.
    :return: output tensor.
    """
    assert conv_mode == "CROSS_CORRELATION" or conv_mode.name == "CROSS_CORRELATION"
    assert compute_mode == "DEFAULT" or compute_mode.name == "DEFAULT"

    if groups != 1:
        raise NotImplementedError("TODO")

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
        strategy=get_conv_execution_strategy(),
    )
    weight, inp = utils.convert_inputs(weight, inp)
    (output,) = apply(op, weight, inp)
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
    conv_mode="CROSS_CORRELATION",
):
    """Applies spatial 2D convolution over an groupped channeled image with untied kernels."""
    assert conv_mode == "CROSS_CORRELATION" or conv_mode.name == "CROSS_CORRELATION"

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
        compute_mode="DEFAULT",
        sparse="DENSE",
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
        mode="MAX",
    )
    (output,) = apply(op, inp)
    return output


def avg_pool2d(
    inp: Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    mode: str = "AVERAGE_COUNT_EXCLUDE_PADDING",
) -> Tensor:
    """
    Applies 2D average pooling over an input tensor.

    Refer to :class:`~.AvgPool2d` for more information.

    :param inp: input tensor.
    :param kernel_size: size of the window.
    :param stride: stride of the window. If not provided, its value is set to ``kernel_size``.
        Default: None
    :param padding: implicit zero padding added on both sides. Default: 0
    :param mode: whether to count padding values. Default: "AVERAGE_COUNT_EXCLUDE_PADDING"
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
    assert isinstance(inp, (Tensor, megbrain_graph.VarNode)), "inp must be Tensor type"
    if isinstance(oshp, int):
        oshp = (oshp, oshp)

    op = builtin.AdaptivePooling(mode="MAX", format="NCHW",)
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
    assert isinstance(inp, (Tensor, megbrain_graph.VarNode)), "inp must be Tensor type"
    if isinstance(oshp, int):
        oshp = (oshp, oshp)

    op = builtin.AdaptivePooling(mode="AVERAGE", format="NCHW",)
    oshp = astensor1d(oshp, inp, dtype="int32", device=inp.device)
    (output,) = apply(op, inp, oshp)
    return output


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
            fwd_mode=BatchNorm.FwdMode.INFERENCE, epsilon=eps, param_dim="DIM_1C11"
        )
        ret = apply(op, inp, weight, bias, running_mean, running_var)[-1]
        return ret

    else:
        op = builtin.BatchNorm(
            avg_factor=1 - momentum, epsilon=eps, param_dim="DIM_1C11"
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
    eps_mode="ADDITIVE",
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
    assert eps_mode in {"MAX", "ADDITIVE"}, "unknown eps_mode: {}".format(eps_mode)
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
            reduce_size = broadcast_to(Tensor(reduce_size, dtype=_dtype), [1] * _ndim)
            stat = concat(
                [reduce_size.astype(_dtype), channel_x1s, channel_x2s], axis=1
            )
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
        maximum(channel_variance, eps) if eps_mode == "MAX" else channel_variance + eps
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


def warp_perspective(
    inp: Tensor,
    M: Tensor,
    dsize: Union[Tuple[int, int], int, Tensor],
    border_mode: str = "REPLICATE",
    border_val: float = 0.0,
    interp_mode: str = "LINEAR",
) -> Tensor:
    r"""
    Applies perspective transformation to batched 2D images.

    The input images are transformed to the output images by the transformation matrix:

    .. math::
            \text{output}(n, c, h, w) = \text{input} \left( n, c,
                \frac{M_{00}h + M_{01}w + M_{02}}{M_{20}h + M_{21}w + M_{22}},
                \frac{M_{10}h + M_{11}w + M_{12}}{M_{20}h + M_{21}w + M_{22}}
                \right)

    :param inp: input image.
    :param M: `(batch, 3, 3)` transformation matrix.
    :param dsize: `(h, w)` size of the output image.
    :param border_mode: pixel extrapolation method.
        Default: "REPLICATE". Currently also support "CONSTANT", "REFLECT",
        "REFLECT_101", "WRAP".
    :param border_val: value used in case of a constant border. Default: 0
    :param interp_mode: interpolation methods.
        Default: "LINEAR". Currently only support "LINEAR" mode.
    :return: output tensor.

    Note:

    The transformation matrix is the inverse of that used by `cv2.warpPerspective`.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        inp_shape = (1, 1, 4, 4)
        x = tensor(np.arange(16, dtype=np.float32).reshape(inp_shape))
        M_shape = (1, 3, 3)
        # M defines a translation: dst(1, 1, h, w) = rst(1, 1, h+1, w+1)
        M = tensor(np.array([[1., 0., 1.],
                             [0., 1., 1.],
                             [0., 0., 1.]], dtype=np.float32).reshape(M_shape))
        out = F.warp_perspective(x, M, (2, 2))
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[[[ 5.  6.]
           [ 9. 10.]]]]

    """
    op = builtin.WarpPerspective(
        imode=interp_mode, bmode=border_mode, format="NCHW", border_val=border_val
    )
    inp, M = utils.convert_inputs(inp, M)
    dsize = astensor1d(dsize, inp, dtype="int32", device=inp.device)
    (result,) = apply(op, inp, M, dsize)
    return result


def remap(
    inp: Tensor,
    map_xy: Tensor,
    border_mode: str = "REPLICATE",
    scalar: float = 0.0,
    interp_mode: str = "LINEAR",
) -> Tensor:
    r"""
    Applies remap transformation to batched 2D images.

    The input images are transformed to the output images by the tensor map_xy.
    The output's H and W are same as map_xy's H and W.

    :param inp: input image
    :param map_xy: (batch, oh, ow, 2) transformation matrix
    :param border_mode: pixel extrapolation method.
        Default: "REPLICATE". Currently also support "CONSTANT", "REFLECT",
        "REFLECT_101", "WRAP".
    :param scalar: value used in case of a constant border. Default: 0
    :param interp_mode: interpolation methods.
        Default: "LINEAR". Currently only support "LINEAR" mode.
    :return: output tensor.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F
        inp_shape = (1, 1, 4, 4)
        inp = tensor(np.arange(16, dtype=np.float32).reshape(inp_shape))
        map_xy_shape = (1, 2, 2, 2)
        map_xy = tensor(np.array([[[1., 0.],[0., 1.]],
                            [[0., 1.],[0., 1.]]],
                             dtype=np.float32).reshape(map_xy_shape))
        out = F.remap(inp, map_xy)
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[[[1. 4.]
           [4. 4.]]]]

    """

    op = builtin.Remap(
        imode=interp_mode, border_type=border_mode, format="NCHW", scalar=scalar
    )
    assert isinstance(inp, (Tensor, megbrain_graph.VarNode)), "inp must be Tensor type"
    (result,) = apply(op, inp, map_xy)
    return result


def matmul(
    inp1: Tensor,
    inp2: Tensor,
    transpose_a=False,
    transpose_b=False,
    compute_mode="DEFAULT",
    format="DEFAULT",
) -> Tensor:
    """
    Performs a matrix multiplication of the matrices ``inp1`` and ``inp2``.

    With different inputs dim, this function behaves differently:

    - Both 1-D tensor, simply forward to ``dot``.
    - Both 2-D tensor, normal matrix multiplication.
    - If one input tensor is 1-D, matrix vector multiplication.
    - If at least one tensor are 3-dimensional or >3-dimensional, the other tensor should have dim >= 2, the batched matrix-matrix is returned, and the tensor with smaller dimension will
      be broadcasted. For example:
        - inp1: `(n, k, m)`, inp2: `(n, m, p)`, return: `(n, k, p)`
        - inp1: `(n, k, m)`, inp2: `(m, p)`, return: `(n, k, p)`
        - inp1: `(n, j, k, m)`, inp2: `(n, j, m, p)`, return: `(n, j, k, p)`

    :param inp1: first matrix to be multiplied.
    :param inp2: second matrix to be multiplied.
    :return: output tensor.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        data1 = tensor(np.arange(0, 6, dtype=np.float32).reshape(2, 3))
        data2 = tensor(np.arange(0, 6, dtype=np.float32).reshape(3, 2))
        out = F.matmul(data1, data2)
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[10. 13.]
         [28. 40.]]

    """
    remove_row, remove_col = False, False
    inp1, inp2 = utils.convert_inputs(inp1, inp2)

    dim1, dim2 = inp1.ndim, inp2.ndim
    # handle dim=1 cases, dot and matrix-vector multiplication
    if dim1 == 1 and dim2 == 1:
        return dot(inp1, inp2)
    # the underlying matmul op requires input dims to be at least 2
    if dim1 == 1:
        inp1 = expand_dims(inp1, 0)
        dim1 = 2
        remove_row = True
    if dim2 == 1:
        inp2 = expand_dims(inp2, 1)
        dim2 = 2
        remove_col = True

    batch_shape = None
    shape1 = inp1.shape
    shape2 = inp2.shape

    maxdim = dim1 if dim1 > dim2 else dim2
    if dim1 >= 3 or dim2 >= 3:
        if use_symbolic_shape():
            if dim1 > dim2:
                shape2 = concat([shape1[:-2], shape2[-2:]])
                inp2 = broadcast_to(inp2, shape2)
            if dim1 < dim2:
                shape1 = concat([shape2[:-2], shape1[-2:]])
                inp1 = broadcast_to(inp1, shape1)
            if maxdim > 3:
                batch_shape = shape1[:-2]
                # compress inputs to 3d
                (inp1,) = apply(
                    builtin.Reshape(), inp1, concat([prod(shape1[:-2]), shape1[-2:]])
                )
                (inp2,) = apply(
                    builtin.Reshape(), inp2, concat([prod(shape2[:-2]), shape2[-2:]])
                )
        else:
            if dim1 > dim2:
                shape2 = shape1[:-2] + shape2[-2:]
                inp2 = broadcast_to(inp2, shape2)
            if dim1 < dim2:
                shape1 = shape2[:-2] + shape1[-2:]
                inp1 = broadcast_to(inp1, shape1)
            if maxdim > 3:
                batch_shape = shape1[:-2]
                # compress inputs to 3d
                inp1 = inp1.reshape((-1, shape1[-2], shape1[-1]))
                inp2 = inp2.reshape((-1, shape2[-2], shape2[-1]))

        op = builtin.BatchedMatrixMul(
            transposeA=transpose_a,
            transposeB=transpose_b,
            compute_mode=compute_mode,
            format=format,
        )
    else:
        op = builtin.MatrixMul(
            transposeA=transpose_a,
            transposeB=transpose_b,
            compute_mode=compute_mode,
            format=format,
        )

    (result,) = apply(op, inp1, inp2)
    if maxdim > 3:
        if use_symbolic_shape():
            (result,) = apply(
                builtin.Reshape(), result, concat([batch_shape, result.shape[-2:]])
            )
        else:
            result = result.reshape(batch_shape + result.shape[-2:])
    if remove_row:
        result = squeeze(result, axis=-2)
    if remove_col:
        result = squeeze(result, axis=-1)
    return result


def dot(inp1: Tensor, inp2: Tensor) -> Tensor:
    """
    Computes dot-product of two vectors ``inp1`` and ``inp2``.
    inputs must be 1-dimensional, scalar input can be automatically broadcasted.

    :param inp1: first vector.
    :param inp2: second vector.
    :return: output value.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        data1 = tensor(np.arange(0, 6, dtype=np.float32))
        data2 = tensor(np.arange(0, 6, dtype=np.float32))
        out = F.dot(data1, data2)
        print(out.numpy())

    Outputs:

    .. testoutput::

        [55.]

    """
    op = builtin.Dot()
    inp1, inp2 = utils.convert_inputs(inp1, inp2)
    (result,) = apply(op, inp1, inp2)
    return result


def svd(inp: Tensor, full_matrices=False, compute_uv=True) -> Tensor:
    """
    Computes the singular value decompositions of input matrix.

    :param inp: input matrix, must has shape `[..., M, N]`.
    :return: output matrices, `(U, sigma, V)`.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = tensor(np.arange(0, 6, dtype=np.float32).reshape(2,3))
        _, y, _ = F.svd(x)
        print(y.numpy().round(decimals=3))

    Outputs:

    .. testoutput::

        [7.348 1.   ]

    """
    op = builtin.SVD(full_matrices=full_matrices, compute_uv=compute_uv)
    U, sigma, V = apply(op, inp)
    return U, sigma, V


def interpolate(
    inp: Tensor,
    size: Optional[Union[int, Tuple[int, int]]] = None,
    scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
    mode: str = "BILINEAR",
    align_corners: bool = None,
) -> Tensor:
    r"""
    Down/up samples the input tensor to either the given size or with the given scale_factor. ``size`` can not coexist with ``scale_factor``.

    :param inp: input tensor.
    :param size: size of the output tensor. Default: None
    :param scale_factor: scaling factor of the output tensor. Default: None
    :param mode: interpolation methods, acceptable values are:
        "BILINEAR", "LINEAR". Default: "BILINEAR"
    :return: output tensor.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = tensor(np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2))
        out = F.nn.interpolate(x, [4, 4], align_corners=False)
        print(out.numpy())
        out2 = F.nn.interpolate(x, scale_factor=2.)
        np.testing.assert_allclose(out.numpy(), out2.numpy())

    Outputs:

    .. testoutput::

        [[[[1.   1.25 1.75 2.  ]
           [1.5  1.75 2.25 2.5 ]
           [2.5  2.75 3.25 3.5 ]
           [3.   3.25 3.75 4.  ]]]]

    """
    mode = mode.upper()
    if mode not in ["BILINEAR", "LINEAR"]:
        raise ValueError("interpolate only support linear or bilinear mode")
    if mode not in ["BILINEAR", "LINEAR"]:
        if align_corners is not None:
            raise ValueError(
                "align_corners option can only be set in the bilinear/linear interpolating mode"
            )
    else:
        if align_corners is None:
            align_corners = False

    if mode == "LINEAR":
        inp = expand_dims(inp, 3)

    if inp.ndim != 4:
        raise ValueError("shape of input tensor must correspond to the operartion mode")

    if size is None:
        if scale_factor is None:
            raise ValueError("scale_factor must not be None when size is None")

        if isinstance(scale_factor, (float, int)):
            scale_factor = float(scale_factor)
            if mode == "LINEAR":
                scale_factor = (scale_factor, float(1))
            else:
                scale_factor = (scale_factor, scale_factor)
        else:
            if mode == "LINEAR":
                raise ValueError(
                    "under LINEAR mode, scale_factor can only be single value"
                )

        assert len(scale_factor) == 2, "shape of scale_factor must be equal to (2, )"
        assert isinstance(scale_factor[0], float) and isinstance(
            scale_factor[1], float
        ), "scale_factor must be float type"
        dsize = tuple(
            floor(
                Tensor(
                    inp.shape[i + 2] * scale_factor[i],
                    dtype="float32",
                    device=inp.device,
                )
            )
            for i in range(2)
        )
        dsize = concat([dsize[0], dsize[1]], axis=0)
    else:
        if scale_factor is not None:
            raise ValueError("scale_factor must be None when size is provided")

        if isinstance(size, int):
            size = (size, 1)
        else:
            if mode == "LINEAR":
                raise ValueError("under LINEAR mode, size can only be single value")
        dsize = size

    oh, ow = dsize[0], dsize[1]
    ih, iw = inp.shape[2], inp.shape[3]

    if align_corners:
        hscale = (ih - 1.0) / (oh - 1.0)
        wscale = 1.0 * iw / ow
        if mode != "LINEAR":
            wscale = (iw - 1.0) / (ow - 1.0)
        row0 = concat(
            [wscale, Tensor([0, 0], dtype="float32", device=inp.device)], axis=0
        ).reshape(1, 3)
        row1 = concat(
            [
                Tensor(0, dtype="float32", device=inp.device),
                hscale,
                Tensor(0, dtype="float32", device=inp.device),
            ],
            axis=0,
        ).reshape(1, 3)
        weight = concat(
            [row0, row1, Tensor([[0, 0, 1]], dtype="float32", device=inp.device)],
            axis=0,
        ).reshape(1, 3, 3)
        weight = broadcast_to(weight, (inp.shape[0], 3, 3))
    else:
        hscale = 1.0 * ih / oh
        wscale = 1.0 * iw / ow
        row0 = concat(
            [wscale, Tensor(0, dtype="float32", device=inp.device), 0.5 * wscale - 0.5],
            axis=0,
        ).reshape(1, 3)
        row1 = concat(
            [Tensor(0, dtype="float32", device=inp.device), hscale, 0.5 * hscale - 0.5],
            axis=0,
        ).reshape(1, 3)
        weight = concat(
            [row0, row1, Tensor([[0, 0, 1]], dtype="float32", device=inp.device)],
            axis=0,
        ).reshape(1, 3, 3)
        weight = broadcast_to(weight, (inp.shape[0], 3, 3))

    weight = weight.astype("float32")
    ret = warp_perspective(inp, weight, dsize, interp_mode="LINEAR")
    if mode == "LINEAR":
        ret = reshape(ret, ret.shape[0:3])
    return ret


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


def roi_pooling(
    inp: Tensor,
    rois: Tensor,
    output_shape: Union[int, tuple, list],
    mode: str = "max",
    scale: float = 1.0,
) -> Tensor:
    """
    Applies roi pooling on input feature.

    :param inp: tensor that represents the input feature, `(N, C, H, W)` images.
    :param rois: `(K, 5)` boxes. First column is the index into N. The other 4 columns are xyxy.
    :param output_shape: `(height, width)` of output rois feature.
    :param mode: "max" or "average", use max/average align just like max/average pooling. Default: "max"
    :param scale: scale the input boxes by this number. Default: 1.0
    :return: `(K, C, output_shape[0], output_shape[1])` feature of rois.

    Examples:

    .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            np.random.seed(42)
            inp = tensor(np.random.randn(1, 1, 128, 128))
            rois = tensor(np.random.random((4, 5)))
            y = F.nn.roi_pooling(inp, rois, (2, 2))
            print(y.numpy()[0].round(decimals=4))

    Outputs:

    .. testoutput::

            [[[-0.1383 -0.1383]
              [-0.5035 -0.5035]]]


    """
    assert mode in ["max", "average"], "only max/average mode is supported"
    if isinstance(output_shape, int):
        output_shape = (output_shape, output_shape)

    op = builtin.ROIPooling(mode=mode, scale=scale)
    inp, rois = utils.convert_inputs(inp, rois)
    result, _ = apply(
        op, inp, rois, Tensor(output_shape, dtype="int32", device=inp.device)
    )
    return result


def roi_align(
    inp: Tensor,
    rois: Tensor,
    output_shape: Union[int, tuple, list],
    mode: str = "average",
    spatial_scale: float = 1.0,
    sample_points: Union[int, tuple, list] = 2,
    aligned: bool = True,
) -> Tensor:
    """
    Applies roi align on input feature.

    :param inp: tensor that represents the input feature, shape is `(N, C, H, W)`.
    :param rois: `(N, 5)` boxes. First column is the box index. The other 4 columns are ``xyxy``.
    :param output_shape: `(height, width)` shape of output rois feature.
    :param mode: "max" or "average", use max/average align just like max/average pooling. Default: "average"
    :param spatial_scale: scale the input boxes by this number. Default: 1.0
    :param sample_points: number of inputs samples to take for each output sample.
        0 to take samples densely. Default: 2
    :param aligned: wheather to align the input feature, with `aligned=True`,
        we first appropriately scale the ROI and then shift it by -0.5. Default: True
    :return: output tensor.

    Examples:

    .. testcode::

            import numpy as np
            from megengine import tensor
            import megengine.functional as F

            np.random.seed(42)
            inp = tensor(np.random.randn(1, 1, 128, 128))
            rois = tensor(np.random.random((4, 5)))
            y = F.nn.roi_align(inp, rois, (2, 2))
            print(y.numpy()[0].round(decimals=4))

    Outputs:

    .. testoutput::

            [[[0.175  0.175 ]
              [0.1359 0.1359]]]

    """
    assert mode in ["max", "average"], "only max/average mode is supported"
    if isinstance(output_shape, int):
        output_shape = (output_shape, output_shape)
    pooled_height, pooled_width = output_shape
    if isinstance(sample_points, int):
        sample_points = (sample_points, sample_points)
    sample_height, sample_width = sample_points
    offset = 0.5 if aligned else 0.0

    op = builtin.ROIAlign(
        mode=mode,
        format="NCHW",
        spatial_scale=spatial_scale,
        offset=offset,
        pooled_height=pooled_height,
        pooled_width=pooled_width,
        sample_height=sample_height,
        sample_width=sample_width,
    )
    inp, rois = utils.convert_inputs(inp, rois)
    result, *_ = apply(op, inp, rois)
    return result


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


def conv1d(
    inp: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    conv_mode="CROSS_CORRELATION",
    compute_mode="DEFAULT",
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
    :param conv_mode: Supports 'CROSS_CORRELATION'. Default:
        'CROSS_CORRELATION'.
    :type compute_mode: string or
        :class:`mgb.opr_param_defs.Convolution.ComputeMode`
    :param compute_mode: When set to 'DEFAULT', no special requirements will be
        placed on the precision of intermediate results. When set to 'FLOAT32',
        Float32 would be used for accumulator and intermediate result, but only
        effective when input and output are of Float16 dtype.

    """

    assert conv_mode == "CROSS_CORRELATION" or conv_mode.name == "CROSS_CORRELATION"
    assert compute_mode == "DEFAULT" or compute_mode.name == "DEFAULT"
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

    sparse_type = "DENSE" if groups == 1 else "GROUP"
    op = builtin.Convolution(
        stride_h=stride_h,
        stride_w=1,
        pad_h=pad_h,
        pad_w=0,
        dilate_h=dilate_h,
        dilate_w=1,
        strategy=get_conv_execution_strategy(),
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


def nms(
    boxes: Tensor, scores: Tensor, iou_thresh: float, max_output: Optional[int] = None
) -> Tensor:
    r"""
    Performs non-maximum suppression (NMS) on the boxes according to their intersection-over-union(IoU).

    :param boxes: tensor of shape `(N, 4)`; the boxes to perform nms on; each box is expected to be in `(x1, y1, x2, y2)` format.
    :param iou_thresh: IoU threshold for overlapping.
    :param scores: tensor of shape `(N,)`, the score of boxes.
    :param max_output: the maximum number of boxes to keep; it is optional if this operator is not traced
        otherwise it required to be specified; if it is not specified, all boxes are kept.
    :return: indices of the elements that have been kept by NMS.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = np.zeros((100,4))
        np.random.seed(42)
        x[:,:2] = np.random.rand(100,2)*20
        x[:,2:] = np.random.rand(100,2)*20 + 100
        scores = tensor(np.random.rand(100))
        inp = tensor(x)
        result = F.nn.nms(inp, scores, iou_thresh=0.7)
        print(result.numpy())

    Outputs:

    .. testoutput::

        [75 69]

    """
    assert (
        boxes.ndim == 2 and boxes.shape[1] == 4
    ), "the expected shape of boxes is (N, 4)"
    assert scores.ndim == 1, "the expected shape of scores is (N,)"
    assert (
        boxes.shape[0] == scores.shape[0]
    ), "number of boxes and scores are not matched"

    boxes = boxes.detach()
    scores = scores.detach()
    sorted_idx = argsort(scores, descending=True)
    boxes = boxes[sorted_idx]

    if is_tracing():
        assert (
            max_output is not None and max_output > 0
        ), "max_output should be specified under tracing"

    if max_output is None:
        max_output = boxes.shape[0]

    op = builtin.NMSKeep(iou_thresh, max_output)
    inp = utils.convert_inputs(boxes.reshape(1, -1, 4))
    indices, count = apply(op, *inp)
    indices = indices[0][: count[0]]
    keep_inds = sorted_idx[indices]
    return keep_inds


def nvof(src: Tensor, precision: int = 1) -> Tensor:
    r"""
    Implements NVIDIA Optical Flow SDK.

    :src shape: input tensor with shape (n, t, h, w, c4).
    :src dtype: uint8.
    :param precision: 0:NV_OF_PERF_LEVEL_SLOW 1:NV_OF_PERF_LEVEL_MEDIUM 2:NV_OF_PERF_LEVEL_FAST.
    :output shape: (n, t-1, h//4, w//4, c2).
    :output dtype: int16.

    .. code-block:: python

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = np.random.random_integers(0, 255, (1,2,224,244,4)).astype("uint8")
        src = tensor(x)
        result = F.nn.nvof(src, precision=1)
        print(result.numpy())

    """
    assert isinstance(src, (Tensor, megbrain_graph.VarNode)), "src must be Tensor type"
    assert src.ndim == 5 and src.shape[4] == 4

    src = src.detach()

    op = builtin.NvOf(precision=precision)
    return apply(op, src)[0]


from .loss import *  # isort:skip
from .quantized import conv_bias_activation  # isort:skip
