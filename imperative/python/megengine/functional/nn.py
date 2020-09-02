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
from ..core.ops import builtin
from ..core.ops._internal import param_defs as P
from ..core.ops.special import Const
from ..core.tensor import utils
from ..core.tensor.core import TensorBase, TensorWrapperBase, apply
from ..distributed import WORLD, is_distributed
from ..random import uniform
from ..tensor import Tensor
from .debug_param import get_conv_execution_strategy
from .distributed import all_reduce_sum
from .elemwise import exp, floor, log, log1p, maximum, minimum, relu
from .math import argsort, max, sum
from .tensor import add_axis, broadcast, concat, remove_axis, reshape
from .types import _pair, _pair_nonzero

__all__ = [
    "linear",
    "conv2d",
    "conv_transpose2d",
    "local_conv2d",
    "max_pool2d",
    "avg_pool2d",
    "prelu",
    "leaky_relu",
    "softplus",
    "log_softmax",
    "logsigmoid",
    "logsumexp",
    "flatten",
    "softmax",
    "batch_norm2d",
    "sync_batch_norm",
    "one_hot",
    "warp_perspective",
    "matmul",
    "interpolate",
    "dropout",
    "identity",
    "embedding",
    "roi_pooling",
    "roi_align",
    "assert_equal",
    "indexing_one_hot",
    "dot",
    "svd",
    "nms",
    "batched_nms",
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
    """Applies a linear transformation to the input.

    Refer to :class:`~.module.linear.Linear` for more information.

    :param inp: the input tensor with shape `(N, in_features)`.
    :param weight: the weight with shape `(out_features, in_features)`.
    :param bias: the bias with shape `(out_features,)`.
        Default: ``None``
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
    """2D convolution operation.

    Refer to :class:`~.Conv2d` for more information.

    :param inp: The feature map of the convolution operation
    :param weight: The convolution kernel
    :param bias: The bias added to the result of convolution (if given)
    :param stride: Stride of the 2D convolution operation. Default: 1
    :param padding: Size of the paddings added to the input on both sides of its
        spatial dimensions. Only zero-padding is supported. Default: 0
    :param dilation: Dilation of the 2D convolution operation. Default: 1
    :param groups: number of groups to divide input and output channels into,
        so as to perform a "grouped convolution". When ``groups`` is not 1,
        ``in_channels`` and ``out_channels`` must be divisible by ``groups``,
        and the shape of weight should be ``(groups, out_channel // groups,
        in_channels // groups, height, width)``.
    :type conv_mode: string or :class:`P.Convolution.Mode`
    :param conv_mode: Supports 'CROSS_CORRELATION' or 'CONVOLUTION'. Default:
        'CROSS_CORRELATION'.
    :type compute_mode: string or
        :class:`P.Convolution.ComputeMode`
    :param compute_mode: When set to 'DEFAULT', no special requirements will be
        placed on the precision of intermediate results. When set to 'FLOAT32',
        Float32 would be used for accumulator and intermediate result, but only
        effective when input and output are of Float16 dtype.

    """
    assert conv_mode == "CROSS_CORRELATION" or conv_mode.name == "CROSS_CORRELATION"
    assert compute_mode == "DEFAULT" or compute_mode.name == "DEFAULT"

    stride_h, stride_w = expand_hw(stride)
    pad_h, pad_w = expand_hw(padding)
    dilate_h, dilate_w = expand_hw(dilation)

    Sparse = P.Convolution.Sparse
    sparse_type = Sparse.DENSE if groups == 1 else Sparse.GROUP
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
    """2D transposed convolution operation.

    Refer to :class:`~.ConvTranspose2d` for more information.

    :param inp: The feature map of the convolution operation
    :param weight: The convolution kernel
    :param bias: The bias added to the result of convolution (if given)
    :param stride: Stride of the 2D convolution operation. Default: 1
    :param padding: Size of the paddings added to the input on both sides of its
        spatial dimensions. Only zero-padding is supported. Default: 0
    :param dilation: Dilation of the 2D convolution operation. Default: 1
    :param groups: number of groups to divide input and output channels into,
        so as to perform a "grouped convolution". When ``groups`` is not 1,
        ``in_channels`` and ``out_channels`` must be divisible by ``groups``,
        and the shape of weight should be ``(groups, out_channel // groups,
        in_channels // groups, height, width)``. Default: 1
    :type conv_mode: string or :class:`P.Convolution.Mode`
    :param conv_mode: Supports 'CROSS_CORRELATION' or 'CONVOLUTION'. Default:
        'CROSS_CORRELATION'.
    :type compute_mode: string or
        :class:`P.Convolution.ComputeMode`
    :param compute_mode: When set to 'DEFAULT', no special requirements will be
        placed on the precision of intermediate results. When set to 'FLOAT32',
        Float32 would be used for accumulator and intermediate result, but only
        effective when input and output are of Float16 dtype.

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
) -> Tensor:
    """Applies spatial 2D convolution over an image with untied kernels.

    Refer to :class:`~.LocalConv2d` for more information.
    """
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
        # strategy=get_conv_execution_strategy(),
    )
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
    """Applies a 2D max pooling over an input.

    Refer to :class:`~.MaxPool2d` for more information.

    :param inp: The input tensor.
    :param kernel_size: The size of the window.
    :param stride: The stride of the window. If not provided, its value is set to ``kernel_size``.
        Default: None
    :param padding: Implicit zero padding to be added on both sides. Default: 0

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
    """ Applies a 2D average pooling over an input.

    Refer to :class:`~.AvgPool2d` for more information.

    :param inp: The input tensor.
    :param kernel_size: The size of the window.
    :param stride: The stride of the window. If not provided, its value is set to ``kernel_size``.
        Default: None
    :param padding: Implicit zero padding to be added on both sides. Default: 0
    :param mode: Whether to count padding values. Default: "AVERAGE_COUNT_EXCLUDE_PADDING"

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
    r"""Applies the element-wise function:

    .. math::
        \text{softplus}(x) = \log(1 + \exp(x))
    
    softplus is a smooth approximation to the ReLU function and can be used
    to constrain the output of a machine to always be positive.
    For numerical stability the implementation follows this transformation:

    .. math::
        \text{softplus}(x) = \log(1 + \exp(x)) 
                           = \log(1 + \exp(-\text{abs}(x))) + \max(x, 0) 
                           = \log1p(\exp(-\text{abs}(x))) + \text{relu}(x)

    :param inp: The input tensor

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = tensor(np.arange(-3, 3, dtype=np.float32))
        y = F.softplus(x)
        print(y.numpy())

    .. output::

        [0.04858735 0.126928   0.3132617  0.6931472  1.3132617  2.126928  ]

    """
    return log1p(exp(-abs(inp))) + relu(inp)


def log_softmax(inp: Tensor, axis: Union[int, Sequence[int]]) -> Tensor:
    r"""Applies the :math:`\log(\text{Softmax}(x))` function to an n-dimensional
    input Tensor. The LogSoftmax formulation can be simplified as:

    .. math::
        \text{LogSoftmax}(x_{i}) = \log(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} )

    For numerical stability the implementation follows this transformation:

    .. math::
        \operatorname{logsoftmax}(x) 
        = \log (\frac{\exp (x)}{\sum_{i}(\exp (x_{i}))})
        = x - \log (\sum_{i}(\exp (x_{i})))
        = x - logsumexp(x)
    
    :param inp: The input tensor
    :param axis: An axis along which log_softmax will be applied.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = tensor(np.arange(-5, 5, dtype=np.float32)).reshape(2,5)
        y = F.log_softmax(x, axis=1)
        print(y.numpy())

    .. output::

        [[-4.4519143 -3.4519143 -2.4519143 -1.4519144 -0.4519144]
         [-4.4519143 -3.4519143 -2.4519143 -1.4519144 -0.4519144]]

    """
    return inp - logsumexp(inp, axis, keepdims=True)


def logsigmoid(inp: Tensor) -> Tensor:
    r"""Applies the element-wise function:

    .. math::
        \text{logsigmoid}(x) = \log(\frac{ 1 }{ 1 + \exp(-x)})
        = \log(1/(1 + exp(-x)))
        = - \log(1 + exp(-x))
        = - \text{softplus}(-x)

    :param inp: The input tensor

    Examples:
    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = tensor(np.arange(-5, 5, dtype=np.float32))
        y = F.logsigmoid(x)
        print(y.numpy())

    .. output::

        [-5.0067153  -4.01815    -3.0485873  -2.126928   -1.3132617  -0.6931472  -0.3132617  -0.126928   -0.04858735 -0.01814993]

    """
    return -softplus(-inp)


def logsumexp(
    inp: Tensor, axis: Union[int, Sequence[int]], keepdims: bool = False
) -> Tensor:
    r"""
    Compute the log of the sum of exponentials of inputs along the given :attr:`axis`. 
    The computation is numerically stabilized.
    
    .. math::
        
        \operatorname{logsumexp}(\boldsymbol{x})= \log \sum_{j=1}^{n} \exp \left(x_{j}\right)

    For numerical stability, the implementation follows this transformation:

    .. math::

        \operatorname{logsumexp}(\boldsymbol{x})= \log \sum_{j=1}^{n} \exp \left(x_{j}\right)
        = \operatorname{logsumexp}(\boldsymbol{x})=b+\log \sum_{j=1}^{n} \exp \left(x_{j}-b\right)
    
    where

    .. math::
        b = \max(x_j)

    :param inp: The input tensor.
    :param axis: Axis over which the sum is taken. It can be a single axis or a list of axes.
    :param keepdims: whether to retain :attr:`axis` or not for the output tensor.

    Examples:
    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = tensor(np.arange(-5, 5, dtype=np.float32)).reshape(2,5)
        y = F.logsumexp(x, axis=1, keepdims=False)
        print(y.numpy())

    .. output::

        [-0.5480856  4.4519143]

    """
    max_value = max(inp, axis, keepdims=True)
    if keepdims:
        return max_value + log(sum(exp(inp - max_value), axis, keepdims))
    else:
        return remove_axis(max_value, axis=None) + log(
            sum(exp(inp - max_value), axis, keepdims)
        )


def flatten(inp: Tensor, start_axis: int = 0, end_axis: int = -1) -> Tensor:
    r"""
    Reshapes the tensor by flattening the sub-tensor from dimension ``start_axis`` to dimension ``end_axis``.

    :param inp: The input tensor.
    :param start_axis: The start dimension that the sub-tensor to be flattened. Default: 0
    :param end_axis: The end dimension that the sub-tensor to be flattened. Default: -1

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        inp_shape = (2, 2, 3, 3)
        inp = tensor(
            np.arange(36, dtype=np.int32).reshape(inp_shape),
        )
        oup = F.flatten(inp, 2)
        print(inp.numpy().shape)
        print(oup.numpy().shape)

    Outputs:

    .. testoutput::

        (2, 2, 3, 3)
        (2, 2, 9)

    """
    target_shape = tuple(inp.shape[i] for i in range(start_axis)) + (-1,)
    if end_axis != -1:
        target_shape += (*inp.shape[end_axis + 1 :],)
    return inp.reshape(*target_shape)


def _get_softmax_axis(ndim: int) -> int:
    if ndim in (0, 1, 3):
        return 0
    return 1


def softmax(inp: Tensor, axis: Optional[int] = None) -> Tensor:
    r"""
    Applies a softmax function. Softmax is defined as:

    .. math::
            \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    It is applied to all elements along axis, and will re-scale them so that
    the elements lie in the range `[0, 1]` and sum to 1.

    See :class:`~megengine.module.activation.Softmax` for more details.

    :param inp: The input tensor.
    :param axis: An axis along which softmax will be applied. By default,
        softmax will apply along the highest ranked axis.

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = tensor(np.arange(-5, 5, dtype=np.float32)).reshape(2,5)
        out = F.softmax(x)
        print(out.numpy())

    Outputs:

    .. testoutput::
        [[0.01165623 0.03168492 0.08612854 0.23412167 0.6364086 ]
         [0.01165623 0.03168492 0.08612854 0.23412167 0.6364086 ]]

    """
    if axis is None:
        axis = _get_softmax_axis(len(inp.shape))
    offset = inp.max(axis=axis, keepdims=True).detach()
    cached = exp(inp - offset)
    down = sum(cached, axis=axis, keepdims=True)
    return cached / down


def batch_norm2d(
    data: Tensor,
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
    """Applies batch normalization to the input.

    Refer to :class:`~.BatchNorm2d` and :class:`~.BatchNorm1d` for more information.

    :param inp: input tensor.
    :param running_mean: tensor to store running mean.
    :param running_var: tensor to store running variance.
    :param weight: scaling tensor in the learnable affine parameters.
        See :math:`\gamma` in :class:`~.BatchNorm2d`
    :param bias: bias tensor in the learnable affine parameters.
        See :math:`\beta` in :class:`~.BatchNorm2d`
    :param training: a boolean value to indicate whether batch norm is performed
        in traning mode. Default: ``False``
    :param momentum: the value used for the ``running_mean`` and ``running_var``
        computation.
        Default: 0.9
    :param eps: a value added to the denominator for numerical stability.
        Default: 1e-5.
    :param inplace: whether to update running_mean and running_var inplace or return new tensors 
        Default: True

    """
    from .tensor import expand_dims, squeeze, broadcast

    def full(value):
        C = data.shape[1]
        (x,) = Const(value, dtype=data.dtype, device=data.device)(data)
        return broadcast(x, [1, C, 1, 1])

    def expand_or_full(x, value):
        if x is None:
            return full(value)
        return expand_dims(x, [0, 2, 3])

    def make_full_if_none(x, value):
        if x is None:
            return full(value)
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

    data, weight, bias, running_mean, running_var = utils.convert_inputs(
        data, weight, bias, running_mean, running_var
    )

    weight = expand_or_full(weight, 1)
    bias = expand_or_full(bias, 0)

    if not training:
        op = builtin.BatchNorm(fwd_mode="INFERENCE", epsilon=eps, param_dim="DIM_1C11")
        ret = apply(op, data, weight, bias, running_mean, running_var)[-1]
        return ret

    else:
        op = builtin.BatchNorm(
            avg_factor=1 - momentum, epsilon=eps, param_dim="DIM_1C11"
        )

        if has_mean or has_var:
            running_mean = make_full_if_none(running_mean, 0)
            running_var = make_full_if_none(running_var, 1)
            new_mean, new_var, _, _, data = apply(
                op, data, weight, bias, running_mean, running_var
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

                return data
            else:
                return data, new_mean, new_var
        else:
            _, _, data, = apply(op, data, weight, bias)
            return data


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
    """ Applies synchronized batch normalization to the input.

    Refer to :class:`~.BatchNorm2d` and :class:`~.BatchNorm1d` for more information.

    :param inp: input tensor.
    :param running_mean: tensor to store running mean.
    :param running_var: tensor to store running variance.
    :param weight: scaling tensor in the learnable affine parameters.
        See :math:`\gamma` in :class:`~.BatchNorm2d`
    :param bias: bias tensor in the learnable affine parameters.
        See :math:`\beta` in :class:`~.BatchNorm2d`
    :param training: a boolean value to indicate whether batch norm is performed
        in traning mode. Default: ``False``
    :param momentum: the value used for the ``running_mean`` and ``running_var``
        computation.
        Default: 0.9
    :param eps: a value added to the denominator for numerical stability.
        Default: 1e-5.
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
            reduce_size = broadcast(Tensor(reduce_size, dtype=_dtype), [1] * _ndim)
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
    # where output = input * invsqrt_channel_variance + (
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
    Perform one-hot encoding for the input tensor.

    :param inp: input tensor
    :param num_classes: number of classes denotes the last dimension of the output tensor

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        inp = tensor(np.arange(1, 4, dtype=np.int32))
        out = F.one_hot(inp, num_classes=4)
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[0 1 0 0]
         [0 0 1 0]
         [0 0 0 1]]

    """
    raise NotImplementedError
    # comp_node, comp_graph = _decide_comp_node_and_comp_graph(inp)

    # zeros = mgb.make_immutable(value=0, comp_node=comp_node, comp_graph=comp_graph)
    # zeros_symvar = zeros.broadcast(inp.shapeof(), num_classes)

    # ones = mgb.make_immutable(value=1, comp_node=comp_node, comp_graph=comp_graph)
    # ones_symvar = ones.broadcast(inp.shapeof(), 1)

    # return Tensor(
    #     mgb.opr.indexing_set_one_hot(
    #         zeros_symvar, axis=len(inp.shapeof()), index=inp, value=ones_symvar
    #     )
    # )


def warp_perspective(
    inp: Tensor,
    M: Tensor,
    dsize: Union[Tuple[int, int], int, Tensor],
    border_mode: str = "REPLICATE",
    border_val: float = 0.0,
    interp_mode: str = "LINEAR",
):
    r"""
    Applies perspective transformation to batched 2D images.

    The input images are transformed to the output images by the transformation matrix:

    .. math::
            \text{output}(n, c, h, w) = \text{input} \left( n, c,
                \frac{M_{00}h + M_{01}w + M_{02}}{M_{20}h + M_{21}w + M_{22}},
                \frac{M_{10}h + M_{11}w + M_{12}}{M_{20}h + M_{21}w + M_{22}}
                \right)

    :param inp: input image
    :param M: (batch, 3, 3) transformation matrix
    :param dsize: (h, w) size of the output image
    :param border_mode: pixel extrapolation method. Default: ``"REPLICATE"``
    :param border_val: value used in case of a constant border. Default: ``0``
    :param interp_mode: interpolation methods. Default: ``"LINEAR"``

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F
        inp_shape = (1, 1, 4, 4)
        inp = tensor(np.arange(16, dtype=np.float32).reshape(inp_shape))
        M_shape = (1, 3, 3)
        # M defines a translation: dst(1, 1, h, w) = rst(1, 1, h+1, w+1)
        M = tensor(np.array([[1., 0., 1.],
                             [0., 1., 1.],
                             [0., 0., 1.]], dtype=np.float32).reshape(M_shape))
        out = F.warp_perspective(inp, M, (2, 2))
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[[[ 5.  6.]
           [ 9. 10.]]]]

    """
    op = builtin.WarpPerspective(
        imode=interp_mode, bmode=border_mode, format="NCHW", border_val=border_val
    )
    (result,) = apply(op, inp, M, Tensor(dsize))
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

    - Both 1-D tensor, simply forward to dot.
    - Both 2-D tensor, normal matrix multiplication.
    - If one input tensor is 1-D, matrix vector multiplication.
    - If at least one tensor are 3-dimensional or >3-dimensional, the batched matrix-matrix is returned, and the tensor with smaller dimension will
      be broadcasted. For example:
        - inp1: `(k, m)`, inp2: `(m, p)`, return: `(k, p)`
        - inp1: `(n, k, m)`, inp2: `(n, m, p)`, return: `(n, k, p)`
        - inp1: `(n, k, m)`, inp2: `(m, p)`, return: `(n, k, p)`
        - inp1: `(n, j, k, m)`, inp2: `(n, j, m, p)`, return: `(n, j, k, p)`

    :param inp1: The first matrix to be multiplied
    :param inp2: The second matrix to be multiplied
    :return: The output tensor

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
    inp1, inp2 = utils.convert_inputs(inp1, inp2)
    dim1, dim2 = inp1.ndim, inp2.ndim
    if dim1 == 1 and dim2 == 1:
        return dot(inp1, inp2)

    shp = None
    if dim1 > 3 or dim2 > 3:
        shape1, shape2 = list(inp1.shape), list(inp2.shape)
        if dim1 != dim2:
            if dim1 < dim2:
                shape1 = shape2[: dim2 - dim1] + shape1
                inp1 = inp1.broadcast(*shape1)
            else:
                shape2 = shape1[: dim1 - dim2] + shape2
                inp2 = inp2.broadcast(*shape2)
        reshaped_batch_size = 1
        for i in shape1[:-2]:
            reshaped_batch_size *= i
        inp1 = inp1.reshape(*([reshaped_batch_size] + shape1[-2:]))
        inp2 = inp2.reshape(*([reshaped_batch_size] + shape2[-2:]))
        op = builtin.BatchedMatrixMul(
            transposeA=transpose_a,
            transposeB=transpose_b,
            compute_mode=compute_mode,
            format=format,
        )
        shp = shape1[:-1] + shape2[-1:]
    elif dim1 == 3 or dim2 == 3:
        if dim2 < 3:
            inp2 = inp2.broadcast(*(inp1.shape[:1] + inp2.shape))
        elif dim1 < 3:
            inp1 = inp1.broadcast(*(inp2.shape[:1] + inp1.shape))
        op = builtin.BatchedMatrixMul(
            transposeA=transpose_a,
            transposeB=transpose_b,
            compute_mode=compute_mode,
            format=format,
        )
    else:
        if dim1 == 1:
            shp = (inp2.shape[1],)
            inp1 = add_axis(inp1, 0)
        if dim2 == 1:
            shp = (inp1.shape[0],)
            inp2 = add_axis(inp2, 1)
        op = builtin.MatrixMul(
            transposeA=transpose_a,
            transposeB=transpose_b,
            compute_mode=compute_mode,
            format=format,
        )

    (result,) = apply(op, inp1, inp2)
    if shp is not None:
        result = result.reshape(shp)
    return result


def dot(inp1: Tensor, inp2: Tensor) -> Tensor:
    """
    Compute dot-product of two vectors ``inp1`` and ``inp2``.
    inputs must be 1-dimensional, scalar input can be automatically broadcasted.

    :param inp1: The first vector
    :param inp2: The second vector
    :return: The output value

    Examples:

    .. teestcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        data1 = tensor(np.arange(0, 6, dtype=np.float32))
        data2 = tensor(np.arange(0, 6, dtype=np.float32))
        out = F.dot(data1, data2)
        print(out.numpy())

    Outputs:

        [55.]

    .. testoutputs::
    """
    op = builtin.Dot()
    inp1, inp2 = utils.convert_inputs(inp1, inp2)
    (result,) = apply(op, inp1, inp2)
    return result


def svd(inp: Tensor, full_matrices=False, compute_uv=True) -> Tensor:
    """
    Compute the singular value decompositions of input matrix ``inp``.

    :param inp: The input matrix, must has shape ``[..., M, N]``
    :return: The output matrices, U, sigma, V

    Examples:

    .. teestcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        x = tensor(np.arange(0, 6, dtype=np.float32).reshape(2,3))
        _, y, _ = F.svd(x)
        print(y.numpy())

    Outputs:

        [7.348, 1.]

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
    Down/up samples the input tensor to either the given :attr:`size` or the given
    :attr:`scale_factor`

    :param inp: input tensor
    :param size: size of the output tensor. Default: ``None``
    :param scale_factor: scaling factor of the output tensor. Default: ``None``
    :param mode: interpolation methods, acceptable values are:
        'BILINEAR', 'LINEAR'. Default: ``BILINEAR``

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F
        from megengine.test import assertTensorClose

        inp = tensor(np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2))
        out = F.interpolate(inp, [4, 4], align_corners=False)
        print(out.numpy())

        out2 = F.interpolate(inp, scale_factor=2.)
        assertTensorClose(out.numpy(), out2.numpy())

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
        inp = add_axis(inp, 3)

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
        weight = broadcast(weight, (inp.shape[0], 3, 3))
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
        weight = broadcast(weight, (inp.shape[0], 3, 3))

    weight = weight.astype("float32")
    ret = warp_perspective(inp, weight, dsize, interp_mode="LINEAR")
    if mode == "LINEAR":
        ret = reshape(ret, ret.shape[0:3])
    return ret


def dropout(inp: Tensor, drop_prob: float, rescale: bool = True) -> Tensor:
    """
    Returns a new tensor where each of the elements are randomly set to zero
    with probability P = ``drop_prob``. Optionally rescale the output tensor.

    :param inp: The input tensor
    :param drop_prob: The probability to drop (set to zero) a single element
    :param rescale: The default behavior of ``dropout`` during training is to rescale the output,
        then it can be replaced by an :class:`~.Identity` during inference, default to True.
    :return: The output tensor

    Examples:

    .. testcode::

        import numpy as np
        import megengine as mge

        import megengine.functional as F
        from megengine import tensor

        data = tensor(np.ones(10, dtype=np.float32))
        out = F.dropout(data, 1./3.)
        print(out.numpy())

    Outputs:

    .. testoutput::
        :options: +SKIP

        [1.5 1.5 0.  1.5 1.5 1.5 1.5 1.5 1.5 1.5]

    """
    assert 0 <= drop_prob < 1
    rv = uniform(inp.shape)
    mask = rv > drop_prob
    inp *= mask.astype(inp.dtype)
    if rescale:
        inp *= 1 / (1 - drop_prob)
    return inp


def identity(inp: Tensor) -> Tensor:
    """applies an identity transform to the input tensor.

    :param inp: The input tensor
    """
    op = builtin.Identity()
    (data,) = utils.convert_inputs(inp)
    (output,) = apply(op, data)
    return output


def embedding(
    input: Tensor,
    weight: Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: Optional[float] = None,
):
    """
    Applies lookup table for embedding.

    :param input: the tensor with indices.
    :param weight: the learnable weights which embedding from.
    :param padding_idx: should be set to None, not support now.
    :param max_norm: should be set to None, not support now.
    :param norm_type: should be set to None, not support now.


    Refer to :class:`~.Embedding` for more information.
    """
    if padding_idx is not None:
        raise ValueError("Not support padding_idx Now!")
    if max_norm is not None or norm_type is not None:
        raise ValueError("Not support weight normlization Now!")

    dest_shp = list(input.shape) + [weight.shape[-1]]
    return weight[input.reshape(-1)].reshape(dest_shp)


def roi_pooling(
    inp: Tensor,
    rois: Tensor,
    output_shape: Union[int, tuple, list],
    mode: str = "max",
    scale: float = 1.0,
) -> Tensor:
    """
    Apply roi pooling on input feature

    :param inp: tensor that represents the input feature, (N, C, H, W) images
    :param rois: (K, 5) boxes. First column is the index into N. The other 4 columns are xyxy
    :param output_shape: (height, width) of output rois feature
    :param mode: "max" or "average", use max/average align just like max/average pooling. Default: ``"max"``
    :param scale: scale the input boxes by this number. Default: 1.0
    :return: (K, C, output_shape[0], output_shape[1]) feature of rois
    """
    assert mode in ["max", "average"], "only max/average mode is supported"
    if isinstance(output_shape, int):
        output_shape = (output_shape, output_shape)

    op = builtin.ROIPooling(mode=mode, scale=scale)
    result, _ = apply(
        op, inp, rois, Tensor(output_shape, dtype="int32", device=inp.device)
    )
    return result


def roi_align(
    input: Tensor,
    rois: Tensor,
    output_shape: Union[int, tuple, list],
    mode: str = "average",
    spatial_scale: float = 1.0,
    sample_points: Union[int, tuple, list] = 2,
    aligned: bool = True,
) -> Tensor:
    """
    Apply roi align on input feature

    :param input: tensor that represents the input feature, (N, C, H, W) images
    :param rois: (N, 5) boxes. First column is the index into N. The other 4 columns are xyxy
    :param output_shape: (height, width) shape of output rois feature.
    :param mode: "max" or "average", use max/average align just like max/average pooling. Default: ``"average"``
    :param spatial_scale: scale the input boxes by this number. Default: 1.0
    :param sample_points: number of inputs samples to take for each output sample.
        0 to take samples densely. Default: 2
    :param aligned: wheather align the input feature, with `aligned=True`,
        we first appropriately scale the ROI and then shift it by -0.5. Default: True
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
    result, *_ = apply(op, input, rois)
    return result


def assert_equal(
    get: Tensor, expect: Tensor, max_err: float = 1e-4, verbose: bool = False
) -> Tensor:
    r"""
    Asserts that ``get`` equals to ``expect``, and returns value of ``expect``.

    :param get: tensor to be checked.
    :param expect: tensor with expected values.
    :param max_err: tolerance that two float values are asserted equal. Default: 1e-4
    :param verbose: whether to print details if two tensors are not equal. Default: False

    Examples:

    .. testcode::

        import megengine.functional as F
        from megengine import tensor

        get = tensor([1.0, 2.0])
        max_err = 0.1
        expect = get + max_err / 2.0
        val = F.assert_equal(expect, get, max_err=max_err)
        print(val.numpy())

    Outputs:

    .. testoutput::

        [1.05 2.05]

    """
    raise NotImplementedError
    # op = builtin.AssertEqual(maxerr=max_err, verbose=verbose)
    # result, = apply(op, get, expect)
    # return result


def indexing_one_hot(
    src: Tensor, index: Tensor, axis: int = 1, keepdims=False
) -> Tensor:
    r"""
    One-hot indexing for some axis.

    :param src: input data tensor.
    :param index: index tensor.
    :param axis: the axis on src for which values in index index. Default: 1
    :param keepdims: whether not to remove the axis in result. Default: ``False``

    Examples:

    .. testcode::

        import megengine.functional as F
        from megengine import tensor

        src = tensor([[1.0, 2.0]])
        index = tensor([0])
        val = F.indexing_one_hot(src, index)
        print(val.numpy())

    .. testoutput::

        [1.]

    """
    assert isinstance(
        src, (TensorWrapperBase, TensorBase)
    ), "src must be of Tensor type"
    op = builtin.IndexingOneHot(axis=axis)
    index = utils.convert_single_value(index, (src,), dtype="int32")
    (result,) = apply(op, src, index)
    if not keepdims:
        result = remove_axis(result, axis)
    return result


def nms(boxes: Tensor, iou_thresh: float, scores: Optional[Tensor] = None) -> Tensor:
    r"""
    Performs non-maximum suppression (NMS) on the boxes according to their intersection-over-union (IoU).

    :param boxes: tensor of shape ``(N, 4)``; the boxes to perform nms on; each box is expected to be in (x1, y1, x2, y2) format.
    :param iou_thresh: iou threshold for overlapping.
    :param scores: tensor of shape ``(N,)``, the score of boxes.
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
        result = F.nms(inp, iou_thresh=0.7, scores=scores)
        print(result.numpy())

    Outputs:

    .. testoutput::
    
        [75 69]

    """
    assert (
        boxes.ndim == 2 and boxes.shape[1] == 4
    ), "the expected shape of boxes is (N, 4)"

    sorted_idx = None
    if not scores is None:
        assert scores.ndim == 1, "the expected shape of scores is (N,)"
        sorted_idx = argsort(scores, descending=True)
        boxes = boxes[sorted_idx]
    max_output = boxes.shape[0]

    op = builtin.NMSKeep(iou_thresh, max_output)
    inp = utils.convert_inputs(boxes.reshape(1, -1, 4))
    indices, count = apply(op, *inp)
    indices = indices[0][: count.item()]
    ret = sorted_idx[indices] if sorted_idx is not None else indices
    return ret


def batched_nms(
    boxes: Tensor, iou_thresh: float, idxs: Tensor, scores: Optional[Tensor] = None
) -> Tensor:
    r"""
    Performs non-maximum suppression (NMS) on the boxes according to their intersection-over-union (IoU).

    :param boxes: tensor of shape ``(N, 4)``; the boxes to perform nms on; each box is expected to be in (x1, y1, x2, y2) format
    :param iou_thresh: iou threshold for overlapping
    :param idxs: tensor of shape ``(N,)``, the class indexs of boxes in the batch.
    :param scores: tensor of shape ``(N,)``, the score of boxes.
    :return: indices and the number of the elements that have been kept by NMS

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
        idxs =  tensor(np.random.randint(0, 10, 100))
        inp = tensor(x)
        result = F.batched_nms(inp, iou_thresh=0.6, idxs=idxs, scores=scores)
        print(result.numpy())

    Outputs:

    .. testoutput::

        [75 41 99 98 69 64 11 27 35 18]

    """
    assert (
        boxes.ndim == 2 and boxes.shape[1] == 4
    ), "the expected shape of boxes is (N, 4)"
    max_coordinate = boxes.max()
    offsets = idxs.astype("float32") * (max_coordinate + 1)
    boxes = boxes + offsets.reshape(-1, 1).broadcast(boxes.shape[0], 4)

    sorted_idx = None
    if not scores is None:
        assert scores.ndim == 1, "the expected shape of scores is (N,)"
        sorted_idx = argsort(scores, descending=True)
        boxes = boxes[sorted_idx]
    max_output = boxes.shape[0]

    op = builtin.NMSKeep(iou_thresh, max_output)
    inp = utils.convert_inputs(boxes.reshape(1, -1, 4))
    indices, count = apply(op, *inp)
    indices = indices[0][: count.item()]
    ret = sorted_idx[indices] if sorted_idx is not None else indices
    return ret
