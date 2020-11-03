# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# pylint: disable=too-many-lines
from typing import Tuple, Union

from ..core.ops import builtin
from ..core.tensor.core import apply
from ..tensor import Tensor
from .debug_param import get_conv_execution_strategy
from .types import _pair, _pair_nonzero


def conv_bias_activation(
    inp: Tensor,
    weight: Tensor,
    bias: Tensor,
    dtype=None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    nonlinear_mode="IDENTITY",
    conv_mode="CROSS_CORRELATION",
    compute_mode="DEFAULT",
) -> Tensor:
    """
    Convolution bias with activation operation, only for inference.

    :param inp: feature map of the convolution operation.
    :param weight: convolution kernel.
    :param bias: bias added to the result of convolution
    :param stride: stride of the 2D convolution operation. Default: 1
    :param padding: size of the paddings added to the input on both sides of its spatial dimensions. Only zero-padding is supported. Default: 0
    :param dilation: dilation of the 2D convolution operation. Default: 1
    :param groups: number of groups into which the input and output channels are divided, so as to perform a "grouped convolution". When ``groups`` is not 1,
        ``in_channels`` and ``out_channels`` must be divisible by ``groups``,
        and the shape of weight should be `(groups, out_channel // groups,
        in_channels // groups, height, width)`.
    :type conv_mode: string or :class:`P.Convolution.Mode`.
    :param conv_mode: supports 'CROSS_CORRELATION' or 'CONVOLUTION'. Default:
        'CROSS_CORRELATION'
    :param dtype: support for ``np.dtype``, Default: np.int8
    :param scale: scale if use quantization, Default: 0.0
    :param zero_point: scale if use quantization quint8, Default: 0.0
    :type compute_mode: string or
        :class:`P.Convolution.ComputeMode`.
    :param compute_mode: when set to "DEFAULT", no special requirements will be
        placed on the precision of intermediate results. When set to "FLOAT32",
        "Float32" would be used for accumulator and intermediate result, but only effective when input and output are of Float16 dtype.

    """
    ph, pw = _pair(padding)
    sh, sw = _pair_nonzero(stride)
    dh, dw = _pair_nonzero(dilation)
    sparse_type = "DENSE" if groups == 1 else "GROUP"
    op = builtin.ConvBiasForward(
        stride_h=sh,
        stride_w=sw,
        pad_h=ph,
        pad_w=pw,
        dilate_h=dh,
        dilate_w=dw,
        dtype=dtype,
        format="NCHW",
        strategy=get_conv_execution_strategy(),
        nonlineMode=nonlinear_mode,
        mode=conv_mode,
        compute_mode=compute_mode,
        sparse=sparse_type,
    )
    (outputs,) = apply(op, inp, weight, bias)
    return outputs
