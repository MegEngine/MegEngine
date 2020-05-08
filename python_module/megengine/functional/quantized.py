# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# pylint: disable=too-many-lines
from typing import Tuple, Union

from .. import _internal as mgb
from ..core import Tensor, wrap_io_tensor
from ..utils.types import _pair, _pair_nonzero
from .debug_param import get_conv_execution_strategy


@wrap_io_tensor
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
    """ convolution bias with activation operation, only for inference.

    :param inp: The feature map of the convolution operation
    :param weight: The convolution kernel
    :param bias: The bias added to the result of convolution
    :param stride: Stride of the 2D convolution operation. Default: 1
    :param padding: Size of the paddings added to the input on both sides of its
        spatial dimensions. Only zero-padding is supported. Default: 0
    :param dilation: Dilation of the 2D convolution operation. Default: 1
    :param groups: number of groups to divide input and output channels into,
        so as to perform a "grouped convolution". When ``groups`` is not 1,
        ``in_channels`` and ``out_channels`` must be divisible by ``groups``,
        and the shape of weight should be ``(groups, out_channel // groups,
        in_channels // groups, height, width)``.
    :type conv_mode: string or :class:`mgb.opr_param_defs.Convolution.Mode`
    :param conv_mode: Supports 'CROSS_CORRELATION' or 'CONVOLUTION'. Default:
        'CROSS_CORRELATION'.
    :param dtype:  Support for np.dtype, Default:
        np.int8.
    :param scale:  scale if use quantization, Default:
        0.0.
    :param zero_point:  scale if use quantization quint8, Default:
        0.0.
    :type compute_mode: string or
        :class:`mgb.opr_param_defs.Convolution.ComputeMode`
    :param compute_mode: When set to 'DEFAULT', no special requirements will be
        placed on the precision of intermediate results. When set to 'FLOAT32',
        Float32 would be used for accumulator and intermediate result, but only
        effective when input and output are of Float16 dtype.

    """
    ph, pw = _pair(padding)
    sh, sw = _pair_nonzero(stride)
    dh, dw = _pair_nonzero(dilation)
    sparse_type = "DENSE" if groups == 1 else "GROUP"
    res = mgb.opr.conv_bias_activation(
        inp,
        weight,
        bias,
        compute_mode=compute_mode,
        dtype=dtype,
        strategy=get_conv_execution_strategy(),
        nonlineMode=nonlinear_mode,
        sparse=sparse_type,
        format="NCHW",
        pad_h=ph,
        pad_w=pw,
        stride_h=sh,
        stride_w=sw,
        dilate_h=dh,
        dilate_w=dw,
        mode=conv_mode,
    )
    return res
