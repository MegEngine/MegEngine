# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# pylint: disable=too-many-lines
from typing import Optional, Tuple, Union

import megengine._internal as mgb
from megengine._internal import CompGraph, CompNode

from ..core import Tensor, wrap_io_tensor
from ..core.graph import _use_default_if_none
from ..jit import barrier, mark_impure
from ..random import uniform
from ..utils.types import _pair, _pair_nonzero
from .debug_param import get_conv_execution_strategy
from .tensor import concat
from .utils import _decide_comp_node_and_comp_graph


@wrap_io_tensor
def linear(inp: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """Applies a linear transformation to the input.

    Refer to :class:`~.Linear` for more information.
    """
    orig_shape = inp.shape
    inp = inp.reshape(-1, orig_shape[-1])
    ret = mgb.opr.matrix_mul(inp, weight, transposeB=True)
    ret = ret.reshape(orig_shape[:-1], weight.shape[0])
    if bias is not None:
        ret += bias
    return ret


@wrap_io_tensor
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
    :type conv_mode: string or :class:`mgb.opr_param_defs.Convolution.Mode`
    :param conv_mode: Supports 'CROSS_CORRELATION' or 'CONVOLUTION'. Default:
        'CROSS_CORRELATION'.
    :type compute_mode: string or
        :class:`mgb.opr_param_defs.Convolution.ComputeMode`
    :param compute_mode: When set to 'DEFAULT', no special requirements will be
        placed on the precision of intermediate results. When set to 'FLOAT32',
        Float32 would be used for accumulator and intermediate result, but only
        effective when input and output are of Float16 dtype.

    Refer to :class:`~.Conv2d` for more information.
    """
    ph, pw = _pair(padding)
    sh, sw = _pair_nonzero(stride)
    dh, dw = _pair_nonzero(dilation)
    Sparse = mgb.opr_param_defs.Convolution.Sparse
    sparse_type = Sparse.DENSE if groups == 1 else Sparse.GROUP
    res = mgb.opr.convolution(
        inp,
        weight,
        pad_h=ph,
        pad_w=pw,
        stride_h=sh,
        stride_w=sw,
        dilate_h=dh,
        dilate_w=dw,
        format="NCHW",
        strategy=get_conv_execution_strategy(),
        mode=conv_mode,
        compute_mode=compute_mode,
        sparse=sparse_type,
    )
    if bias is not None:
        res += bias
    return res


@wrap_io_tensor
def max_pool2d(
    inp: Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
) -> Tensor:
    """Applies a 2D max pooling over an input.

    :param inp: The input tensor.
    :param kernel_size: The size of the window.
    :param stride: The stride of the window. If not provided, its value is set to ``kernel_size``.
        Default: None
    :param padding: Implicit zero padding to be added on both sides. Default: 0
    
    Refer to :class:`~.MaxPool2d` for more information.
    """

    kh, kw = _pair_nonzero(kernel_size)
    sh, sw = _pair_nonzero(stride or kernel_size)
    ph, pw = _pair(padding)
    mode = mgb.opr_param_defs.Pooling.Mode.MAX
    return mgb.opr.pooling(
        inp,
        mode=mode,
        format="NCHW",
        stride_h=sh,
        stride_w=sw,
        pad_h=ph,
        pad_w=pw,
        window_h=kh,
        window_w=kw,
    )


@wrap_io_tensor
def avg_pool2d(
    inp: Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
) -> Tensor:
    """ Applies a 2D average pooling over an input.

    :param inp: The input tensor.
    :param kernel_size: The size of the window.
    :param stride: The stride of the window. If not provided, its value is set to ``kernel_size``.
        Default: None
    :param padding: Implicit zero padding to be added on both sides. Default: 0

    Refer to :class:`~.AvgPool2d` for more information.
    """
    kh, kw = _pair_nonzero(kernel_size)
    sh, sw = _pair_nonzero(stride or kernel_size)
    ph, pw = _pair(padding)
    mode = mgb.opr_param_defs.Pooling.Mode.AVERAGE
    return mgb.opr.pooling(
        inp,
        mode=mode,
        format="NCHW",
        stride_h=sh,
        stride_w=sw,
        pad_h=ph,
        pad_w=pw,
        window_h=kh,
        window_w=kw,
    )


@wrap_io_tensor
def prelu(inp: Tensor, weight: Tensor) -> Tensor:
    r"""
    Applies the element-wise PReLU function.

    Refer to :class:`~.PReLU` for more information.
    """

    return mgb.opr.elemwise(inp, 0, mode="MAX") + weight * mgb.opr.elemwise(
        inp, 0, mode="MIN"
    )


@wrap_io_tensor
def leaky_relu(inp: Tensor, negative_slope: float = 0.01) -> Tensor:
    r"""
    Applies the element-wise leaky_relu function

    Refer to :class:`~.LeakyReLU` for more information.
    """

    return mgb.opr.elemwise(inp, 0, mode="MAX") + negative_slope * mgb.opr.elemwise(
        inp, 0, mode="MIN"
    )


@wrap_io_tensor
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
        target_shape += (inp.shape[end_axis + 1 :],)
    return inp.reshape(*target_shape)


def _get_softmax_axis(ndim: int) -> int:
    if ndim in (0, 1, 3):
        return 0
    return 1


@wrap_io_tensor
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

    """
    if axis is None:
        axis = _get_softmax_axis(len(inp.imm_shape))
    offset = mgb.opr.zero_grad(inp.max(axis=axis, keepdims=True))
    inp = inp - offset
    down = mgb.opr.elem.exp(inp).sum(axis=axis, keepdims=True)
    return mgb.opr.elem.exp(inp) / down


@wrap_io_tensor
def batch_norm2d(
    inp: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    training: bool = False,
    momentum: float = 0.9,
    eps: float = 1e-5,
) -> Tensor:
    """Applies batch normalization to the input.

    :type inp: Tensor
    :param inp: The input tensor.
    :type num_features: int
    :param num_features: usually the :math:`C` from an input of size
        :math:`(N, C, H, W)` or the highest ranked dimension of an input with
        less than 4D.
    :type eps: float
    :param eps: a value added to the denominator for numerical stability.
        Default: 1e-5.
    :type momentum: float
    :param momentum: the value used for the `running_mean` and `running_var`
        computation.
        Default: 0.1
    :type affine: bool
    :param affine: a boolean value that when set to ``True``, this module has
        learnable affine parameters. Default: ``True``
    :type track_running_stats: bool
    :param track_running_stats: when set to ``True``, this module tracks the
        running mean and variance. When set to ``False``, this module does not
        track such statistics and always uses batch statistics in both training
        and eval modes. Default: ``True``.

    Refer to :class:`~.BatchNorm2d` and :class:`~.BatchNorm1d` for more information.
    """

    inp = mgb.opr.mark_no_broadcast_elemwise(inp)
    _channels = inp.imm_shape[1]
    _ndim = len(inp.imm_shape)
    _param_shape = (1, _channels) + (1,) * (_ndim - 2)

    assert _ndim == 4, "only 4D tensor supported"

    if weight is not None:
        weight = weight.reshape(*_param_shape)
    else:
        weight = mgb.make_immutable(*_use_default_if_none(None, None), 1.0).broadcast(
            *_param_shape
        )

    if bias is not None:
        bias = bias.reshape(*_param_shape)
    else:
        bias = mgb.make_immutable(*_use_default_if_none(None, None), 0.0).broadcast(
            *_param_shape
        )

    FwdMode = mgb.opr_param_defs.BN.FwdMode
    fwdmode = FwdMode.TRAINING if training else FwdMode.INFERENCE
    avg_factor = 1 - momentum

    if running_mean is not None and running_var is not None:
        if training:
            inp = barrier(inp)

        output = mgb.opr.batch_norm(
            inp,
            weight,
            bias,
            running_mean,
            running_var,
            param_dim="DIM_1C11",
            fwd_mode=fwdmode,
            epsilon=eps,
            avg_factor=avg_factor,
        )[-1]
        if training:
            mark_impure(output)
    else:
        output = mgb.opr.batch_norm_no_statistic(
            inp,
            weight,
            bias,
            param_dim="DIM_1C11",
            fwd_mode=fwdmode,
            epsilon=eps,
            avg_factor=avg_factor,
        )[-1]

    return output


def one_hot(inp: Tensor, num_classes: int = -1) -> Tensor:
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
        out = F.one_hot(inp)
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[0 1 0 0]
         [0 0 1 0]
         [0 0 0 1]]

    """
    comp_node, comp_graph = _decide_comp_node_and_comp_graph(inp)

    if num_classes == -1:
        num_classes = inp.max() + 1
    zeros = mgb.make_immutable(value=0, comp_node=comp_node, comp_graph=comp_graph)
    zeros_symvar = zeros.broadcast(inp.shapeof(), num_classes)

    ones = mgb.make_immutable(value=1, comp_node=comp_node, comp_graph=comp_graph)
    ones_symvar = ones.broadcast(inp.shapeof(), 1)

    return Tensor(
        mgb.opr.indexing_set_one_hot(
            zeros_symvar, axis=len(inp.shapeof()), index=inp, value=ones_symvar
        )
    )


@wrap_io_tensor
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

    return mgb.opr.warp_perspective(
        inp,
        M,
        dsize,
        bmode=border_mode,
        border_val=border_val,
        imode=interp_mode,
        format="NCHW",
    )


@wrap_io_tensor
def eye(
    n: int,
    m: Optional[int] = None,
    *,
    dtype=None,
    device: Optional[CompNode] = None,
    comp_graph: Optional[CompGraph] = None
) -> Tensor:
    """
    Fills the 2-dimensional input :class:`SymbolVar` with the identity matrix.

    :param n: The number of rows
    :param m: The number of columns, default to None
    :param dtype: The data type, default to None
    :param device: Compute node of the matrix, defaults to None
    :param comp_graph: Compute graph of the matrix, defaults to None
    :return: The eye matrix

    Examples:

    .. testcode::

        import numpy as np
        import megengine.functional as F

        data_shape = (4, 6)
        n, m = data_shape
        out = F.eye(n, m, dtype=np.float32)
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[1. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0.]]

    """

    device, comp_graph = _use_default_if_none(device, comp_graph)
    if m is None:
        m = n
    return mgb.opr.eye((n, m), dtype=dtype, comp_node=device, comp_graph=comp_graph)


@wrap_io_tensor
def matrix_mul(inp1: Tensor, inp2: Tensor) -> Tensor:
    """
    Performs a matrix multiplication of the matrices ``inp1`` and ``inp2``

    :param inp1: The first matrix to be multiplied (a, b)
    :param inp2: The second matrix to be multiplied (b, c)
    :return: The output tensor (a, c)

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        shape_1 = (2, 3)
        shape_2 = (3, 4)
        data1 = tensor(np.arange(0, 6, dtype=np.float32).reshape(2, 3))
        data2 = tensor(np.arange(0, 6, dtype=np.float32).reshape(3, 2))
        out = F.matrix_mul(data1, data2)
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[10. 13.]
         [28. 40.]]

    """
    return mgb.opr.matrix_mul(inp1, inp2)


@wrap_io_tensor
def batched_matrix_mul(inp1: Tensor, inp2: Tensor) -> Tensor:
    """
    Performs a batched multiplication of th batched matrices ``inp1`` and ``inp2``

    :param inp1: The first batch matrix to be multiplied (n, a, b)
    :param inp2: The second batch matrix to be multiplied (n, b, c)
    :return: The output batch (n, a, c)

    Examples:

    .. testcode::

        import numpy as np
        from megengine import tensor
        import megengine.functional as F

        batch_size = 3
        shape_1 = (batch_size, 2, 3)
        shape_2 = (batch_size, 3, 4)
        data1 = tensor(
            np.arange(0, batch_size * 6, dtype=np.float32).reshape(batch_size, 2, 3))
        data2 = tensor(
            np.arange(0, batch_size * 12, dtype=np.float32).reshape(batch_size, 3, 4))
        out = F.batched_matrix_mul(data1, data2)
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[[  20.   23.   26.   29.]
          [  56.   68.   80.   92.]]

         [[ 344.  365.  386.  407.]
          [ 488.  518.  548.  578.]]

         [[1100. 1139. 1178. 1217.]
          [1352. 1400. 1448. 1496.]]]

    """
    return mgb.opr.batched_matrix_mul(inp1, inp2)


@wrap_io_tensor
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
        'bilinear'(default), 'linear', 'nearest' (todo), 'cubic' (todo), 'area' (todo)

    

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
        :options: +NUMBER

        [[[[1.   1.25 1.75 2.  ]
           [1.5  1.75 2.25 2.5 ]
           [2.5  2.75 3.25 3.5 ]
           [3.   3.25 3.75 4.  ]]]]

    """
    mode = mode.upper()
    if mode not in ["BILINEAR", "LINEAR"]:
        raise ValueError("interpolate only support bilinear mode")
    if mode not in ["BILINEAR", "LINEAR"]:
        if align_corners is not None:
            raise ValueError(
                "align_corners option can only be set in the bilinear/linear interpolating mode"
            )
    else:
        if align_corners is None:
            align_corners = False

    if mode == "LINEAR":
        inp = mgb.opr.add_axis(inp, 3)

    if len(inp.imm_shape) != 4:
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
            mgb.opr.elemwise(inp.shape[i + 2] * scale_factor[i], mode="FLOOR")
            for i in range(2)
        )
        dsize = mgb.opr.concat([dsize[0], dsize[1]], axis=0)
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
        row0 = mgb.opr.concat([wscale, [0, 0]], axis=0).reshape(1, 3)
        row1 = mgb.opr.concat([[0], hscale, [0]], axis=0).reshape(1, 3)
        weight = mgb.opr.concat([row0, row1, [[0, 0, 1]]], axis=0).reshape(1, 3, 3)
        weight = mgb.opr.broadcast(weight, (inp.shape[0], 3, 3))
    else:
        hscale = 1.0 * ih / oh
        wscale = 1.0 * iw / ow
        row0 = mgb.opr.concat([wscale, [0], 0.5 * wscale - 0.5], axis=0).reshape(1, 3)
        row1 = mgb.opr.concat([[0], hscale, 0.5 * hscale - 0.5], axis=0).reshape(1, 3)
        weight = mgb.opr.concat([row0, row1, [[0, 0, 1]]], axis=0).reshape(1, 3, 3)
        weight = mgb.opr.broadcast(weight, (inp.shape[0], 3, 3))

    ret = mgb.opr.warp_perspective(inp, weight, dsize, imode="LINEAR", format="NCHW")
    if mode == "LINEAR":
        ret = mgb.opr.reshape(ret, ret.shape[0:3])
    return ret


@wrap_io_tensor
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
        from megengine import tensor
        import megengine.functional as F

        data = tensor(np.ones(10, dtype=np.float32))
        out = F.dropout(data, 1./3.)
        print(out.numpy())

    Outputs:

    .. testoutput::

        [1.5 1.5 0.  1.5 1.5 1.5 1.5 1.5 1.5 1.5]

    """
    assert 0 <= drop_prob < 1
    rv = uniform(inp.shape)
    mask = rv > drop_prob
    inp *= mask.astype(inp.dtype)
    if rescale:
        inp *= 1 / (1 - drop_prob)
    return inp


@wrap_io_tensor
def identity(inp: Tensor) -> Tensor:
    """applies an identity transform to the input tensor.
    
    :param inp: The input tensor
    """
    return mgb.opr.identity(inp)


@wrap_io_tensor
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

    return mgb.opr.advanced_indexing(weight)[input.reshape(-1), :].reshape(
        input.shape, weight.shape[-1]
    )


@wrap_io_tensor
def roi_pooling(
    input: Tensor,
    rois: Tensor,
    output_shape: Union[int, tuple, list],
    mode: str = "max",
    scale: float = 1.0,
) -> Tensor:
    """
    Apply roi pooling on input feature

    :param input: tensor that represents the input feature, (N, C, H, W) images
    :param rois: (K, 5) boxes. First column is the index into N. The other 4 columns are xyxy
    :param output_shape: (height, width) of output rois feature
    :param mode: "max" or "average", use max/average align just like max/average pooling. Default: ``"max"``
    :param scale: scale the input boxes by this number. Default: 1.0
    :return: (K, C, output_shape[0], output_shape[1]) feature of rois
    """
    assert mode in ["max", "average"], "only max/average mode is supported"
    if isinstance(output_shape, int):
        output_shape = (output_shape, output_shape)

    return mgb.opr.roi_pooling(
        input, rois, output_shape, mode=mode.upper(), scale=scale
    )


@wrap_io_tensor
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

    return mgb.opr.roi_align(
        input,
        rois,
        mode=mode.upper(),
        spatial_scale=spatial_scale,
        offset=offset,
        pooled_height=pooled_height,
        pooled_width=pooled_width,
        sample_height=sample_height,
        sample_width=sample_width,
    )


@wrap_io_tensor
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

    return mgb.opr.assert_equal(get, expect, maxerr=max_err, verbose=verbose)


@wrap_io_tensor
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

    return mgb.opr.indexing_one_hot(src, axis, index, keepdims=keepdims)
