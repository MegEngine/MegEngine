# -*- coding: utf-8 -*-
from abc import abstractmethod
from typing import Tuple, Union

from ..functional import adaptive_avg_pool2d, adaptive_max_pool2d
from ..tensor import Parameter, Tensor
from .module import Module


class _AdaptivePoolNd(Module):
    def __init__(self, oshp: Union[Tuple[int, int], int, Tensor], **kwargs):
        super(_AdaptivePoolNd, self).__init__(**kwargs)
        self.oshp = oshp

    @abstractmethod
    def forward(self, inp):
        pass


class AdaptiveMaxPool2d(_AdaptivePoolNd):
    r"""Applies a 2D max adaptive pooling over an input.

    For instance, given an input of the size :math:`(N, C, H, W)` and
    an output shape :math:`(OH, OW)`, this layer generates the output of
    the size :math:`(N, C, OH, OW)` through a process described as:

    .. math::
        \begin{aligned}
            out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1}
                \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                \text{stride[1]} \times w + n)
        \end{aligned}

    ``kernel_size`` and ``stride`` can be inferred from input shape and out shape:

    * padding: (0, 0)
    * stride: (floor(IH / OH), floor(IW / OW))
    * kernel_size: (IH - (OH - 1) * stride_h, IW - (OW - 1) * stride_w)

    Args:
        oshp(Union[Tuple[int, int], int, Tensor]): the target output shape of the image of the form Height * Width.
            Can be tuple (H, W) or a single H for a square image H * H.
        
    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where
          :math:`(H_{out}, W_{out})=\text{output\_shape}`.

    Returns:
        Return type: module. The instance of the ``AdaptiveMaxPool2d`` module.

    Examples:
        >>> import numpy as np
        >>> m = M.AdaptiveMaxPool2d((2, 2))
        >>> inp = mge.tensor(np.arange(0, 16).astype("float32").reshape(1, 1, 4, 4))
        >>> oup = m(inp)
        >>> oup.numpy()
        array([[[[ 5.,  7.],
                 [13., 15.]]]], dtype=float32)
    """

    def forward(self, inp):
        return adaptive_max_pool2d(inp, self.oshp)


class AdaptiveAvgPool2d(_AdaptivePoolNd):
    r"""Applies a 2D average pooling over an input.

    For instance, given an input of the size :math:`(N, C, H, W)` and
    an output shape :math:`(OH, OW)`, this layer generates the output of
    the size :math:`(N, C, OH, OW)` through a process described as:

    .. math::

        out(N_i, C_j, h, w)  = \frac{1}{kH * kW} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
                               input(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)

    ``kernel_size`` and ``stride`` can be inferred from input shape and out shape:

    * padding: (0, 0)
    * stride: (floor(IH / OH), floor(IW / OW))
    * kernel_size: (IH - (OH - 1) * stride_h, IW - (OW - 1) * stride_w)

    Args:
        oshp(Union[Tuple[int, int], int, Tensor]): the target output shape of the image of the form Height * Width.
            Can be tuple (H, W) or a single H for a square image H * H.

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or :math:`(C, D_{out}, H_{out}, W_{out})`,
          where :math:`(D_{out}, H_{out}, W_{out})=\text{output\_shape}`.

    Returns:
        Return type: module. The instance of the ``AdaptiveAvgPool2d`` module.

    Examples:
        >>> import numpy as np
        >>> m = M.AdaptiveAvgPool2d((2, 2))
        >>> inp = mge.tensor(np.arange(0, 16).astype("float32").reshape(1, 1, 4, 4))
        >>> oup = m(inp)
        >>> oup.numpy()
        array([[[[ 2.5,  4.5],
                 [10.5, 12.5]]]], dtype=float32)
    """

    def forward(self, inp):
        return adaptive_avg_pool2d(inp, self.oshp)
