# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from abc import abstractmethod
from typing import Tuple, Union

from ..functional import adaptive_avg_pool2d, adaptive_max_pool2d
from ..tensor import Parameter, Tensor
from .module import Module


class _AdaptivePoolNd(Module):
    def __init__(
        self, oshp: Union[Tuple[int, int], int, Tensor],
    ):
        super(_AdaptivePoolNd, self).__init__()
        self.oshp = oshp

    @abstractmethod
    def forward(self, inp):
        pass


class AdaptiveMaxPool2d(_AdaptivePoolNd):
    r"""
    Applies a 2D max adaptive pooling over an input.

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

    Examples:

    .. testcode::

        import numpy as np
        import megengine as mge
        import megengine.module as M

        m = M.AdaptiveMaxPool2d((2, 2))
        inp = mge.tensor(np.arange(0, 16).astype("float32").reshape(1, 1, 4, 4))
        oup = m(inp)
        print(oup.numpy())

    Outputs:

    .. testoutput::

        [[[[ 5.  7.]
           [13. 15.]]]]

    """

    def forward(self, inp):
        return adaptive_max_pool2d(inp, self.oshp)


class AdaptiveAvgPool2d(_AdaptivePoolNd):
    r"""
    Applies a 2D average pooling over an input.

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

    Examples:

    .. testcode::

        import numpy as np
        import megengine as mge
        import megengine.module as M

        m = M.AdaptiveAvgPool2d((2, 2))
        inp = mge.tensor(np.arange(0, 16).astype("float32").reshape(1, 1, 4, 4))
        oup = m(inp)
        print(oup.numpy())

    Outputs:

    .. testoutput::

        [[[[ 2.5  4.5]
           [10.5 12.5]]]]

    """

    def forward(self, inp):
        return adaptive_avg_pool2d(inp, self.oshp)
