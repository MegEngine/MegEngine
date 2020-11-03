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

from ..functional import avg_pool2d, max_pool2d
from .module import Module


class _PoolNd(Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
    ):
        super(_PoolNd, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    @abstractmethod
    def forward(self, inp):
        pass

    def _module_info_string(self) -> str:
        return "kernel_size={kernel_size}, stride={stride}, padding={padding}".format(
            **self.__dict__
        )


class MaxPool2d(_PoolNd):
    r"""
    Applies a 2D max pooling over an input.

    For instance, given an input of the size :math:`(N, C, H, W)` and
    :attr:`kernel_size` :math:`(kH, kW)`, this layer generates the output of
    the size :math:`(N, C, H_{out}, W_{out})` through a process described as:

    .. math::
        \begin{aligned}
            out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1}
                \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                \text{stride[1]} \times w + n)
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on
    both sides for :attr:`padding` number of points.

    :param kernel_size: the size of the window to take a max over.
    :param stride: the stride of the window. Default value is kernel_size.
    :param padding: implicit zero padding to be added on both sides.

    Examples:

    .. testcode::

        import numpy as np
        import megengine as mge
        import megengine.module as M

        m = M.MaxPool2d(kernel_size=3, stride=1, padding=0)
        inp = mge.tensor(np.arange(0, 16).astype("float32").reshape(1, 1, 4, 4))
        oup = m(inp)
        print(oup.numpy())

    Outputs:

    .. testoutput::

        [[[[10. 11.]
           [14. 15.]]]]

    """

    def forward(self, inp):
        return max_pool2d(inp, self.kernel_size, self.stride, self.padding)


class AvgPool2d(_PoolNd):
    r"""
    Applies a 2D average pooling over an input.

    For instance, given an input of the size :math:`(N, C, H, W)` and
    :attr:`kernel_size` :math:`(kH, kW)`, this layer generates the output of
    the size :math:`(N, C, H_{out}, W_{out})` through a process described as:

    .. math::

        out(N_i, C_j, h, w)  = \frac{1}{kH * kW} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
                               input(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on
    both sides for :attr:`padding` number of points.

    :param kernel_size: the size of the window.
    :param stride: the stride of the window. Default value is kernel_size。
    :param padding: implicit zero padding to be added on both sides.

    Examples:

    .. testcode::

        import numpy as np
        import megengine as mge
        import megengine.module as M

        m = M.AvgPool2d(kernel_size=3, stride=1, padding=0)
        inp = mge.tensor(np.arange(0, 16).astype("float32").reshape(1, 1, 4, 4))
        oup = m(inp)
        print(oup.numpy())

    Outputs:

    .. testoutput::

        [[[[ 5.  6.]
           [ 9. 10.]]]]

    """

    def forward(self, inp):
        return avg_pool2d(inp, self.kernel_size, self.stride, self.padding)
