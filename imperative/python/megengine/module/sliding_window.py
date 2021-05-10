# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Tuple, Union

from ..functional import sliding_window
from .module import Module


class SlidingWindow(Module):
    r"""
    Apply a sliding window to input tensor and copy content in the window to
    corresponding output location. Assume input shape is :math:`(N, C, IH, IW)`,
    then output shape would be :math:`(N, C, OH, OW, window_h, window_w)` where
    :math:`(OH, OW)` would be computed from padding, stride, window and
    :math:`(IH, IW)`, as in convolution. For each output location, we have;

    .. math::

        out_{n, c, oh, ow, wh, ww} &= src_{n, c, ih+wh, iw+ww} \\
        \text{where } & ih=-pad_h+oh \times stride_h + (wh-1) \times (dilation_h-1) \\
                       & iw=-pad_w+ow \times stride_w + (ww-1) \times (dilation_w-1)


    :param kernel_size: the size of the window to take a max over.
    :param padding: implicit zero padding to be added on both sides. Default: 0
    :param stride: the stride of the window. Default: 1
    :param dilation: the dilation of the window. Default: 1

    Example:

    .. testcode::

        from megengine import tensor
        import megengine.module as M
        import numpy as np

        inp = tensor(np.arange(30).reshape(1,1,5,6))
        op = M.SlidingWindow(kernel_size=3, padding=1, stride=2, dilation=2)
        out = op(inp)
        print(out.numpy())

    Outputs:

    .. testoutput::

        [[[[[[ 0  0  0]
             [ 0  7  9]
             [ 0 19 21]]

            [[ 0  0  0]
             [ 7  9 11]
             [19 21 23]]]


           [[[ 0  7  9]
             [ 0 19 21]
             [ 0  0  0]]

            [[ 7  9 11]
             [19 21 23]
             [ 0  0  0]]]]]]

    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]] = 0,
        stride: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        **kwargs
    ):
        super(SlidingWindow, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def forward(self, inp):
        return sliding_window(
            inp, self.kernel_size, self.padding, self.stride, self.dilation
        )
