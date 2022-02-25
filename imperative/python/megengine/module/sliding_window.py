# -*- coding: utf-8 -*-
from typing import Tuple, Union

from ..functional import sliding_window, sliding_window_transpose
from .module import Module


class SlidingWindow(Module):
    r"""Apply a sliding window to input tensor and copy content in the window to
    corresponding output location. Assume input shape is :math:`(N, C, IH, IW)`,
    then output shape would be :math:`(N, C, OH, OW, window_h, window_w)` where
    :math:`(OH, OW)` would be computed from padding, stride, window and
    :math:`(IH, IW)`, as in convolution. For each output location, we have;

    .. math::

        out_{n, c, oh, ow, wh, ww} &= src_{n, c, ih+wh, iw+ww} \\
        \text{where } & ih=-pad_h+oh \times stride_h + (wh-1) \times (dilation_h-1) \\
                       & iw=-pad_w+ow \times stride_w + (ww-1) \times (dilation_w-1)

    Args:
        kernel_size: the size of the window to take a max over.
        padding: implicit zero padding to be added on both sides. Default: 0
        stride: the stride of the window. Default: 1
        dilation: the dilation of the window. Default: 1

    Example:
        >>> import numpy as np
        >>> inp = Tensor(np.arange(30).reshape(1,1,5,6))
        >>> op = M.SlidingWindow(kernel_size=3, padding=1, stride=2, dilation=2)
        >>> out = op(inp)
        >>> print(out.numpy())
        [[[[[[ 0  0  0]
             [ 0  7  9]
             [ 0 19 21]]
        <BLANKLINE>
            [[ 0  0  0]
             [ 7  9 11]
             [19 21 23]]]
        <BLANKLINE>
        <BLANKLINE>
           [[[ 0  7  9]
             [ 0 19 21]
             [ 0  0  0]]
        <BLANKLINE>
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


class SlidingWindowTranspose(Module):
    r"""Opposite opration of SlidingWindow, sum over the sliding windows on the
    corresponding input location. Given an input of the size
    :math:`(N, C,  IH, IW, window_h, window_w)` and :attr:`output_size`, the
    output shape would be :math:`(N, C, output\_size_{h}, output\_size_{w})` and the
    arguments must satisfy

    .. math::
        \text{IH} = \lfloor \frac{\text{output_size}_{h} + 2 * \text{padding}_{h} -
        \text{dilation}_{h} * (\text{kernel_size}_{h} - 1) - 1}{\text{stride}_{h}} + 1 \rfloor

    .. math::
        \text{IW} = \lfloor \frac{\text{output_size}_{w} + 2 * \text{padding}_{w} -
        \text{dilation}_{w} * (\text{kernel_size}_{w} - 1) - 1}{\text{stride}_{w}} + 1 \rfloor

    For each output location, we have:

    .. math::
        \text{out}_{n, c, oh, ow} = \sum_{n,c,oh,ow=location(n, c, ih, iw, wh, ww)}\text{src}_{n, c, ih, iw, wh, ww}

    .. math::
        \text{location}(n, c, ih, iw, wh, ww) &= (n, c, oh+wh, ow+ww) \\
        \text{where } & oh=-pad_h+ih \times stride_h + (wh-1) \times (dilation_h-1) \\
                       & ow=-pad_w+iw \times stride_w + (ww-1) \times (dilation_w-1)

    Args:
        output_size: the size of the output tensor.
        kernel_size: the size of the window to take a max over.
        padding: implicit zero padding to be added on both sides. Default: 0
        stride: the stride of the window. Default: 1
        dilation: the dilation of the window. Default: 1
    """

    def __init__(
        self,
        output_size: Union[int, Tuple[int, int]],
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]] = 0,
        stride: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        **kwargs
    ):
        super(SlidingWindowTranspose, self).__init__(**kwargs)
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

    def forward(self, inp):
        return sliding_window_transpose(
            inp,
            self.output_size,
            self.kernel_size,
            self.padding,
            self.stride,
            self.dilation,
        )
