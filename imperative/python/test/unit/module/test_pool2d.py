# -*- coding: utf-8 -*-
import itertools

import numpy as np

from megengine import Parameter, tensor
from megengine.module import AvgPool2d, MaxPool2d


def test_avg_pool2d():
    def test_func(
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_size,
        stride,
        padding,
    ):
        pool = AvgPool2d(kernel_size, stride=stride, padding=padding, mode="average")
        inp = np.random.normal(
            size=(batch_size, in_channels, in_height, in_width)
        ).astype(np.float32)
        out_height = (in_height + padding * 2 - kernel_size) // stride + 1
        out_width = (in_width + padding * 2 - kernel_size) // stride + 1
        out = pool(tensor(inp))
        inp = np.pad(inp, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
        expected = np.zeros(
            (batch_size, out_channels, out_height, out_width), dtype=np.float32,
        )
        for n, c, oh, ow in itertools.product(
            *map(range, [batch_size, out_channels, out_height, out_width])
        ):
            ih, iw = oh * stride, ow * stride
            expected[n, c, oh, ow] = np.sum(
                inp[n, c, ih : ih + kernel_size, iw : iw + kernel_size,]
            ) / (kernel_size * kernel_size)
        np.testing.assert_almost_equal(out.numpy(), expected, 1e-5)

    test_func(10, 4, 4, 5, 5, 2, 2, 1)
    test_func(10, 4, 4, 6, 6, 2, 2, 0)
    test_func(10, 16, 16, 14, 14, 2, 2, 0)
