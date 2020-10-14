# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import itertools

import numpy as np

from megengine import Parameter, tensor
from megengine.module import ConvTranspose2d, LocalConv2d


def test_conv_transpose2d():
    SH, SW = 3, 1
    PH, PW = 2, 0
    N, IC, IH, IW = 4, 5, 8, 6
    KH, KW = 3, 4
    OC = 3
    BIAS = False

    def getsize(inp, kern, stride):
        return (inp - 1) * stride + kern

    OH = getsize(IH, KH, SH)
    OW = getsize(IW, KW, SW)

    inp = np.random.normal(size=(N, IC, IH, IW)).astype(np.float32)
    out = np.zeros((N, OC, OH, OW), dtype=np.float32)
    weight = np.random.normal(size=(IC, OC, KH, KW)).astype(np.float32)
    bias = np.random.normal(size=(1, OC, 1, 1)).astype(np.float32)

    # naive calculation use numpy
    for n, ic, ih, iw in itertools.product(*map(range, [N, IC, IH, IW])):
        oh, ow = ih * SH, iw * SW
        out[n, :, oh : oh + KH, ow : ow + KW] += inp[n, ic, ih, iw] * weight[ic]
    out = out[:, :, PH : OH - PH, PW : OW - PW]
    if BIAS:
        out += bias

    # megengine conv_transpose2d calculation
    conv_transpose2d = ConvTranspose2d(IC, OC, (KH, KW), (SH, SW), (PH, PW), bias=BIAS)
    conv_transpose2d.weight = Parameter(weight, dtype=np.float32)
    if BIAS:
        conv_transpose2d.bias = Parameter(bias, dtype=np.float32)
    y = conv_transpose2d(tensor(inp))

    np.testing.assert_almost_equal(out, y.numpy(), 2e-6)


def test_local_conv2d():
    def test_func(
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
    ):
        local_conv2d = LocalConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            input_height=input_height,
            input_width=input_width,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        inputs = np.random.normal(
            size=(batch_size, in_channels, input_height, input_width)
        ).astype(np.float32)
        output_height = (input_height + padding * 2 - kernel_size) // stride + 1
        output_width = (input_width + padding * 2 - kernel_size) // stride + 1
        weights = np.random.normal(
            size=(
                groups,
                output_height,
                output_width,
                in_channels // groups,
                kernel_size,
                kernel_size,
                out_channels // groups,
            )
        ).astype(np.float32)
        local_conv2d.weight = Parameter(weights)
        outputs = local_conv2d(tensor(inputs))
        # naive calculation use numpy
        # only test output_height == input_height, output_width == input_width
        inputs = np.pad(inputs, ((0, 0), (0, 0), (1, 1), (1, 1)))
        expected = np.zeros(
            (batch_size, out_channels, output_height, output_width), dtype=np.float32,
        )
        ic_group_size = in_channels // groups
        oc_group_size = out_channels // groups
        for n, oc, oh, ow in itertools.product(
            *map(range, [batch_size, out_channels, output_height, output_width])
        ):
            ih, iw = oh * stride, ow * stride
            g_id = oc // oc_group_size
            expected[n, oc, ih, iw] = np.sum(
                inputs[
                    n,
                    g_id * ic_group_size : (g_id + 1) * ic_group_size,
                    ih : ih + kernel_size,
                    iw : iw + kernel_size,
                ]
                * weights[g_id, oh, ow, :, :, :, oc % oc_group_size]
            )
        np.testing.assert_almost_equal(outputs.numpy(), expected, 1e-5)

    test_func(10, 4, 4, 5, 5, 3, 1, 1, 1, 1)
    test_func(10, 32, 32, 8, 8, 3, 1, 1, 1, 2)
    test_func(10, 32, 32, 8, 8, 3, 1, 1, 1, 4)
