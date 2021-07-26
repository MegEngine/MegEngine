# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import itertools

import numpy as np
import pytest

import megengine.module as M
from megengine import Parameter, tensor
from megengine.functional.debug_param import (
    get_execution_strategy,
    set_execution_strategy,
)
from megengine.module import ConvTranspose2d, ConvTranspose3d, LocalConv2d


@pytest.fixture
def reproducible():
    old = get_execution_strategy()
    set_execution_strategy("HEURISTIC_REPRODUCIBLE")
    yield
    set_execution_strategy(old)


# NOTE: test in module for convenience. should really test in functional
@pytest.mark.parametrize(
    "name",
    ["Conv1d", "Conv2d", "Conv3d", "ConvTranspose2d", "ConvTranspose3d", "LocalConv2d"],
)
def test_conv_dtype_promotion(name, reproducible):
    N, Ci, Co, K = 2, 16, 32, 3
    S = (7,) * int(name[-2])
    if "Local" in name:
        m = getattr(M, name)(Ci, Co, *S, K)
    else:
        m = getattr(M, name)(Ci, Co, K)
    x = tensor(np.random.random(size=(N, Ci) + S).astype("float16"))
    np.testing.assert_equal(m(x).numpy(), m(x.astype("float32")).numpy())


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
        weights = local_conv2d.weight.numpy()
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


def test_conv_transpose3d():
    def getsize(inp, kernel, stride, dilate):
        return (inp - 1) * stride + kernel * dilate - dilate + 1

    def test_func(
        N,
        IC,
        ID,
        IH,
        IW,
        OC,
        KD,
        KH,
        KW,
        SD,
        SH,
        SW,
        PD,
        PH,
        PW,
        DD,
        DH,
        DW,
        bias=True,
    ):
        conv_transpose3d = ConvTranspose3d(
            in_channels=IC,
            out_channels=OC,
            kernel_size=(KD, KH, KW),
            stride=(SD, SH, SW),
            padding=(PD, PH, PW),
            dilation=(DD, DH, DW),
            bias=bias,
        )

        OD = getsize(ID, KD, SD, DD)
        OH = getsize(IH, KH, SH, DH)
        OW = getsize(IW, KW, SW, DW)

        inp = np.random.normal(size=(N, IC, ID, IH, IW))
        weight = np.random.normal(size=(IC, OC, KD, KH, KW))
        out_np = np.zeros((N, OC, OD, OH, OW), dtype=np.float32)

        for n, ic, idepth, ih, iw in itertools.product(
            *map(range, [N, IC, ID, IH, IW])
        ):
            od, oh, ow = idepth * SD, ih * SH, iw * SW
            out_np[n, :, od : od + KD, oh : oh + KH, ow : ow + KW] += (
                inp[n, ic, idepth, ih, iw] * weight[ic]
            )
        out_np = out_np[:, :, PD : OD - PD, PH : OH - PH, PW : OW - PW]

        assert conv_transpose3d.weight.numpy().shape == weight.shape
        conv_transpose3d.weight = Parameter(weight)
        out_meg = conv_transpose3d.forward(tensor(inp))

        np.testing.assert_almost_equal(out_meg.numpy(), out_np, 1e-5)

    test_func(4, 3, 8, 16, 16, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    test_func(4, 8, 16, 32, 32, 16, 1, 3, 1, 2, 1, 2, 0, 1, 0, 1, 1, 1)
