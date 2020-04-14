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
from megengine.module import ConvTranspose2d
from megengine.test import assertTensorClose


def test_conv_transpose2d():
    SH, SW = 3, 1
    PH, PW = 2, 0
    N, IC, IH, IW = 4, 5, 8, 6
    KH, KW = 3, 4
    OC = 3
    BIAS = True

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

    assertTensorClose(out, y.numpy(), max_err=2e-6)
