# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np
import pytest

import megengine.functional as F
from megengine import tensor
from megengine.test import assertTensorClose


def test_linear_interpolate():
    inp = tensor(np.arange(1, 3, dtype=np.float32).reshape(1, 1, 2))

    out = F.interpolate(inp, scale_factor=2.0, mode="LINEAR")
    out2 = F.interpolate(inp, 4, mode="LINEAR")

    assertTensorClose(
        out.numpy(), np.array([[[1.0, 1.25, 1.75, 2.0]]], dtype=np.float32)
    )
    assertTensorClose(
        out2.numpy(), np.array([[[1.0, 1.25, 1.75, 2.0]]], dtype=np.float32)
    )


def test_many_batch_interpolate():
    inp = tensor(np.arange(1, 9, dtype=np.float32).reshape(2, 1, 2, 2))

    out = F.interpolate(inp, [4, 4])
    out2 = F.interpolate(inp, scale_factor=2.0)

    assertTensorClose(out.numpy(), out2.numpy())


def test_assign_corner_interpolate():
    inp = tensor(np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2))

    out = F.interpolate(inp, [4, 4], align_corners=True)
    out2 = F.interpolate(inp, scale_factor=2.0, align_corners=True)

    assertTensorClose(out.numpy(), out2.numpy())


def test_error_shape_linear_interpolate():
    inp = tensor(np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2))

    with pytest.raises(ValueError):
        F.interpolate(inp, scale_factor=2.0, mode="LINEAR")


def test_inappropriate_scale_linear_interpolate():
    inp = tensor(np.arange(1, 3, dtype=np.float32).reshape(1, 1, 2))

    with pytest.raises(ValueError):
        F.interpolate(inp, scale_factor=[2.0, 3.0], mode="LINEAR")
