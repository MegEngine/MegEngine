# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

import megengine.functional as F
from megengine import tensor
from megengine.test import assertTensorClose


def test_abs():
    assertTensorClose(
        F.abs(tensor([-3.0, -4.0, -5.0])).numpy(),
        np.abs(np.array([-3.0, -4.0, -5.0], dtype=np.float32)),
    )

    assertTensorClose(F.abs(-3.0), np.abs(np.float32(-3.0)))


def test_multiply():
    assertTensorClose(
        F.multiply(-3.0, -4.0), np.multiply(np.float32(-3.0), np.float32(-4.0))
    )

    assertTensorClose(
        F.multiply(tensor([3.0, 4.0]), 4.0).numpy(),
        np.multiply(np.array([3.0, 4.0], dtype=np.float32), 4.0),
    )

    assertTensorClose(
        F.multiply(4.0, tensor([3.0, 4.0])).numpy(),
        np.multiply(4.0, np.array([3.0, 4.0], dtype=np.float32)),
    )

    assertTensorClose(
        F.multiply(tensor([3.0, 4.0]), tensor([3.0, 4.0])).numpy(),
        np.multiply(
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        ),
    )


def test_clamp():
    """Fix an issue when `lower` or `upper` is 0, it will be recognized as `False` and
    `F.clamp` will fall into wrong conditions unexpectedly.
    """
    x = np.linspace(-6, 6, dtype="float32")
    assertTensorClose(F.clamp(tensor(x) + 3, 0, 6).numpy(), np.clip(x + 3, 0, 6))
    assertTensorClose(F.clamp(tensor(x) - 3, -6, 0).numpy(), np.clip(x - 3, -6, 0))
