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

import megengine as mge
from megengine.test import assertTensorClose


def test_zeros():
    assertTensorClose(
        mge.zeros((2, 2), dtype=np.int32).numpy(), np.zeros((2, 2), dtype=np.int32)
    )

    assertTensorClose(
        mge.zeros(mge.tensor([2, 2], dtype=np.int32), dtype=np.int32).numpy(),
        np.zeros((2, 2), dtype=np.int32),
    )


def test_ones():
    assertTensorClose(
        mge.ones((2, 2), dtype=np.int32).numpy(), np.ones((2, 2), dtype=np.int32)
    )

    assertTensorClose(
        mge.ones(mge.tensor([2, 2], dtype=np.int32), dtype=np.int32).numpy(),
        np.ones((2, 2), dtype=np.int32),
    )
