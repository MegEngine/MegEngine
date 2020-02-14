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


def test_wrong_dtype():
    with pytest.raises(TypeError):
        mge.tensor(np.zeros((5, 5), dtype=np.float64))

    with pytest.raises(TypeError):
        mge.Parameter(np.zeros((5, 5), dtype=np.int64))


def test_tensor_routine():
    mge.tensor(np.zeros((1, 2), dtype=np.int32))

    mge.tensor([1])

    mge.tensor(1.5)
