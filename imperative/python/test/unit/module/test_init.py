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

from megengine import tensor
from megengine.module import Conv2d, Linear
from megengine.module.init import calculate_fan_in_and_fan_out, fill_


def test_fill_():
    x = tensor(np.zeros((2, 3, 4)), dtype=np.float32)
    fill_(x, 5.0)

    np.testing.assert_array_equal(
        x.numpy(), np.full(shape=(2, 3, 4), fill_value=5.0, dtype=np.float32)
    )


def test_calculate_fan_in_and_fan_out():
    l = Linear(in_features=3, out_features=8)
    fanin, fanout = calculate_fan_in_and_fan_out(l.weight)
    assert fanin == 3
    assert fanout == 8

    with pytest.raises(ValueError):
        calculate_fan_in_and_fan_out(l.bias)

    l = Conv2d(in_channels=2, out_channels=3, kernel_size=(5, 7))
    fanin, fanout = calculate_fan_in_and_fan_out(l.weight)
    assert fanin == 2 * 5 * 7
    assert fanout == 3 * 5 * 7
