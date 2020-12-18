# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import pickle

import numpy as np
import pytest

from megengine.core.tensor.dtype import intb1, intb2, intb4
from megengine.tensor import Tensor


def bit_define_test(bit, low_bit_type):
    max_value = (1 << bit) - 1
    min_value = 1 - (1 << bit)

    a = np.array([i for i in range(min_value, max_value + 2, 2)], dtype=low_bit_type)

    for i in range(max_value + 1):
        np.testing.assert_equal(a[i], i * 2 - max_value)
        np.testing.assert_equal(str(a[i]), str(i * 2 - max_value))

    with pytest.raises(ValueError):
        np.arange(min_value, max_value, dtype=low_bit_type)

    with pytest.raises(ValueError):
        np.arange(min_value - 2, max_value + 4, 2, dtype=low_bit_type)

    np.testing.assert_allclose(
        np.arange(min_value, 12, 2, dtype=low_bit_type),
        (np.arange((13 - min_value) // 2, dtype=np.int8) % (max_value + 1)) * 2
        - max_value,
    )

    np.testing.assert_allclose(
        np.arange(max_value, max_value - 20, -2, dtype=low_bit_type),
        (np.arange(max_value, max_value - 10, -1, dtype=np.int8) % (max_value + 1)) * 2
        - max_value,
    )


def test_define():
    bit_define_test(1, intb1)
    bit_define_test(2, intb2)
    bit_define_test(4, intb4)


def _bit_cast_test(bit, low_bit_type):
    dtypes = [np.int8, np.int16, np.int32, np.float32, np.float64]

    max_value = (1 << bit) - 1
    min_value = 1 - (1 << bit)
    for dtype in dtypes:
        np.testing.assert_allclose(
            np.arange(min_value, max_value + 2, 2, dtype=low_bit_type).astype(dtype),
            np.arange(min_value, max_value + 2, 2, dtype=dtype),
        )

    with pytest.raises(ValueError):
        np.array([2, 1, -1], dtype=int).astype(low_bit_type)
    with pytest.raises(ValueError):
        np.array([min_value - 2, 1, max_value + 2], dtype=int).astype(low_bit_type)


def test_cast():
    _bit_cast_test(1, intb1)
    _bit_cast_test(2, intb2)
    _bit_cast_test(4, intb4)


def _shared_nd_test(bit, low_bit_type):
    max_value = (1 << bit) - 1
    min_value = 1 - (1 << bit)

    data = np.arange(min_value, max_value + 2, 2, dtype=low_bit_type)
    snd = Tensor(data, dtype=low_bit_type, device="xpux")
    np.testing.assert_allclose(snd.numpy(), range(min_value, max_value + 2, 2))

    data = np.arange(min_value, max_value + 2, 4, dtype=low_bit_type)
    snd = Tensor(data, dtype=low_bit_type, device="xpux")
    np.testing.assert_allclose(snd.numpy(), range(min_value, max_value + 2, 4))


def test_shared_nd():
    _shared_nd_test(1, intb1)
    _shared_nd_test(2, intb2)
    _shared_nd_test(4, intb4)


def test_pickle():
    x = np.ascontiguousarray(np.random.randint(2, size=8192) * 2 - 1, dtype=intb1)
    pkl = pickle.dumps(x, pickle.HIGHEST_PROTOCOL)
    y = pickle.loads(pkl)
    assert x.dtype is y.dtype
    np.testing.assert_allclose(x.astype(np.float32), y.astype(np.float32))
