# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import pickle

import numpy as np

from megengine.core.tensor.dtype import bfloat16
from megengine.tensor import Tensor


def test_define():
    np.testing.assert_allclose(
        np.array([0.5, 0.13425, 3.4687, -1.34976, -9.34673, 0.0], dtype=bfloat16),
        np.array([0.5, 0.133789, 3.46875, -1.351562, -9.375, 0.0], dtype=np.float32),
        atol=1e-6,
    )


def test_cast():
    dtypes = [np.int8, np.int16, np.int32, np.float32, np.float64]
    fp32_values = [0.34985, 10.943, -0.5, -19.3, 21.49673]
    bf16_values = [0.349609, 10.9375, -0.5, -19.25, 21.5]
    int_values = [34, 10, -5, -19, 21]
    for dtype in dtypes:
        np.testing.assert_allclose(
            np.array(fp32_values, dtype=bfloat16).astype(dtype),
            np.array(bf16_values, dtype=dtype),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.array(int_values, dtype=dtype),
            np.array(int_values, dtype=bfloat16).astype(dtype),
            atol=1e-6,
        )


def test_shared_nd():
    data = np.array([-3.4, 1.394683, 2.323497, -7.439948, -5.2397], dtype=bfloat16)
    snd = Tensor(data, dtype=bfloat16, device="xpux")
    assert snd.numpy().dtype == bfloat16
    np.testing.assert_allclose(
        snd.numpy(), [-3.40625, 1.398438, 2.328125, -7.4375, -5.25], atol=1e-6
    )

    data = np.array([-9.34964, -8.342, 9.4385, 0.18746, 1.48], dtype=bfloat16)
    snd = Tensor(data, dtype=bfloat16, device="xpux")
    np.testing.assert_allclose(
        snd.numpy(), [-9.375, -8.3125, 9.4375, 0.1875, 1.476562], atol=1e-6
    )


def test_pickle():
    x = np.ascontiguousarray(np.random.rand(8192), dtype=bfloat16)
    pkl = pickle.dumps(x, pickle.HIGHEST_PROTOCOL)
    y = pickle.loads(pkl)
    assert x.dtype is y.dtype
    np.testing.assert_allclose(x.astype(np.float32), y.astype(np.float32), atol=1e-6)
