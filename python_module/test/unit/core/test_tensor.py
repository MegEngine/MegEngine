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
import megengine._internal as mgb


def test_wrong_dtype():
    with pytest.raises(TypeError):
        mge.tensor(np.zeros((5, 5), dtype=np.float64))

    with pytest.raises(TypeError):
        mge.Parameter(np.zeros((5, 5), dtype=np.int64))


def test_tensor_routine():
    mge.tensor(np.zeros((1, 2), dtype=np.int32))

    mge.tensor([1])

    mge.tensor(1.5)


def test_tensor_set_dtype():
    def check_dtype_value(tensor, dtype_scale, value):
        if mgb.dtype.is_quantize(tensor.dtype):
            if np.abs(mgb.dtype.get_scale(tensor.dtype) - dtype_scale) > 1e-5:
                raise AssertionError(
                    "compare scale failed expect {} got {}".format(
                        dtype_scale, mgb.dtype.get_scale(tensor.dtype)
                    )
                )
            if np.abs(tensor.numpy()[0][0] - value) > 1e-5:
                raise AssertionError(
                    "compare value failed expect {} got {}".format(
                        tensor.numpy()[0][0], value
                    )
                )

    t = mge.Parameter(np.ones((3, 4), dtype="float32"))
    t.set_dtype(mgb.dtype.qint8(0.1))
    check_dtype_value(t, 0.1, 10)

    t = mge.Parameter(np.ones((3, 4), dtype=mgb.dtype.qint8(1)))
    t.set_dtype(mgb.dtype.qint8(0.3))
    check_dtype_value(t, 0.3, 3)

    t = mge.Buffer(np.ones((3, 4), dtype="float32"))
    t.set_dtype(mgb.dtype.qint8(0.1))
    check_dtype_value(t, 0.1, 10)

    t = mge.Buffer(np.ones((3, 4), dtype=mgb.dtype.qint8(1)))
    t.set_dtype(mgb.dtype.qint8(0.3))
    check_dtype_value(t, 0.3, 3)

    t = mge.Buffer(np.ones((3, 4), dtype="float32"))
    s = t + 1
    s.set_dtype(mgb.dtype.qint8(0.2))
    check_dtype_value(s, 0.2, 10)

    t.set_dtype(mgb.dtype.qint8(0.3))
    s = t + 1
    s.set_dtype(mgb.dtype.qint8(0.1))
    check_dtype_value(s, 0.1, 18)
    s.set_dtype("float32")
    check_dtype_value(s, 0, 1.8)
