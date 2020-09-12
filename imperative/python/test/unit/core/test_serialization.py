# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import pickle
from tempfile import TemporaryFile

import numpy as np

import megengine as mge
from megengine import Parameter, Tensor


def test_tensor_serialization():
    def tensor_eq(a, b):
        assert a.dtype == b.dtype
        assert a.device == b.device
        np.testing.assert_equal(a.numpy(), b.numpy())

    with TemporaryFile() as f:
        data = np.random.randint(low=0, high=7, size=[233])
        a = Tensor(data, device="xpux", dtype=np.int32)
        pickle.dump(a, f)
        f.seek(0)
        b = pickle.load(f)
        np.testing.assert_equal(a.numpy(), b.numpy())

    with TemporaryFile() as f:
        a = Parameter(np.random.random(size=(233, 2)).astype(np.float32))
        pickle.dump(a, f)
        f.seek(0)
        b = pickle.load(f)
        assert isinstance(b, Parameter)
        np.testing.assert_equal(a.numpy(), b.numpy())

    with TemporaryFile() as f:
        a = Tensor(np.random.random(size=(2, 233)).astype(np.float32))
        pickle.dump(a, f)
        f.seek(0)
        b = pickle.load(f)
        assert type(b) is Tensor
        np.testing.assert_equal(a.numpy(), b.numpy())

    with TemporaryFile() as f:
        a = Tensor(np.random.random(size=(2, 233)).astype(np.float32))
        mge.save(a, f)
        f.seek(0)
        b = mge.load(f, map_location="cpux")
        assert type(b) is Tensor
        assert "cpu" in str(b.device)
        np.testing.assert_equal(a.numpy(), b.numpy())

    with TemporaryFile() as f:
        if mge.is_cuda_available():
            device_org = mge.get_default_device()
            mge.set_default_device("gpu0")
            a = Tensor(np.random.random(size=(2, 233)).astype(np.float32))
            mge.save(a, f)
            f.seek(0)
            mge.set_default_device("cpux")
            b = mge.load(f, map_location={"gpu0": "cpu0"})
            assert type(b) is Tensor
            assert "cpu0" in str(b.device)
            np.testing.assert_equal(a.numpy(), b.numpy())
            mge.set_default_device(device_org)
