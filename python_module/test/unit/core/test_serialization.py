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

from megengine.core import Buffer, Parameter, tensor
from megengine.test import assertTensorClose


def test_tensor_serialization():
    def tensor_eq(a, b):
        assert a.dtype == b.dtype
        assert a.device == b.device
        assert a.requires_grad == b.requires_grad
        assertTensorClose(a, b)

    with TemporaryFile() as f:
        data = np.random.randint(low=0, high=7, size=[233])
        a = tensor(data, device="xpux", dtype=np.int32)
        pickle.dump(a, f)
        f.seek(0)
        b = pickle.load(f)
        tensor_eq(a, b)

    with TemporaryFile() as f:
        a = Parameter(np.random.random(size=(233, 2)).astype(np.float32))
        pickle.dump(a, f)
        f.seek(0)
        b = pickle.load(f)
        assert isinstance(b, Parameter)
        tensor_eq(a, b)

    with TemporaryFile() as f:
        a = Buffer(np.random.random(size=(2, 233)).astype(np.float32))
        pickle.dump(a, f)
        f.seek(0)
        b = pickle.load(f)
        assert isinstance(b, Buffer)
        tensor_eq(a, b)
