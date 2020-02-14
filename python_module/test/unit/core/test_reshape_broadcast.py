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
from megengine.test import assertTensorClose


def test_reshape_tuple():
    inp = tensor(np.arange(1, 17, dtype=np.int32).reshape(4, 4))
    out = tensor(np.arange(100, 116, dtype=np.int32).reshape(1, 16))
    out = out.reshape(inp.shape)

    assertTensorClose(out.numpy(), np.arange(100, 116, dtype=np.int32).reshape(4, 4))


def test_reshape_asterisk():
    inp = tensor(np.arange(1, 17, dtype=np.int32).reshape(4, 4))
    out = tensor(np.arange(100, 116, dtype=np.int32).reshape(1, 16))
    out = out.reshape(*inp.shape)

    assertTensorClose(out.numpy(), np.arange(100, 116, dtype=np.int32).reshape(4, 4))


def test_reshape_shapeof():
    inp = tensor(np.arange(1, 17, dtype=np.int32).reshape(4, 4))
    out = tensor(np.arange(100, 116, dtype=np.int32).reshape(1, 16))
    out = out.reshape(inp.shapeof())

    assertTensorClose(out.numpy(), np.arange(100, 116, dtype=np.int32).reshape(4, 4))


def test_reshape_tensor():
    out = tensor(np.arange(100, 116, dtype=np.int32).reshape(1, 16))
    out = out.reshape(tensor([4, 4]))

    assertTensorClose(out.numpy(), np.arange(100, 116, dtype=np.int32).reshape(4, 4))


def test_reshape_tensor_fused():
    out = tensor(np.arange(100, 116, dtype=np.int32).reshape(1, 16))
    out = out.reshape(tensor([4, 4]), 1)

    assertTensorClose(out.numpy(), np.arange(100, 116, dtype=np.int32).reshape(4, 4, 1))


def test_reshape_fused():
    out = tensor(np.arange(100, 116, dtype=np.int32).reshape(1, 16))
    out = out.reshape(tensor(2), 2, tensor(4), 1)

    assertTensorClose(
        out.numpy(), np.arange(100, 116, dtype=np.int32).reshape(2, 2, 4, 1)
    )


def test_reshape_wrong_tuple():
    out = tensor(np.arange(100, 116, dtype=np.int32).reshape(1, 16))
    with pytest.raises(ValueError):
        out = out.reshape((2, 2), 4)


def test_reshape_wrong_tuple2():
    out = tensor(np.arange(100, 116, dtype=np.int32).reshape(1, 16))
    with pytest.raises(AssertionError):
        out = out.reshape(4, (2, 2))


def test_broadcast_tuple():
    inp = tensor(np.arange(1, 17, dtype=np.int32).reshape(4, 4))
    out = tensor(np.arange(100, 104, dtype=np.int32).reshape(1, 4))

    out = out.broadcast(inp.shape)

    tmp = np.array([[100, 101, 102, 103]], dtype=np.int32)
    out2 = np.repeat(tmp, 4, axis=0)

    assertTensorClose(out.numpy(), out2)


def test_broadcast_asterisk():
    inp = tensor(np.arange(1, 17, dtype=np.int32).reshape(4, 4))
    out = tensor(np.arange(100, 104, dtype=np.int32).reshape(1, 4))

    out = out.broadcast(*inp.shape)

    tmp = np.array([[100, 101, 102, 103]], dtype=np.int32)
    out2 = np.repeat(tmp, 4, axis=0)

    assertTensorClose(out.numpy(), out2)


def test_broadcast_shapeof():
    inp = tensor(np.arange(1, 17, dtype=np.int32).reshape(4, 4))
    out = tensor(np.arange(100, 104, dtype=np.int32).reshape(1, 4))

    out = out.broadcast(inp.shapeof())

    tmp = np.array([[100, 101, 102, 103]], dtype=np.int32)
    out2 = np.repeat(tmp, 4, axis=0)

    assertTensorClose(out.numpy(), out2)
