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
import megengine.functional as F
from megengine.autodiff import GradManager


def test_basic():
    x = mge.tensor([1.0, 3.0, 5.0]).reshape(1, 3)
    w = mge.tensor([2.0, 4.0, 6.0]).reshape(3, 1)
    b = mge.tensor(-1.0)

    gm = GradManager().attach([w, b])
    gm.record()

    p = F.matmul(x, w)
    y = p + b

    gm.backward(y)
    gm.release()  # is not necessary
    np.testing.assert_equal(w.grad.numpy(), [[1], [3], [5]])
    np.testing.assert_equal(b.grad.numpy(), [1])

    gm.clear_grad()
    with gm:
        p = F.matmul(x, w)
        y = p + b
        gm.backward(y)

    np.testing.assert_equal(w.grad.numpy(), [[1], [3], [5]])
    np.testing.assert_equal(b.grad.numpy(), [1])


def test_attach_in_with_block():
    a = mge.Parameter([1.0])
    gm = GradManager()
    with gm:
        b = a * 3
        gm.attach(b)
        c = b + 1
        gm.backward(c)
    assert int(b.grad.numpy()) == 1
