# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

from megengine import tensor
from megengine.core._imperative_rt import CompNode
from megengine.core._imperative_rt.ops import delete_rng_handle, new_rng_handle
from megengine.core.ops.builtin import GaussianRNG, UniformRNG
from megengine.core.tensor.core import apply


def test_gaussian_rng():
    shape = (
        8,
        9,
        11,
        12,
    )
    shape = tensor(shape, dtype="int32")
    op = GaussianRNG(1.0, 3.0)
    (output,) = apply(op, shape)
    assert np.fabs(output.numpy().mean() - 1.0) < 1e-1
    assert np.sqrt(output.numpy().var()) - 3.0 < 1e-1
    assert str(output.device) == str(CompNode("xpux"))

    cn = CompNode("xpu1")
    op = GaussianRNG(-1.0, 2.0, cn)
    (output,) = apply(op, shape)
    assert np.fabs(output.numpy().mean() - (-1.0)) < 1e-1
    assert np.sqrt(output.numpy().var()) - 2.0 < 1e-1
    assert str(output.device) == str(cn)

    cn = CompNode("xpu2")
    seed = 233333
    h = new_rng_handle(cn, seed)
    op = GaussianRNG(3.0, 1.0, h)
    (output,) = apply(op, shape)
    delete_rng_handle(h)
    assert np.fabs(output.numpy().mean() - 3.0) < 1e-1
    assert np.sqrt(output.numpy().var()) - 1.0 < 1e-1
    assert str(output.device) == str(cn)


def test_uniform_rng():
    shape = (
        8,
        9,
        11,
        12,
    )
    shape = tensor(shape, dtype="int32")
    op = UniformRNG()
    (output,) = apply(op, shape)
    assert np.fabs(output.numpy().mean() - 0.5) < 1e-1
    assert str(output.device) == str(CompNode("xpux"))

    cn = CompNode("xpu1")
    op = UniformRNG(cn)
    (output,) = apply(op, shape)
    assert np.fabs(output.numpy().mean() - 0.5) < 1e-1
    assert str(output.device) == str(cn)

    cn = CompNode("xpu2")
    seed = 233333
    h = new_rng_handle(cn, seed)
    op = UniformRNG(h)
    (output,) = apply(op, shape)
    delete_rng_handle(h)
    assert np.fabs(output.numpy().mean() - 0.5) < 1e-1
    assert str(output.device) == str(cn)
