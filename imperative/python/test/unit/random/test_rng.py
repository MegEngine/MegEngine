# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

import megengine
from megengine import tensor
from megengine.core._imperative_rt import CompNode
from megengine.core._imperative_rt.core2 import apply
from megengine.core._imperative_rt.ops import (
    delete_rng_handle,
    get_global_rng_seed,
    new_rng_handle,
)
from megengine.core.ops.builtin import GaussianRNG, UniformRNG
from megengine.random import RNG
from megengine.random.rng import _normal, _uniform


def test_gaussian_op():
    shape = (
        8,
        9,
        11,
        12,
    )
    shape = tensor(shape, dtype="int32")
    op = GaussianRNG(seed=get_global_rng_seed(), mean=1.0, std=3.0)
    (output,) = apply(op, shape)
    assert np.fabs(output.numpy().mean() - 1.0) < 1e-1
    assert np.sqrt(output.numpy().var()) - 3.0 < 1e-1
    assert str(output.device) == str(CompNode("xpux"))

    cn = CompNode("xpu2")
    seed = 233333
    h = new_rng_handle(cn, seed)
    op = GaussianRNG(seed=seed, mean=3.0, std=1.0, handle=h)
    (output,) = apply(op, shape)
    delete_rng_handle(h)
    assert np.fabs(output.numpy().mean() - 3.0) < 1e-1
    assert np.sqrt(output.numpy().var()) - 1.0 < 1e-1
    assert str(output.device) == str(cn)


def test_uniform_op():
    shape = (
        8,
        9,
        11,
        12,
    )
    shape = tensor(shape, dtype="int32")
    op = UniformRNG(seed=get_global_rng_seed())
    (output,) = apply(op, shape)
    assert np.fabs(output.numpy().mean() - 0.5) < 1e-1
    assert str(output.device) == str(CompNode("xpux"))

    cn = CompNode("xpu2")
    seed = 233333
    h = new_rng_handle(cn, seed)
    op = UniformRNG(seed=seed, handle=h)
    (output,) = apply(op, shape)
    delete_rng_handle(h)
    assert np.fabs(output.numpy().mean() - 0.5) < 1e-1
    assert str(output.device) == str(cn)


def test_UniformRNG():
    m1 = RNG(seed=111, device="xpu0")
    m2 = RNG(seed=111, device="xpu1")
    m3 = RNG(seed=222, device="xpu0")
    out1 = m1.uniform(size=(100,))
    out1_ = m1.uniform(size=(100,))
    out2 = m2.uniform(size=(100,))
    out3 = m3.uniform(size=(100,))

    np.testing.assert_equal(out1.numpy(), out2.numpy())
    assert out1.device == "xpu0" and out2.device == "xpu1"
    assert not (out1.numpy() == out3.numpy()).all()
    assert not (out1.numpy() == out1_.numpy()).all()

    low = -234
    high = 123
    out = m1.uniform(low=low, high=high, size=(20, 30, 40))
    out_shp = out.shape
    if isinstance(out_shp, tuple):
        assert out_shp == (20, 30, 40)
    else:
        assert all(out.shape.numpy() == np.array([20, 30, 40]))
    assert np.abs(out.mean().numpy() - ((low + high) / 2)) / (high - low) < 0.1


def test_NormalRNG():
    m1 = RNG(seed=111, device="xpu0")
    m2 = RNG(seed=111, device="xpu1")
    m3 = RNG(seed=222, device="xpu0")
    out1 = m1.normal(size=(100,))
    out1_ = m1.uniform(size=(100,))
    out2 = m2.normal(size=(100,))
    out3 = m3.normal(size=(100,))

    np.testing.assert_equal(out1.numpy(), out2.numpy())
    assert out1.device == "xpu0" and out2.device == "xpu1"
    assert not (out1.numpy() == out3.numpy()).all()
    assert not (out1.numpy() == out1_.numpy()).all()

    mean = -1
    std = 2
    out = m1.normal(mean=mean, std=std, size=(20, 30, 40))
    out_shp = out.shape
    if isinstance(out_shp, tuple):
        assert out_shp == (20, 30, 40)
    else:
        assert all(out.shape.numpy() == np.array([20, 30, 40]))
    assert np.abs(out.mean().numpy() - mean) / std < 0.1
    assert np.abs(np.std(out.numpy()) - std) < 0.1
