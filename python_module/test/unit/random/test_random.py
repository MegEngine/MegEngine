# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

import megengine as mge
import megengine.functional as F
import megengine.jit as jit
import megengine.module as M
import megengine.random as R


def test_random_static_diff_result():
    @jit.trace(symbolic=True)
    def graph_a():
        return R.uniform(5) + R.gaussian(5)

    @jit.trace(symbolic=True)
    def graph_b():
        return R.uniform(5) + R.gaussian(5)

    a = graph_a()
    b = graph_b()
    assert np.any(a.numpy() != b.numpy())


def test_random_static_same_result():
    @jit.trace(symbolic=True)
    def graph_a():
        R.manual_seed(731)
        return R.uniform(5) + R.gaussian(5)

    @jit.trace(symbolic=True)
    def graph_b():
        R.manual_seed(731)
        return R.uniform(5) + R.gaussian(5)

    a = graph_a()
    b = graph_b()
    assert np.all(a.numpy() == b.numpy())


def test_random_dynamic_diff_result():
    a = R.uniform(5) + R.gaussian(5)
    b = R.uniform(5) + R.gaussian(5)
    assert np.any(a.numpy() != b.numpy())


def test_random_dynamic_same_result():
    R.manual_seed(0)
    a = R.uniform(5) + R.gaussian(5)
    R.manual_seed(0)
    b = R.uniform(5) + R.gaussian(5)
    assert np.all(a.numpy() == b.numpy())


def test_dropout_dynamic_diff_result():
    x = mge.ones(10)
    a = F.dropout(x, 0.5)
    b = F.dropout(x, 0.5)
    assert np.any(a.numpy() != b.numpy())


def test_dropout_dynamic_same_result():
    x = mge.ones(10)
    R.manual_seed(0)
    a = F.dropout(x, 0.5)
    R.manual_seed(0)
    b = F.dropout(x, 0.5)
    assert np.all(a.numpy() == b.numpy())


def test_M_dropout_static_diff_result():
    m = M.Dropout(0.5)

    @jit.trace(symbolic=True)
    def graph_a(x):
        return m(x)

    @jit.trace(symbolic=True)
    def graph_b(x):
        return m(x)

    x = np.ones(10, dtype="float32")
    a = graph_a(x)
    a = a.numpy().copy()
    b = graph_b(x)
    c = graph_a(x)
    assert np.any(a != b.numpy())
    assert np.any(a != c.numpy())


def test_M_dropout_static_same_result():
    m = M.Dropout(0.5)

    @jit.trace(symbolic=True)
    def graph_a(x):
        return m(x)

    @jit.trace(symbolic=True)
    def graph_b(x):
        return m(x)

    x = np.ones(10, dtype="float32")
    R.manual_seed(0)
    a = graph_a(x)
    a = a.numpy().copy()
    R.manual_seed(0)
    b = graph_b(x)
    R.manual_seed(0)  # useless
    c = graph_a(x)
    assert np.all(a == b.numpy())
    assert np.any(a != c.numpy())
