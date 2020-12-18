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

import megengine
from megengine.core._imperative_rt.core2 import apply
from megengine.tensor import Tensor


def elemwise(*args, mode):
    from megengine.core.ops.builtin import Elemwise

    return apply(Elemwise(mode), *args)


def test_basic_interface():
    cf = megengine.core._imperative_rt.OperatorNodeConfig()
    cf.name = "megengine.core"
    cf.dtype = "float32"
    cf.comp_node_arr = ["xpux"]
    print(cf.name)
    print(cf.dtype)
    print(cf.comp_node_arr)
    print(cf.comp_node)
    cf.comp_node_arr = ["xpux", "xpux:1"]
    with pytest.raises(ValueError):
        cf.comp_node


def test_opr_attr():
    from megengine.core.ops.builtin import Elemwise

    assert Elemwise(Elemwise.Mode.ADD) == Elemwise(Elemwise.Mode.ADD)


def test_simple_arith():
    from megengine.core.ops.builtin import Elemwise

    x = np.random.rand(10).astype("float32")
    xx = Tensor(x)
    (yy,) = elemwise(xx, xx, mode=Elemwise.Mode.MUL)
    np.testing.assert_allclose(x * x, yy.numpy())
    del xx
    del yy


def test_tensor_on_device():
    device = megengine.core._imperative_rt.CompNode("cpu0:1")
    x = np.random.rand(10).astype("float32")
    xx = megengine.core._imperative_rt.put(x, device=device)
    assert str(megengine.core._imperative_rt.get_device(xx)) == "cpu0:1"
    np.testing.assert_equal(x, megengine.core._imperative_rt.get_value(xx))
    megengine.core._imperative_rt.delete(xx)


def test_raw_tensor():
    from megengine.core.ops.builtin import Elemwise

    x = np.random.rand(10).astype("float32")
    xx = Tensor(x)
    (yy,) = apply(Elemwise(Elemwise.Mode.MUL), xx, xx)
    np.testing.assert_allclose(x * x, yy.numpy())
    (yy,) = apply(Elemwise(Elemwise.Mode.MUL), xx, xx)
    np.testing.assert_allclose(x * x, yy.numpy())
