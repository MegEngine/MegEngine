# -*- coding: utf-8 -*-
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
    xx = megengine.tensor(x, device=device)
    assert str(xx.device) == "cpu0:1"
    np.testing.assert_equal(x, xx.numpy())
    del xx


def test_raw_tensor():
    from megengine.core.ops.builtin import Elemwise

    x = np.random.rand(10).astype("float32")
    xx = Tensor(x)
    (yy,) = apply(Elemwise(Elemwise.Mode.MUL), xx, xx)
    np.testing.assert_allclose(x * x, yy.numpy())
    (yy,) = apply(Elemwise(Elemwise.Mode.MUL), xx, xx)
    np.testing.assert_allclose(x * x, yy.numpy())


def test_opdef_path():
    from megengine.core.ops.builtin import Elemwise

    assert Elemwise.__module__ == "megengine.core._imperative_rt.ops"
    assert Elemwise.__name__ == "Elemwise"
    assert Elemwise.__qualname__ == "Elemwise"

    Mode = Elemwise.Mode
    assert Mode.__module__ == "megengine.core._imperative_rt.ops"
    assert Mode.__name__ == "Mode"
    assert Mode.__qualname__ == "Elemwise.Mode"
