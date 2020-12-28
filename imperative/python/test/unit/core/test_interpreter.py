import numpy as np
import pytest

import megengine as mge
import megengine.functional as F
from megengine.core._imperative_rt.core2 import (
    _set_drop_flag,
    _set_swap_flag,
    config_async_level,
    get_async_level,
)


def test_basic():
    config_async_level(2)
    assert get_async_level() == 2
    with pytest.raises(RuntimeError):
        config_async_level(3)


def test_level1_infer_value():
    config_async_level(1)
    a = mge.tensor([[1, 2], [2, 3], [3, 4]], dtype="float32")
    b = mge.tensor([1, 1], dtype="float32")
    identity = mge.tensor(np.array([[1, 0], [0, 1]]), dtype="float32")
    # make DepType::VALUE unknown
    c = F.matmul(b, identity)
    with pytest.raises(RuntimeError):
        d = F.reshape(a, c)
    config_async_level(2)


def test_level1_infer_shape_with_unknown():
    config_async_level(2)
    a = mge.tensor([[1, 2, 2, 3]], dtype="float32")
    b = mge.tensor([1, 1])
    multi2 = mge.tensor(np.array([[2, 0], [0, 2]]), dtype="float32")
    c = F.matmul(b, multi2)
    # make DepType::SHAPE unknown
    d = F.reshape(a, c)
    e = mge.tensor([[1, 2]], dtype="float32")
    config_async_level(1)
    # test src no shape, throw in level1
    with pytest.raises(RuntimeError):
        f = F.reshape(d, b)
    with pytest.raises(RuntimeError):
        g = F.matmul(d, e)
    config_async_level(2)


def test_host_compute_elemwise():
    a = mge.tensor([[1, 2], [2, 3], [3, 4]], dtype="float32")
    b = mge.tensor([1, 1], dtype="int32")
    # check DepType::VALUE is still known
    c = b * 2
    with pytest.raises(RuntimeError):
        d = F.reshape(a, c)


def test_swap_drop_basic():
    _set_swap_flag(True)
    _set_drop_flag(True)
    # test xpu compute
    x = mge.tensor(np.ones((3, 3)), dtype=np.float32)
    y = mge.tensor(np.ones((3, 3)), dtype=np.float32)
    z = x + y
    x._swap_out()
    z._drop()
    z.numpy()
    # test host value compute
    x = mge.tensor(np.ones((2, 2)), dtype=np.float32)
    y = mge.tensor(np.ones((2, 2)), dtype=np.float32)
    z = x + y
    x._swap_out()
    z._drop()
    z.numpy()
    _set_swap_flag(False)
    _set_drop_flag(False)
