import subprocess
import sys

import numpy as np
import pytest

import megengine as mge
import megengine.functional as F
from megengine.core import set_option
from megengine.core._imperative_rt.core2 import AsyncError


def test_basic():
    mge.config.async_level = 2
    assert mge.config.async_level == 2
    with pytest.raises(AssertionError):
        mge.config.async_level = 3


def test_level1_infer_value():
    mge.config.async_level = 1
    a = mge.tensor([[1, 2], [2, 3], [3, 4]], dtype="float32")
    b = mge.tensor([1, 1], dtype="float32")
    identity = mge.tensor(np.array([[1, 0], [0, 1]]), dtype="float32")
    # make DepType::VALUE unknown
    c = F.matmul(b, identity)
    with pytest.raises(RuntimeError):
        d = F.reshape(a, c)
    mge.config.async_level = 2


def test_level1_infer_shape_with_unknown():
    mge.config.async_level = 2
    a = mge.tensor([[1, 2, 2, 3]], dtype="float32")
    b = mge.tensor([1, 1], dtype="float32")
    multi2 = mge.tensor(np.array([[2, 0], [0, 2]]), dtype="float32")
    c = F.matmul(b, multi2)
    # make DepType::SHAPE unknown
    d = F.reshape(a, c)
    e = mge.tensor([[1, 2]], dtype="float32")
    mge.config.async_level = 1
    # test src no shape, throw in level1
    with pytest.raises(RuntimeError):
        f = F.reshape(d, b)
    with pytest.raises(RuntimeError):
        g = F.matmul(d, e)
    mge.config.async_level = 2


def test_host_compute_elemwise():
    a = mge.tensor([[1, 2], [2, 3], [3, 4]], dtype="float32")
    b = mge.tensor([1, 1], dtype="int32")
    # check DepType::VALUE is still known
    c = b * 2
    with pytest.raises(RuntimeError):
        d = F.reshape(a, c)


def test_drop_basic():
    set_option("enable_drop", True)
    # test xpu compute
    x = mge.tensor(np.ones((3, 3)), dtype=np.float32)
    y = mge.tensor(np.ones((3, 3)), dtype=np.float32)
    z = x + y
    z._drop()
    z.numpy()
    # test host value compute
    x = mge.tensor(np.ones((2, 2)), dtype=np.float32)
    y = mge.tensor(np.ones((2, 2)), dtype=np.float32)
    z = x + y
    z._drop()
    z.numpy()
    set_option("enable_drop", False)


def test_finalize():
    prog = """
import megengine
megengine.core.set_option("enable_host_compute", 0)
x = megengine.tensor(0)
y = x + 1
y.numpy()
megengine.core.set_option("enable_host_compute", 1)
"""
    subprocess.check_call([sys.executable, "-c", prog])


def test_regression_2870():
    x = F.zeros(1000)
    y = F.utils._simulate_error()
    with pytest.raises(RuntimeError):
        y.numpy()
    (x + x).numpy()


@pytest.mark.require_ngpu(1)
def test_async_error_check():
    src = mge.tensor([[1.0, 2.0]])
    index = mge.tensor([3])
    val = F.indexing_one_hot(src, index)
    with pytest.raises(RuntimeError):
        val.numpy()


# NOTE: DO NOT REMOVE THIS TEST
#   This is also a compatibility test for
#   mge.config.async_level = 0.
#   If you change the canonical API to set async level,
#   update the error message of AsyncError as well.
def test_async_error():
    orig_lvl = mge.config.async_level
    try:
        mge.config.async_level = 1
        x = F.utils._simulate_error()
        try:
            x.numpy()
        except AsyncError as e:
            assert isinstance(e.__cause__, RuntimeError)

        mge.config.async_level = 0
        with pytest.raises(RuntimeError):
            F.utils._simulate_error()
    finally:
        mge.config.async_level = orig_lvl
