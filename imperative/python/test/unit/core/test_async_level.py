import pytest

import megengine as mge
import megengine.functional as F
from megengine.core._imperative_rt.core2 import config_async_level, get_async_level


def test_basic():
    config_async_level(2)
    assert get_async_level() == 2
    with pytest.raises(RuntimeError):
        config_async_level(3)


def test_level1_infer_value():
    config_async_level(1)
    a = mge.tensor([[1, 2], [2, 3], [3, 4]], dtype="float32")
    b = mge.tensor([1, 1], dtype="float32")
    # make DepType::VALUE unknown
    c = b * 2
    with pytest.raises(RuntimeError):
        d = F.reshape(a, c)


def test_level1_infer_shape_with_unknown():
    config_async_level(2)
    a = mge.tensor([[1, 2, 2, 3]], dtype="float32")
    b = mge.tensor([1, 1])
    c = b * 2
    # make DepType::SHAPE unknown
    d = F.reshape(a, c)
    config_async_level(1)
    e = mge.tensor([[1, 2]], dtype="float32")
    with pytest.raises(RuntimeError):
        f = F.matmul(d, e)
