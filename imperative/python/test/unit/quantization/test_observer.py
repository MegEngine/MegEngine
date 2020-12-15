import multiprocessing as mp
import platform

import numpy as np
import pytest

import megengine as mge
import megengine.distributed as dist
from megengine.distributed.helper import get_device_count_by_fork
from megengine.quantization.observer import (
    ExponentialMovingAverageObserver,
    MinMaxObserver,
    Observer,
    PassiveObserver,
    SyncExponentialMovingAverageObserver,
    SyncMinMaxObserver,
)


def test_observer():
    with pytest.raises(TypeError):
        Observer("qint8")


def test_min_max_observer():
    x = np.random.rand(3, 3, 3, 3).astype("float32")
    np_min, np_max = x.min(), x.max()
    x = mge.tensor(x)
    m = MinMaxObserver()
    m(x)
    np.testing.assert_allclose(m.min_val.numpy(), np_min)
    np.testing.assert_allclose(m.max_val.numpy(), np_max)


def test_exponential_moving_average_observer():
    t = np.random.rand()
    x1 = np.random.rand(3, 3, 3, 3).astype("float32")
    x2 = np.random.rand(3, 3, 3, 3).astype("float32")
    expected_min = x1.min() * t + x2.min() * (1 - t)
    expected_max = x1.max() * t + x2.max() * (1 - t)
    m = ExponentialMovingAverageObserver(momentum=t)
    m(mge.tensor(x1, dtype=np.float32))
    m(mge.tensor(x2, dtype=np.float32))
    np.testing.assert_allclose(m.min_val.numpy(), expected_min)
    np.testing.assert_allclose(m.max_val.numpy(), expected_max)


def test_passive_observer():
    q_dict = {"scale": mge.tensor(1.0)}
    m = PassiveObserver(q_dict, "qint8")
    assert m.orig_scale == 1.0
    assert m.scale == 1.0
    m.scale = 2.0
    assert m.scale == 2.0
    assert m.get_qparams() == {"scale": mge.tensor(2.0)}


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="windows disable MGB_ENABLE_OPR_MM"
)
@pytest.mark.skipif(get_device_count_by_fork("gpu") < 2, reason="need more gpu device")
@pytest.mark.isolated_distributed
def test_sync_min_max_observer():
    word_size = get_device_count_by_fork("gpu")
    x = np.random.rand(3 * word_size, 3, 3, 3).astype("float32")
    np_min, np_max = x.min(), x.max()

    @dist.launcher
    def worker():
        rank = dist.get_rank()
        m = SyncMinMaxObserver()
        y = mge.tensor(x[rank * 3 : (rank + 1) * 3])
        m(y)
        assert m.min_val == np_min and m.max_val == np_max

    worker()


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="windows disable MGB_ENABLE_OPR_MM"
)
@pytest.mark.skipif(get_device_count_by_fork("gpu") < 2, reason="need more gpu device")
@pytest.mark.isolated_distributed
def test_sync_exponential_moving_average_observer():
    word_size = get_device_count_by_fork("gpu")
    t = np.random.rand()
    x1 = np.random.rand(3 * word_size, 3, 3, 3).astype("float32")
    x2 = np.random.rand(3 * word_size, 3, 3, 3).astype("float32")
    expected_min = x1.min() * t + x2.min() * (1 - t)
    expected_max = x1.max() * t + x2.max() * (1 - t)

    @dist.launcher
    def worker():
        rank = dist.get_rank()
        m = SyncExponentialMovingAverageObserver(momentum=t)
        y1 = mge.tensor(x1[rank * 3 : (rank + 1) * 3])
        y2 = mge.tensor(x2[rank * 3 : (rank + 1) * 3])
        m(y1)
        m(y2)
        np.testing.assert_allclose(m.min_val.numpy(), expected_min, atol=1e-6)
        np.testing.assert_allclose(m.max_val.numpy(), expected_max, atol=1e-6)

    worker()
