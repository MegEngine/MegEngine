import multiprocessing as mp
import platform

import numpy as np
import pytest

import megengine as mge
import megengine.distributed as dist
import megengine.quantization.observer as ob
from megengine.distributed.helper import get_device_count_by_fork


def test_min_max_observer():
    x = np.random.rand(3, 3, 3, 3).astype("float32")
    np_min, np_max = x.min(), x.max()
    x = mge.tensor(x)
    m = ob.MinMaxObserver()
    m(x)
    assert m.min_val == np_min and m.max_val == np_max


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
        m = ob.SyncMinMaxObserver()
        y = mge.tensor(x[rank * 3 : (rank + 1) * 3])
        m(y)
        assert m.min_val == np_min and m.max_val == np_max

    worker()
