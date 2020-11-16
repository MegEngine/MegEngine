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
    x = np.random.rand(6, 3, 3, 3).astype("float32")
    np_min, np_max = x.min(), x.max()
    world_size = 2
    port = dist.get_free_ports(1)[0]
    server = dist.Server(port)

    def worker(rank, slc):
        dist.init_process_group("localhost", port, world_size, rank, rank)
        m = ob.SyncMinMaxObserver()
        y = mge.tensor(x[slc])
        m(y)
        assert m.min_val == np_min and m.max_val == np_max

    procs = []
    for rank in range(world_size):
        slc = slice(rank * 3, (rank + 1) * 3)
        p = mp.Process(target=worker, args=(rank, slc,), daemon=True)
        p.start()
        procs.append(p)
    for p in procs:
        p.join(20)
        assert p.exitcode == 0
