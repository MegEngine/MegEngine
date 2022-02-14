import gc

import numpy as np
import pytest

import megengine as mge
import megengine.functional as F
from megengine._multistream import record_event, wait_event


class MemStat:
    def __init__(self, *args):
        for d in args:
            mge.Tensor([], device=d)
        gc.collect()
        mge._full_sync()
        self.baseline = {d: mge.device.get_allocated_memory(d) for d in args}
        for d in args:
            mge.device.reset_max_memory_stats(d)

    def get_max(self, device):
        return mge.device.get_max_allocated_memory(device) - self.baseline[device]


@pytest.mark.require_ngpu(1)
def test_mem_stats():
    memstat = MemStat("xpux:0", "xpux:1")

    F.arange(1024, device="xpux:0")

    mge._full_sync()
    assert 4096 <= memstat.get_max("xpux:0") == memstat.get_max("xpux:1") <= 4096 + 128


@pytest.mark.require_ngpu(1)
def test_borrow():
    memstat = MemStat("xpux:0", "xpux:1")

    x_np = np.random.randint(2 ** 30, size=(1 * 1024 * 1024,), dtype="int32")
    unit = x_np.size * 4
    x0 = mge.Tensor(x_np, device="xpux:0")
    x1 = x0.to("xpux:1", _borrow=True)
    y = -x1
    np.testing.assert_equal(-x_np, y.numpy())

    mge._full_sync()
    assert memstat.get_max("xpux:0") / unit < 2.1


@pytest.mark.require_ngpu(1)
def test_stream_mem():
    memstat = MemStat("xpux:0", "xpux:1")

    x_np = np.random.randint(2 ** 10, size=(1 * 1024 * 1024,), dtype="int32")
    unit = x_np.size * 4
    x0 = mge.Tensor(x_np, device="xpux:0")

    results = []
    events = []
    for i in range(100):
        if len(events) >= 2:
            wait_event("xpux:0", events[-2])
        x0 = x0 + 1
        results.append(x0.to("xpux:1", _borrow=True).sum())
        events.append(record_event("xpux:1"))
        del events[:-2]

    y_np = x_np.sum()
    for i, y in enumerate(results):
        y_np += x_np.size
        assert y_np == y.numpy()

    mge._full_sync()
    assert memstat.get_max("xpux:0") / unit < 2.1
