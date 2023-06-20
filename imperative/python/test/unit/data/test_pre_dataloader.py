# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import gc
import math
import multiprocessing
import os
import platform
import time

import numpy as np
import pytest

from megengine.data.collator import Collator
from megengine.data.dataloader import DataLoader, get_worker_info
from megengine.data.dataset import ArrayDataset, StreamDataset
from megengine.data.sampler import RandomSampler, SequentialSampler, StreamSampler
from megengine.data.transform import (
    Compose,
    Normalize,
    PseudoTransform,
    ToMode,
    Transform,
)


def init_dataset():
    sample_num = 100
    rand_data = np.random.randint(0, 255, size=(sample_num, 1, 32, 32), dtype=np.uint8)
    label = np.random.randint(0, 10, size=(sample_num,), dtype=int)
    dataset = ArrayDataset(rand_data, label)
    return dataset


def test_dataloader_init():
    dataset = init_dataset()
    with pytest.raises(ValueError):
        dataloader = DataLoader(dataset, num_workers=-1)
    with pytest.raises(ValueError):
        dataloader = DataLoader(dataset, timeout=-1)

    dataloader = DataLoader(dataset, preload=True)
    assert isinstance(dataloader.sampler, SequentialSampler)
    assert isinstance(dataloader.transform, PseudoTransform)
    assert isinstance(dataloader.collator, Collator)

    dataloader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset, batch_size=6, drop_last=False),
        preload=True,
    )
    assert len(dataloader) == 17
    dataloader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset, batch_size=6, drop_last=True),
        preload=True,
    )
    assert len(dataloader) == 16


class MyStream(StreamDataset):
    def __init__(self, number, block=False):
        self.number = number
        self.block = block

    def __iter__(self):
        for cnt in range(self.number):
            if self.block:
                for _ in range(10):
                    time.sleep(1)
            data = np.random.randint(0, 256, (2, 2, 3), dtype="uint8")
            yield (data, cnt)
        raise StopIteration


@pytest.mark.skipif(
    np.__version__ >= "1.20.0",
    reason="pyarrow is incompatible with numpy vserion 1.20.0",
)
@pytest.mark.parametrize("num_workers", [0, 2])
def test_stream_dataloader(num_workers):
    dataset = MyStream(100)
    sampler = StreamSampler(batch_size=4)
    dataloader = DataLoader(
        dataset,
        sampler,
        Compose([Normalize(mean=(103, 116, 123), std=(57, 57, 58)), ToMode("CHW")]),
        num_workers=num_workers,
        preload=True,
    )

    check_set = set()

    for step, data in enumerate(dataloader):
        if step == 10:
            break
        assert data[0]._tuple_shape == (4, 3, 2, 2)
        assert data[1]._tuple_shape == (4,)
        for i in data[1]:
            assert i not in check_set
            check_set.add(i)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_stream_dataloader_timeout(num_workers):
    dataset = MyStream(100, block=True)
    sampler = StreamSampler(batch_size=4)

    dataloader = DataLoader(
        dataset, sampler, num_workers=num_workers, timeout=2, preload=True
    )
    with pytest.raises(RuntimeError, match=r".*timeout.*"):
        data_iter = iter(dataloader)
        next(data_iter)


def test_dataloader_serial():
    dataset = init_dataset()
    dataloader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset, batch_size=4, drop_last=False),
        preload=True,
    )
    for (data, label) in dataloader:
        assert data._tuple_shape == (4, 1, 32, 32)
        assert label._tuple_shape == (4,)


@pytest.mark.skipif(
    np.__version__ >= "1.20.0",
    reason="pyarrow is incompatible with numpy vserion 1.20.0",
)
def test_dataloader_parallel():
    # set max shared memory to 100M
    os.environ["MGE_PLASMA_MEMORY"] = "100000000"

    dataset = init_dataset()
    dataloader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset, batch_size=4, drop_last=False),
        num_workers=2,
        preload=True,
    )
    for (data, label) in dataloader:
        assert data._tuple_shape == (4, 1, 32, 32)
        assert label._tuple_shape == (4,)


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="dataloader do not support parallel on windows",
)
@pytest.mark.skipif(
    multiprocessing.get_start_method() != "fork",
    reason="the runtime error is only raised when fork",
)
def test_dataloader_parallel_timeout():
    dataset = init_dataset()

    class TimeoutTransform(Transform):
        def __init__(self):
            pass

        def apply(self, input):
            time.sleep(10)
            return input

    dataloader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset, batch_size=4, drop_last=False),
        transform=TimeoutTransform(),
        num_workers=2,
        timeout=2,
        preload=True,
    )
    with pytest.raises(RuntimeError, match=r".*timeout.*"):
        data_iter = iter(dataloader)
        batch_data = next(data_iter)


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="dataloader do not support parallel on windows",
)
@pytest.mark.skipif(
    multiprocessing.get_start_method() != "fork",
    reason="the runtime error is only raised when fork",
)
def test_dataloader_parallel_worker_exception():
    dataset = init_dataset()

    class FakeErrorTransform(Transform):
        def __init__(self):
            pass

        def apply(self, input):
            raise RuntimeError("test raise error")
            return input

    dataloader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset, batch_size=4, drop_last=False),
        transform=FakeErrorTransform(),
        num_workers=2,
        preload=True,
    )
    with pytest.raises(
        RuntimeError, match=r"Caught RuntimeError in DataLoader worker process"
    ):
        data_iter = iter(dataloader)
        batch_data = next(data_iter)


def _multi_instances_parallel_dataloader_worker():
    dataset = init_dataset()

    train_dataloader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset, batch_size=4, drop_last=False),
        num_workers=2,
        preload=True,
    )
    val_dataloader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset, batch_size=10, drop_last=False),
        num_workers=2,
        preload=True,
    )
    for idx, (data, label) in enumerate(train_dataloader):
        assert data._tuple_shape == (4, 1, 32, 32)
        assert label._tuple_shape == (4,)
        if idx % 5 == 0:
            for val_data, val_label in val_dataloader:
                assert val_data._tuple_shape == (10, 1, 32, 32)
                assert val_label._tuple_shape == (10,)


@pytest.mark.skipif(
    np.__version__ >= "1.20.0",
    reason="pyarrow is incompatible with numpy vserion 1.20.0",
)
def test_dataloader_parallel_multi_instances():
    # set max shared memory to 100M
    os.environ["MGE_PLASMA_MEMORY"] = "100000000"

    _multi_instances_parallel_dataloader_worker()


@pytest.mark.isolated_distributed
def test_dataloader_parallel_multi_instances_multiprocessing():
    gc.collect()
    # set max shared memory to 100M
    os.environ["MGE_PLASMA_MEMORY"] = "100000000"

    import multiprocessing as mp

    # mp.set_start_method("spawn")
    processes = []
    for i in range(4):
        p = mp.Process(target=_multi_instances_parallel_dataloader_worker)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        assert p.exitcode == 0
