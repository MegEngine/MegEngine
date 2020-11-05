# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import os
import platform
import time

import numpy as np
import pytest

from megengine.data.collator import Collator
from megengine.data.dataloader import DataLoader
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
        dataloader = DataLoader(dataset, num_workers=2, divide=True)
    with pytest.raises(ValueError):
        dataloader = DataLoader(dataset, num_workers=-1)
    with pytest.raises(ValueError):
        dataloader = DataLoader(dataset, timeout=-1)
    with pytest.raises(ValueError):
        dataloader = DataLoader(dataset, num_workers=0, divide=True)

    dataloader = DataLoader(dataset)
    assert isinstance(dataloader.sampler, SequentialSampler)
    assert isinstance(dataloader.transform, PseudoTransform)
    assert isinstance(dataloader.collator, Collator)

    dataloader = DataLoader(
        dataset, sampler=RandomSampler(dataset, batch_size=6, drop_last=False)
    )
    assert len(dataloader) == 17
    dataloader = DataLoader(
        dataset, sampler=RandomSampler(dataset, batch_size=6, drop_last=True)
    )
    assert len(dataloader) == 16


class MyStream(StreamDataset):
    def __init__(self, number, batch=False, error=False):
        self.number = number
        self.batch = batch
        self.error = error

    def __iter__(self):
        for cnt in range(self.number):
            if self.batch:
                data = np.random.randint(0, 256, (2, 32, 32, 3), dtype="uint8")
                yield (True, (data, [cnt, cnt - self.number]))
            else:
                data = np.random.randint(0, 256, (32, 32, 3), dtype="uint8")
                if self.error:
                    yield (data, cnt)
                else:
                    yield (False, (data, cnt))
        raise StopIteration


@pytest.mark.parametrize("batch", [True, False])
@pytest.mark.parametrize("num_workers", [0, 2])
def test_stream_dataloader(batch, num_workers):
    dataset = MyStream(100, batch)
    sampler = StreamSampler(batch_size=4)
    dataloader = DataLoader(
        dataset,
        sampler,
        Compose([Normalize(mean=(103, 116, 123), std=(57, 57, 58)), ToMode("CHW")]),
        num_workers=num_workers,
    )

    check_set = set()

    for step, data in enumerate(dataloader):
        if step == 10:
            break
        assert data[0].shape == (4, 3, 32, 32)
        assert data[1].shape == (4,)
        for i in data[1]:
            assert i not in check_set
            check_set.add(i)


def test_stream_dataloader_error():
    dataset = MyStream(100, error=True)
    sampler = StreamSampler(batch_size=4)
    dataloader = DataLoader(dataset, sampler)
    with pytest.raises(AssertionError, match=r".*tuple.*"):
        data_iter = iter(dataloader)
        next(data_iter)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_stream_dataloader_timeout(num_workers):
    dataset = MyStream(100, False)
    sampler = StreamSampler(batch_size=4)

    class TimeoutTransform(Transform):
        def __init__(self):
            pass

        def apply(self, input):
            time.sleep(10)
            return input

    dataloader = DataLoader(
        dataset, sampler, TimeoutTransform(), num_workers=num_workers, timeout=5
    )
    with pytest.raises(RuntimeError, match=r".*timeout.*"):
        data_iter = iter(dataloader)
        next(data_iter)


def test_dataloader_serial():
    dataset = init_dataset()
    dataloader = DataLoader(
        dataset, sampler=RandomSampler(dataset, batch_size=4, drop_last=False)
    )
    for (data, label) in dataloader:
        assert data.shape == (4, 1, 32, 32)
        assert label.shape == (4,)


def test_dataloader_parallel():
    # set max shared memory to 100M
    os.environ["MGE_PLASMA_MEMORY"] = "100000000"

    dataset = init_dataset()
    dataloader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset, batch_size=4, drop_last=False),
        num_workers=2,
        divide=False,
    )
    for (data, label) in dataloader:
        assert data.shape == (4, 1, 32, 32)
        assert label.shape == (4,)

    dataloader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset, batch_size=4, drop_last=False),
        num_workers=2,
        divide=True,
    )
    for (data, label) in dataloader:
        assert data.shape == (4, 1, 32, 32)
        assert label.shape == (4,)


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="dataloader do not support parallel on windows",
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
    )
    with pytest.raises(RuntimeError, match=r".*timeout.*"):
        data_iter = iter(dataloader)
        batch_data = next(data_iter)


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="dataloader do not support parallel on windows",
)
def test_dataloader_parallel_worker_exception():
    dataset = init_dataset()

    class FakeErrorTransform(Transform):
        def __init__(self):
            pass

        def apply(self, input):
            y = x + 1
            return input

    dataloader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset, batch_size=4, drop_last=False),
        transform=FakeErrorTransform(),
        num_workers=2,
    )
    with pytest.raises(RuntimeError, match=r"worker.*died"):
        data_iter = iter(dataloader)
        batch_data = next(data_iter)


def _multi_instances_parallel_dataloader_worker():
    dataset = init_dataset()

    for divide_flag in [True, False]:
        train_dataloader = DataLoader(
            dataset,
            sampler=RandomSampler(dataset, batch_size=4, drop_last=False),
            num_workers=2,
            divide=divide_flag,
        )
        val_dataloader = DataLoader(
            dataset,
            sampler=RandomSampler(dataset, batch_size=10, drop_last=False),
            num_workers=2,
            divide=divide_flag,
        )
        for idx, (data, label) in enumerate(train_dataloader):
            assert data.shape == (4, 1, 32, 32)
            assert label.shape == (4,)
            if idx % 5 == 0:
                for val_data, val_label in val_dataloader:
                    assert val_data.shape == (10, 1, 32, 32)
                    assert val_label.shape == (10,)


def test_dataloader_parallel_multi_instances():
    # set max shared memory to 100M
    os.environ["MGE_PLASMA_MEMORY"] = "100000000"

    _multi_instances_parallel_dataloader_worker()


def test_dataloader_parallel_multi_instances_multiprocessing():
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
