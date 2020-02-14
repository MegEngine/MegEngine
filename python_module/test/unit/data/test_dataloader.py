# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import os
import time

import numpy as np
import pytest

from megengine.data.collator import Collator
from megengine.data.dataloader import DataLoader
from megengine.data.dataset import ArrayDataset
from megengine.data.sampler import RandomSampler, SequentialSampler
from megengine.data.transform import PseudoTransform, Transform


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
