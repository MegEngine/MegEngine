# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import copy
import os
import sys

import numpy as np
import pytest

from megengine.data.dataset import ArrayDataset
from megengine.data.sampler import RandomSampler, ReplacementSampler, SequentialSampler


def test_sequential_sampler():
    indices = list(range(100))
    sampler = SequentialSampler(ArrayDataset(indices))
    assert indices == list(each[0] for each in sampler)


def test_RandomSampler():
    indices = list(range(20))
    indices_copy = copy.deepcopy(indices)
    sampler = RandomSampler(ArrayDataset(indices_copy))
    sample_indices = sampler
    assert indices != list(each[0] for each in sample_indices)
    assert indices == sorted(list(each[0] for each in sample_indices))


def test_random_sampler_seed():
    seed = [0, 1]
    indices = list(range(20))
    indices_copy1 = copy.deepcopy(indices)
    indices_copy2 = copy.deepcopy(indices)
    indices_copy3 = copy.deepcopy(indices)
    sampler1 = RandomSampler(ArrayDataset(indices_copy1), seed=seed[0])
    sampler2 = RandomSampler(ArrayDataset(indices_copy2), seed=seed[0])
    sampler3 = RandomSampler(ArrayDataset(indices_copy3), seed=seed[1])
    assert indices != list(each[0] for each in sampler1)
    assert indices != list(each[0] for each in sampler2)
    assert indices != list(each[0] for each in sampler3)
    assert indices == sorted(list(each[0] for each in sampler1))
    assert indices == sorted(list(each[0] for each in sampler2))
    assert indices == sorted(list(each[0] for each in sampler3))
    assert list(each[0] for each in sampler1) == list(each[0] for each in sampler2)
    assert list(each[0] for each in sampler1) != list(each[0] for each in sampler3)


def test_ReplacementSampler():
    num_samples = 30
    indices = list(range(20))
    weights = list(range(20))
    sampler = ReplacementSampler(
        ArrayDataset(indices), num_samples=num_samples, weights=weights
    )
    assert len(list(each[0] for each in sampler)) == num_samples


def test_sampler_drop_last_false():
    batch_size = 5
    drop_last = False
    indices = list(range(24))
    sampler = SequentialSampler(
        ArrayDataset(indices), batch_size=batch_size, drop_last=drop_last
    )
    assert len([each for each in sampler]) == len(sampler)


def test_sampler_drop_last_true():
    batch_size = 5
    drop_last = True
    indices = list(range(24))
    sampler = SequentialSampler(
        ArrayDataset(indices), batch_size=batch_size, drop_last=drop_last
    )
    assert len([each for each in sampler]) == len(sampler)
