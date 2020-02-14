# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

from megengine.data.transform import *

data_shape = (100, 100, 3)
label_shape = (4,)
ToMode_target_shape = (3, 100, 100)
CenterCrop_size = (90, 70)
CenterCrop_target_shape = CenterCrop_size + (3,)
RandomResizedCrop_size = (50, 50)
RandomResizedCrop_target_shape = RandomResizedCrop_size + (3,)


def generate_data():
    return [
        (
            (np.random.rand(*data_shape) * 255).astype(np.uint8),
            np.random.randint(10, size=label_shape),
        )
        for _ in range(*label_shape)
    ]


def test_ToMode():
    t = ToMode(mode="CHW")
    aug_data = t.apply_batch(generate_data())
    aug_data_shape = [(a.shape, b.shape) for a, b in aug_data]
    target_shape = [(ToMode_target_shape, label_shape)] * 4
    assert aug_data_shape == target_shape


def test_CenterCrop():
    t = CenterCrop(output_size=CenterCrop_size)
    aug_data = t.apply_batch(generate_data())
    aug_data_shape = [(a.shape, b.shape) for a, b in aug_data]
    target_shape = [(CenterCrop_target_shape, label_shape)] * 4
    assert aug_data_shape == target_shape


def test_ColorJitter():
    t = ColorJitter()
    aug_data = t.apply_batch(generate_data())
    aug_data_shape = [(a.shape, b.shape) for a, b in aug_data]
    target_shape = [(data_shape, label_shape)] * 4
    assert aug_data_shape == target_shape


def test_RandomHorizontalFlip():
    t = RandomHorizontalFlip(prob=1)
    aug_data = t.apply_batch(generate_data())
    aug_data_shape = [(a.shape, b.shape) for a, b in aug_data]
    target_shape = [(data_shape, label_shape)] * 4
    assert aug_data_shape == target_shape


def test_RandomVerticalFlip():
    t = RandomVerticalFlip(prob=1)
    aug_data = t.apply_batch(generate_data())
    aug_data_shape = [(a.shape, b.shape) for a, b in aug_data]
    target_shape = [(data_shape, label_shape)] * 4
    assert aug_data_shape == target_shape


def test_RandomResizedCrop():
    t = RandomResizedCrop(output_size=RandomResizedCrop_size)
    aug_data = t.apply_batch(generate_data())
    aug_data_shape = [(a.shape, b.shape) for a, b in aug_data]
    target_shape = [(RandomResizedCrop_target_shape, label_shape)] * 4
    assert aug_data_shape == target_shape


def test_Normalize():
    t = Normalize()
    aug_data = t.apply_batch(generate_data())
    aug_data_shape = [(a.shape, b.shape) for a, b in aug_data]
    target_shape = [(data_shape, label_shape)] * 4
    assert aug_data_shape == target_shape


def test_RandomCrop():
    t = RandomCrop((150, 120), padding_size=10, padding_value=[1, 2, 3])
    aug_data = t.apply_batch(generate_data())
    aug_data_shape = [(a.shape, b.shape) for a, b in aug_data]
    target_shape = [((150, 120, 3), label_shape)] * 4
    assert aug_data_shape == target_shape


def test_Compose():
    t = Compose(
        [
            CenterCrop(output_size=CenterCrop_size),
            RandomHorizontalFlip(prob=1),
            ToMode(mode="CHW"),
        ]
    )
    aug_data = t.apply_batch(generate_data())
    aug_data_shape = [(a.shape, b.shape) for a, b in aug_data]
    print(aug_data_shape)
    target_shape = [((3, 90, 70), label_shape)] * 4
    assert aug_data_shape == target_shape
