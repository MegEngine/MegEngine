# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import pytest

from megengine.data.dataset import ArrayDataset, Dataset, StreamDataset


def test_abstract_cls():
    with pytest.raises(TypeError):
        Dataset()
    with pytest.raises(TypeError):
        StreamDataset()


def test_array_dataset():
    size = (10,)
    data_shape = (3, 256, 256)
    label_shape = (1,)
    data = np.random.randint(0, 255, size + data_shape)
    label = np.random.randint(0, 9, size + label_shape)
    dataset = ArrayDataset(data, label)
    assert dataset[0][0].shape == data_shape
    assert dataset[0][1].shape == label_shape
    assert len(dataset) == size[0]


def test_array_dataset_dim_error():
    data = np.random.randint(0, 255, (10, 3, 256, 256))
    label = np.random.randint(0, 9, (1,))
    with pytest.raises(ValueError):
        ArrayDataset(data, label)
