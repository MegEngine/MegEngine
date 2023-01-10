# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import pytest

from megengine.data.dataset import ArrayDataset, ConcatDataset, Dataset, StreamDataset


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


def test_concat_dataset():
    size1 = (10,)
    size2 = (20,)
    data_shape1 = (3, 256, 256)
    data_shape2 = (2, 128, 128)
    label_shape1 = (1,)
    label_shape2 = (2,)
    data1 = np.random.randint(0, 255, size1 + data_shape1)
    data2 = np.random.randint(0, 255, size2 + data_shape2)
    label1 = np.random.randint(0, 9, size1 + label_shape1)
    label2 = np.random.randint(0, 9, size2 + label_shape2)
    dataset1 = ArrayDataset(data1, label1)
    dataset2 = ArrayDataset(data2, label2)
    dataset = ConcatDataset([dataset1, dataset2])
    assert dataset[15][0].shape == data_shape2
    assert dataset[15][1].shape == label_shape2
