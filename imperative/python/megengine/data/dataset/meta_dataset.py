# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from abc import ABC, abstractmethod
from typing import Tuple


class Dataset(ABC):
    r"""
    An abstract class for all datasets.
    __getitem__ and __len__ method are aditionally needed.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass


class StreamDataset(Dataset):
    r"""
    An abstract class for stream data.
    __iter__ method is aditionally needed.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    def __getitem__(self):
        raise AssertionError("can not get item from StreamDataset by index")

    def __len__(self):
        raise AssertionError("StreamDataset does not have length")


class ArrayDataset(Dataset):
    def __init__(self, *arrays):
        r"""
        ArrayDataset is a dataset for numpy array data, one or more numpy arrays
         are needed to initiate the dataset. And the dimensions represented sample number
         are expected to be the same.
        """
        super().__init__()
        if not all(len(arrays[0]) == len(array) for array in arrays):
            raise ValueError("lengths of input arrays are inconsistent")
        self.arrays = arrays

    def __getitem__(self, index: int) -> Tuple:
        return tuple(array[index] for array in self.arrays)

    def __len__(self) -> int:
        return len(self.arrays[0])
