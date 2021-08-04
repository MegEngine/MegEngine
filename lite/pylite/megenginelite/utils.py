# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import threading

import numpy as np

from .base import *
from .struct import *
from .tensor import *


class TensorBatchCollector:
    """
    this is a tensor utils to collect subtensor in batch continuous
    """

    def __init__(
        self,
        shape,
        dtype=LiteDataType.LITE_INT8,
        device_type=LiteDeviceType.LITE_CUDA,
        device_id=0,
        is_pinned_host=False,
        tensor=None,
    ):
        self._mutex = threading.Lock()
        self.dev_type = device_type
        self.is_pinned_host = is_pinned_host
        self.dev_id = 0
        self.shape = shape
        self.dtype = LiteLayout(dtype=dtype).data_type
        self._free_list = list(range(self.shape[0]))

        if tensor is not None:
            assert (
                tensor.layout.shapes[0 : tensor.layout.ndim] == shape
            ), "The tensor set to TensorBatchCollector is not right."
            self._tensor = tensor
            self.dtype = tensor.layout.data_type
            self.device_type = tensor.device_type
            self.device_id = tensor.device_type
        else:
            self._tensor = LiteTensor(
                LiteLayout(shape, dtype), device_type, device_id, is_pinned_host
            )

    def collect_id(self, array, batch_id):
        if isinstance(array, np.ndarray):
            shape = array.shape
            assert list(shape) == self.shape[1:]
            in_dtype = ctype_to_lite_dtypes[np.ctypeslib.as_ctypes_type(array.dtype)]
            assert in_dtype == self.dtype
            # get the batch index
            with self._mutex:
                if batch_id in self._free_list:
                    self._free_list.remove(batch_id)
            # get the subtensor
            subtensor = self._tensor.slice([batch_id], [batch_id + 1])
            if subtensor.device_type == LiteDeviceType.LITE_CPU:
                subtensor.set_data_by_copy(array)
            else:
                pinned_tensor = LiteTensor(
                    subtensor.layout, self.dev_type, self.dev_id, True
                )
                pinned_tensor.set_data_by_share(array)
                subtensor.copy_from(pinned_tensor)
        else:
            assert isinstance(array, LiteTensor)
            ndim = array.layout.ndim
            shape = list(array.layout.shapes)[0:ndim]
            assert list(shape) == self.shape[1:]
            in_dtype = array.layout.data_type
            assert in_dtype == self.dtype
            # get the batch index
            with self._mutex:
                if batch_id in self._free_list:
                    self._free_list.remove(batch_id)
            # get the subtensor
            subtensor = self._tensor.slice([batch_id], [batch_id + 1])
            subtensor.copy_from(array)

        return batch_id

    def collect(self, array):
        with self._mutex:
            if len(self._free_list) == 0:
                return -1
            idx = self._free_list.pop(0)
        return self.collect_id(array, idx)

    def collect_by_ctypes(self, data, length):
        """
        collect with ctypes data input
        """
        with self._mutex:
            if len(self._free_list) == 0:
                return -1
            idx = self._free_list.pop(0)
        # get the subtensor
        subtensor = self._tensor.slice([idx], [idx + 1])
        if subtensor.device_type == LiteDeviceType.LITE_CPU:
            subtensor.set_data_by_copy(data, length)
        else:
            pinned_tensor = LiteTensor(
                subtensor.layout, self.dev_type, self.dev_id, True
            )
            pinned_tensor.set_data_by_share(data, length)
            subtensor.copy_from(pinned_tensor)

    def free(self, indexes):
        with self._mutex:
            self._free_list.extend(indexes)

    def get(self):
        return self._tensor

    def to_numpy(self):
        return self._tensor.to_numpy()
