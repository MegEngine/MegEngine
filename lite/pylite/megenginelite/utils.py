# -*- coding: utf-8 -*-
import threading
import warnings

import numpy as np

from .base import *
from .struct import *
from .tensor import *


class TensorBatchCollector:
    """
    A tensor utils is used to collect many single batch tensor to a multi batch
    size tensor, when the multi batch size tensor collect finish, the result
    tensor can be get and send to the model input for forwarding.

    when collect single batch tensor, the single batch tensor is no need in the
    same device_type and device_id with the result tensor, however the dtype must
    match and the shape must match except the highest dimension.

    Args:
        shape: the multi batch size tensor shape, After collection, the result
            tensor shape.
        dtype(LiteDataType): the datatype of the single batch tensor and the
            result tensor, default value is LiteDataType.LITE_INT8.
        device_type(LiteDeviceType): the target device type the result tensor
            will allocate, default value is LiteDeviceType.LITE_CUDA.
        device_id: the device id the result tensor will allocate, default 0.
        is_pinned_host: Whether the memory is pinned memory, refer to CUDA
            pinned memory, default False.
        tensor(LiteTensor): the result tensor, user can also create the multi
            batch size tensor and then create the TensorBatchColletor, if tensor is
            not None, all the member, such as shape, dtype, device_type,
            device_id, is_pinned_host will get from the tensor, if the tensor is
            None and the result tensor will create by the TensorBatchCollector,
            default is None.

    Note:
        when collect tensor, the single batch tensor or array shape must match the 
        result tensor shape except the batch size dimension (the highest dimension)

    Examples:

        .. code-block:: python

            import numpy as np
            batch_tensor = TensorBatchCollector([4, 8, 8])
            arr = np.ones([8, 8], "int8")
            for i in range(4):
                batch_tensor.collect(arr)
                arr += 1
            data = batch_tensor.to_numpy()
            assert data.shape[0] == 4
            assert data.shape[1] == 8
            assert data.shape[2] == 8
            for i in range(4):
                for j in range(64):
                    assert data[i][j // 8][j % 8] == i + 1

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
        self.dev_id = device_id
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
        """
        Collect a single batch through an array and store the array data to the
        specific batch_id.

        Args:
            array: an array maybe LiteTensor or numpy ndarray, the shape of
                array must match the result tensor shape except the highest
                dimension.
            batch_id: the batch id to store the array data to the result tensor,
                if the batch_id has already collected, a warning will generate.
        """
        # get the batch index
        with self._mutex:
            if batch_id in self._free_list:
                self._free_list.remove(batch_id)
            else:
                warnings.warn(
                    "batch {} has been collected, please call free before collected it again.".format(
                        batch_id
                    )
                )
        self._collect_with_id(array, batch_id)

    def _collect_with_id(self, array, batch_id):
        if isinstance(array, np.ndarray):
            shape = array.shape
            assert list(shape) == self.shape[1:]
            in_dtype = ctype_to_lite_dtypes[np.ctypeslib.as_ctypes_type(array.dtype)]
            assert in_dtype == self.dtype
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
            # get the subtensor
            subtensor = self._tensor.slice([batch_id], [batch_id + 1])
            subtensor.copy_from(array)

        return batch_id

    def collect(self, array):
        """
        Collect a single batch through an array and store the array data to an
        empty batch, the empty batch is the front batch id in free list.

        Args:
            array: an array maybe LiteTensor or numpy ndarray, the shape must
                match the result tensor shape except the highest dimension
        """
        with self._mutex:
            if len(self._free_list) == 0:
                warnings.warn(
                    "all batch has been collected, please call free before collect again."
                )
                return -1
            idx = self._free_list.pop(0)
        return self._collect_with_id(array, idx)

    def collect_by_ctypes(self, data, length):
        """
        Collect a single batch through an ctypes memory buffer and store the
        ctypes memory data to an empty batch, the empty batch is the front
        batch id in free list.

        Args:
            array: an array maybe LiteTensor or numpy ndarray, the shape must
                match the result tensor shape except the highest dimension
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
        """
        free the batch ids in the indexes, after the batch id is freed, it can
        be collected again without warning.

        Args:
            indexes: a list of to be freed batch id
        """
        with self._mutex:
            for i in indexes:
                if i in self._free_list:
                    warnings.warn(
                        "batch id {} has not collected before free it.".format(i)
                    )
                    self._free_list.remove(i)
            self._free_list.extend(indexes)

    def get(self):
        """
        After finish collection, get the result tensor
        """
        return self._tensor

    def to_numpy(self):
        """
        Convert the result tensor to a numpy ndarray
        """
        return self._tensor.to_numpy()
