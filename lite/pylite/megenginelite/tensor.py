# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from ctypes import *

import numpy as np

from .base import _Ctensor, _lib, _LiteCObjBase
from .struct import LiteDataType, LiteDeviceType, LiteIOType, Structure

MAX_DIM = 7

_lite_type_to_nptypes = {
    LiteDataType.LITE_INT: np.int32,
    LiteDataType.LITE_FLOAT: np.float32,
    LiteDataType.LITE_UINT8: np.uint8,
    LiteDataType.LITE_INT8: np.int8,
    LiteDataType.LITE_INT16: np.int16,
    LiteDataType.LITE_UINT16: np.uint16,
    LiteDataType.LITE_HALF: np.float16,
}

_nptype_to_lite_type = {val: key for key, val in _lite_type_to_nptypes.items()}

_str_nptypes_to_lite_nptypes = {
    np.dtype("int32"): LiteDataType.LITE_INT,
    np.dtype("float32"): LiteDataType.LITE_FLOAT,
    np.dtype("uint8"): LiteDataType.LITE_UINT8,
    np.dtype("int8"): LiteDataType.LITE_INT8,
    np.dtype("int16"): LiteDataType.LITE_INT16,
    np.dtype("uint16"): LiteDataType.LITE_UINT16,
    np.dtype("float16"): LiteDataType.LITE_HALF,
}

ctype_to_lite_dtypes = {
    c_int: LiteDataType.LITE_INT,
    c_uint: LiteDataType.LITE_INT,
    c_float: LiteDataType.LITE_FLOAT,
    c_ubyte: LiteDataType.LITE_UINT8,
    c_byte: LiteDataType.LITE_INT8,
    c_short: LiteDataType.LITE_INT16,
    c_ushort: LiteDataType.LITE_UINT16,
}


class LiteLayout(Structure):
    """
    the simple layout description
    """

    _fields_ = [
        ("shapes", c_size_t * MAX_DIM),
        ("ndim", c_size_t),
        ("data_type", c_int),
    ]

    def __init__(self, shape=None, dtype=None):
        if shape:
            shape = list(shape)
            assert len(shape) <= MAX_DIM, "Layout max dim is 7."
            self.shapes = (c_size_t * MAX_DIM)(*shape)
            self.ndim = len(shape)
        else:
            self.shapes = (c_size_t * MAX_DIM)()
            self.ndim = 0
        if not dtype:
            self.data_type = LiteDataType.LITE_FLOAT
        elif isinstance(dtype, LiteDataType):
            self.data_type = dtype
        elif type(dtype) == str:
            self.data_type = _str_nptypes_to_lite_nptypes[np.dtype(dtype)]
        elif isinstance(dtype, np.dtype):
            ctype = np.ctypeslib.as_ctypes_type(dtype)
            self.data_type = ctype_to_lite_dtypes[ctype]
        elif isinstance(dtype, type):
            self.data_type = _nptype_to_lite_type[dtype]
        else:
            raise RuntimeError("unkonw data type")

    def __repr__(self):
        data = {
            "shapes": list(self.shapes)[0 : self.ndim],
            "ndim": self.ndim,
            "data_type": _lite_type_to_nptypes[LiteDataType(self.data_type)],
        }
        return data.__repr__()


class _LiteTensorDesc(Structure):
    """
    warpper of the MegEngine Tensor

    :is_pinned_host: when set, the storage memory of the tensor is pinned memory,
    this is used to Optimize the H2D or D2H memory copy, if the device or layout
    is not set, when copy form other device(CUDA) tensor, this tensor
    will be automatically set to pinned tensor
    """

    _fields_ = [
        ("is_pinned_host", c_int),
        ("layout", LiteLayout),
        ("device_type", c_int),
        ("device_id", c_int),
    ]

    def __init__(self):
        self.layout = LiteLayout()
        self.device_type = LiteDeviceType.LITE_CPU
        self.is_pinned_host = False
        self.device_id = 0

    def __repr__(self):
        data = {
            "is_pinned_host": self.is_pinned_host,
            "layout": LiteLayout(self.layout),
            "device_type": LiteDeviceType(self.device_type.value),
            "device_id": self.device_id,
        }
        return data.__repr__()


class _TensorAPI(_LiteCObjBase):
    """
    get the api from the lib
    """

    _api_ = [
        ("LITE_make_tensor", [_LiteTensorDesc, POINTER(_Ctensor)]),
        ("LITE_set_tensor_layout", [_Ctensor, LiteLayout]),
        ("LITE_reset_tensor_memory", [_Ctensor, c_void_p, c_size_t]),
        ("LITE_reset_tensor", [_Ctensor, LiteLayout, c_void_p]),
        ("LITE_tensor_reshape", [_Ctensor, POINTER(c_int), c_int]),
        (
            "LITE_tensor_slice",
            [
                _Ctensor,
                POINTER(c_size_t),
                POINTER(c_size_t),
                POINTER(c_size_t),
                c_size_t,
                POINTER(_Ctensor),
            ],
        ),
        (
            "LITE_tensor_concat",
            [POINTER(_Ctensor), c_int, c_int, c_int, c_int, POINTER(_Ctensor),],
        ),
        ("LITE_tensor_fill_zero", [_Ctensor]),
        ("LITE_tensor_copy", [_Ctensor, _Ctensor]),
        ("LITE_tensor_share_memory_with", [_Ctensor, _Ctensor]),
        ("LITE_get_tensor_memory", [_Ctensor, POINTER(c_void_p)]),
        ("LITE_get_tensor_total_size_in_byte", [_Ctensor, POINTER(c_size_t)]),
        ("LITE_get_tensor_layout", [_Ctensor, POINTER(LiteLayout)]),
        ("LITE_get_tensor_device_type", [_Ctensor, POINTER(c_int)]),
        ("LITE_get_tensor_device_id", [_Ctensor, POINTER(c_int)]),
        ("LITE_destroy_tensor", [_Ctensor]),
        ("LITE_is_pinned_host", [_Ctensor, POINTER(c_int)]),
    ]


class LiteTensor(object):
    """
    the tensor to hold a block of data
    """

    _api = _TensorAPI()._lib

    def __init__(
        self,
        layout=None,
        device_type=LiteDeviceType.LITE_CPU,
        device_id=0,
        is_pinned_host=False,
    ):
        """
        create a Tensor with layout, device, is_pinned_host param
        """
        self._tensor = _Ctensor()
        if layout:
            self._layout = layout
        else:
            self._layout = LiteLayout()
        self._device_type = device_type
        self._device_id = device_id
        self._is_pinned_host = is_pinned_host

        tensor_desc = _LiteTensorDesc()
        tensor_desc.layout = self._layout
        tensor_desc.device_type = device_type
        tensor_desc.device_id = device_id
        tensor_desc.is_pinned_host = is_pinned_host
        self._api.LITE_make_tensor(tensor_desc, byref(self._tensor))

    def __del__(self):
        self._api.LITE_destroy_tensor(self._tensor)

    def fill_zero(self):
        """
        fill the buffer memory with zero
        """
        self._api.LITE_tensor_fill_zero(self._tensor)
        self.update()

    def share_memory_with(self, src_tensor):
        """
        share the same memory with the src_tensor, the self memory will be freed
        """
        assert isinstance(src_tensor, LiteTensor)
        self._api.LITE_tensor_share_memory_with(self._tensor, src_tensor._tensor)
        self.update()

    @property
    def layout(self):
        self._api.LITE_get_tensor_layout(self._tensor, byref(self._layout))
        return self._layout

    @layout.setter
    def layout(self, layout):
        assert isinstance(layout, LiteLayout)
        self._layout = layout
        self._api.LITE_set_tensor_layout(self._tensor, layout)

    @property
    def is_pinned_host(self):
        """
        whether the tensor is pinned tensor
        """
        pinned = c_int()
        self._api.LITE_is_pinned_host(self._tensor, byref(pinned))
        self._is_pinned_host = pinned
        return bool(self._is_pinned_host)

    @property
    def device_type(self):
        """
        get device of the tensor
        """
        device_type = c_int()
        self._api.LITE_get_tensor_device_type(self._tensor, byref(device_type))
        self._device_type = device_type
        return LiteDeviceType(device_type.value)

    @property
    def device_id(self):
        """
        get device id of the tensor
        """
        device_id = c_int()
        self._api.LITE_get_tensor_device_id(self._tensor, byref(device_id))
        self._device_id = device_id.value
        return device_id.value

    @property
    def is_continue(self):
        """
        whether the tensor memory is continue
        """
        is_continue = c_int()
        self._api.LITE_is_memory_continue(self._tensor, byref(is_continue))
        return bool(is_continue.value)

    @property
    def nbytes(self):
        """
        get the length of the meomry in byte
        """
        self.update()
        length = c_size_t()
        self._api.LITE_get_tensor_total_size_in_byte(self._tensor, byref(length))
        return length.value

    def update(self):
        """
        update the member from C, this will auto used after slice, share
        """
        pinned = c_int()
        self._api.LITE_is_pinned_host(self._tensor, byref(pinned))
        self._is_pinned_host = pinned
        device_type = c_int()
        self._api.LITE_get_tensor_device_type(self._tensor, byref(device_type))
        self._device_type = device_type
        self._api.LITE_get_tensor_layout(self._tensor, byref(self._layout))

    def copy_from(self, src_tensor):
        """
        copy memory form the src_tensor
        """
        assert isinstance(src_tensor, LiteTensor)
        self._api.LITE_tensor_copy(self._tensor, src_tensor._tensor)
        self.update()

    def reshape(self, shape):
        """
        reshape the tensor with data not change, only change the shape
        :param shape: int arrary of dst_shape
        """
        shape = list(shape)
        length = len(shape)
        c_shape = (c_int * length)(*shape)
        self._api.LITE_tensor_reshape(self._tensor, c_shape, length)
        self.update()

    def slice(self, start, end, step=None):
        """
        slice the tensor with gaven start, end, step
        :param start: silce begin index of each dim
        :param end: silce end index of each dim
        :param step: silce step of each dim
        """
        start = list(start)
        end = list(end)
        length = len(start)
        assert length == len(end), "slice with different length of start and end."
        if step:
            assert length == len(step), "slice with different length of start and step."
            step = list(step)
        else:
            step = [1 for i in range(length)]
        c_start = (c_size_t * length)(*start)
        c_end = (c_size_t * length)(*end)
        c_step = (c_size_t * length)(*step)
        slice_tensor = LiteTensor()
        self._api.LITE_tensor_slice(
            self._tensor, c_start, c_end, c_step, length, byref(slice_tensor._tensor)
        )
        slice_tensor.update()
        return slice_tensor

    def get_ctypes_memory(self):
        """
        get the memory of the tensor, return c_void_p of the tensor memory
        """
        self.update()
        mem = c_void_p()
        self._api.LITE_get_tensor_memory(self._tensor, byref(mem))
        return mem

    def set_data_by_share(self, data, length=0, layout=None):
        """
        share the data to the tensor
        param data: the data will shared to the tensor, it should be a
        numpy.ndarray or ctypes data
        """
        self.update()
        if isinstance(data, np.ndarray):
            assert (
                self.is_continue
            ), "set_data_by_share can only apply in continue tensor."
            assert (
                self.is_pinned_host or self.device_type == LiteDeviceType.LITE_CPU
            ), "set_data_by_share can only apply in cpu tensor or pinned tensor."

            np_type = _lite_type_to_nptypes[LiteDataType(self._layout.data_type)]
            c_type = np.ctypeslib.as_ctypes_type(np_type)

            if self.nbytes != data.nbytes:
                self.layout = LiteLayout(data.shape, ctype_to_lite_dtypes[c_type])

            self._shared_data = data
            data = data.ctypes.data_as(POINTER(c_type))

        if layout is not None:
            self.layout = layout
        else:
            assert length == 0 or length == self.nbytes, "the data length is not match."
        self._api.LITE_reset_tensor_memory(self._tensor, data, self.nbytes)

    def set_data_by_copy(self, data, data_length=0, layout=None):
        """
        copy the data to the tensor
        param data: the data to copy to tensor, it should be list,
        numpy.ndarraya or ctypes with length
        """
        self.update()
        if layout is not None:
            self.layout = layout

        assert self.is_continue, "set_data_by_copy can only apply in continue tensor."
        assert (
            self.is_pinned_host or self.device_type == LiteDeviceType.LITE_CPU
        ), "set_data_by_copy can only apply in cpu tensor or pinned tensor."

        np_type = _lite_type_to_nptypes[LiteDataType(self._layout.data_type)]
        c_type = np.ctypeslib.as_ctypes_type(np_type)

        tensor_memory = c_void_p()

        if type(data) == list:
            length = len(data)
            self._api.LITE_get_tensor_memory(self._tensor, byref(tensor_memory))
            tensor_length = self.nbytes
            assert (
                length * sizeof(c_type) <= tensor_length
            ), "the length of input data to set to the tensor is too large."
            arr = (c_type * length)(*data)
            memmove(tensor_memory, arr, sizeof(c_type) * length)

        elif type(data) == np.ndarray:
            if self.nbytes != data.nbytes:
                self.layout = LiteLayout(data.shape, data.dtype)
            arr = data.ctypes.data_as(POINTER(c_type))
            self._api.LITE_get_tensor_memory(self._tensor, byref(tensor_memory))
            assert self.nbytes == data.nbytes
            memmove(tensor_memory, arr, self.nbytes)
        else:
            assert (
                data_length == self.nbytes or layout is not None
            ), "when input data is ctypes, the length of input data or layout must set"
            self._api.LITE_get_tensor_memory(self._tensor, byref(tensor_memory))
            memmove(tensor_memory, data, data_length)

    def to_numpy(self):
        """
        get the buffer of the tensor
        """
        self.update()
        if self.nbytes <= 0:
            return np.array([])
        if self.is_continue and (
            self.is_pinned_host or self.device_type == LiteDeviceType.LITE_CPU
        ):
            ptr = c_void_p()
            self._api.LITE_get_tensor_memory(self._tensor, byref(ptr))

            np_type = _lite_type_to_nptypes[LiteDataType(self._layout.data_type)]
            shape = [self._layout.shapes[i] for i in range(self._layout.ndim)]
            np_arr = np.zeros(shape, np_type)
            if np_arr.nbytes:
                memmove(np_arr.ctypes.data_as(c_void_p), ptr, np_arr.nbytes)
            return np_arr
        else:
            tmp_tensor = LiteTensor(self.layout)
            tmp_tensor.copy_from(self)
            return tmp_tensor.to_numpy()

    def __repr__(self):
        self.update()
        data = {
            "layout": self._layout,
            "device_type": LiteDeviceType(self._device_type.value),
            "device_id": int(self.device_id),
            "is_pinned_host": bool(self._is_pinned_host),
        }
        return data.__repr__()


def LiteTensorConcat(
    tensors, dim, device_type=LiteDeviceType.LITE_DEVICE_DEFAULT, device_id=-1
):
    """
    concat tensor in input dim to one tensor
    dim : the dim to act concat
    device_type: the result tensor device type
    device_id: the result tensor device id
    """
    api = _TensorAPI()._lib
    length = len(tensors)
    c_tensors = [t._tensor for t in tensors]
    c_tensors = (_Ctensor * length)(*c_tensors)
    result_tensor = LiteTensor()
    api.LITE_tensor_concat(
        cast(byref(c_tensors), POINTER(c_void_p)),
        length,
        dim,
        device_type,
        device_id,
        byref(result_tensor._tensor),
    )
    result_tensor.update()
    return result_tensor
