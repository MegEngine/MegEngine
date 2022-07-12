# -*- coding: utf-8 -*-

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

_lite_dtypes_to_ctype = {
    LiteDataType.LITE_INT: c_int,
    LiteDataType.LITE_FLOAT: c_float,
    LiteDataType.LITE_UINT8: c_ubyte,
    LiteDataType.LITE_INT8: c_byte,
    LiteDataType.LITE_INT16: c_short,
    LiteDataType.LITE_UINT16: c_ushort,
}


class LiteLayout(Structure):
    """
    Description of layout using in Lite. A Lite layout will be totally defined
        by shape and data type.

    Args:
        shape: the shape of data.
        dtype: data type.

    Note:
        Dims of shape should be less than 8. The supported data type defines at
        LiteDataType

    Examples:

        .. code-block:: python

            import numpy as np
            layout = LiteLayout([1, 4, 8, 8], LiteDataType.LITE_FLOAT)
            assert(layout.shape()) == [1, 4, 8, 8]
            assert(layout.dtype()) == LiteDataType.LITE_FLOAT
    """

    _fields_ = [
        ("_shapes", c_size_t * MAX_DIM),
        ("ndim", c_size_t),
        ("data_type", c_int),
    ]

    def __init__(self, shape=None, dtype=None):
        if shape:
            shape = list(shape)
            assert len(shape) <= MAX_DIM, "Layout max dim is 7."
            self._shapes = (c_size_t * MAX_DIM)(*shape)
            self.ndim = len(shape)
        else:
            self._shapes = (c_size_t * MAX_DIM)()
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

    @property
    def dtype(self):
        return _lite_type_to_nptypes[LiteDataType(self.data_type)]

    @property
    def shapes(self):
        return list(self._shapes)[0 : self.ndim]

    @shapes.setter
    def shapes(self, shape):
        shape = list(shape)
        assert len(shape) <= MAX_DIM, "Layout max dim is 7."
        self._shapes = (c_size_t * MAX_DIM)(*shape)
        self.ndim = len(shape)

    def __repr__(self):
        data = {
            "shapes": self.shapes,
            "ndim": self.ndim,
            "data_type": _lite_type_to_nptypes[LiteDataType(self.data_type)],
        }
        return data.__repr__()


class _LiteTensorDesc(Structure):
    """
    warpper of the MegEngine Tensor

    Args:
        is_pinned_host: when set, the storage memory of the tensor is pinned
            memory. This is used to Optimize the H2D or D2H memory copy, if the
            device or layout is not set, when copy form other device(CUDA)
            tensor, this tensor will be automatically set to pinned tensor
        layout(LiteLayout): layout of this tensor
        device_type: type of device
        device_id: id of device
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
    Get the API from the lib
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
    Description of a block of data with neccessary information.

    Args:
        layout: layout of Tensor
        device_type: device type of Tensor
        device_id: device id of Tensor
        is_pinned_host: when set, the storage memory of the tensor is pinned
            memory. This is used to Optimize the H2D or D2H memory copy, if the
            device or layout is not set, when copy form other device(CUDA)
            tensor, this tensor will be automatically set to pinned tensor
        shapes: the shape of data
        dtype: data type

    Note:
        Dims of shape should be less than 8. The supported data type defines at
        LiteDataType
    """

    _api = _TensorAPI()._lib

    def __init__(
        self,
        layout=None,
        device_type=LiteDeviceType.LITE_CPU,
        device_id=0,
        is_pinned_host=False,
        shapes=None,
        dtype=None,
        physic_construct=True,
    ):
        self._tensor = _Ctensor()
        self._layout = LiteLayout()
        if layout is not None:
            self._layout = layout
        elif shapes is not None:
            shapes = list(shapes)
            self._layout = LiteLayout(shapes, dtype)
        self._device_type = device_type
        self._device_id = device_id
        self._is_pinned_host = is_pinned_host

        tensor_desc = _LiteTensorDesc()
        tensor_desc.layout = self._layout
        tensor_desc.device_type = device_type
        tensor_desc.device_id = device_id
        tensor_desc.is_pinned_host = is_pinned_host

        if physic_construct:
            self._api.LITE_make_tensor(tensor_desc, byref(self._tensor))
            self.update()

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
        share the same memory with the ``src_tensor``, the self memory will be
            freed

        Args:
            src_tensor: the source tensor that will share memory with this tensor
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
        if isinstance(layout, LiteLayout):
            self._layout = layout
        elif isinstance(layout, list):
            self._layout.shapes = layout

        self._api.LITE_set_tensor_layout(self._tensor, self._layout)

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
        get device type of the tensor
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

        c_types = _lite_dtypes_to_ctype[self._layout.data_type]
        self.np_array_type = np.ctypeslib._ctype_ndarray(
            c_types, list(self._layout.shapes)[0 : self._layout.ndim]
        )

    def copy_from(self, src_tensor):
        """
        copy memory form the src_tensor

        Args:
            src_tensor: source tensor
        """
        assert isinstance(src_tensor, LiteTensor)
        self._api.LITE_tensor_copy(self._tensor, src_tensor._tensor)
        self.update()

    def reshape(self, shape):
        """
        reshape the tensor with data not change.

        Args:
            shape: target shape
        """
        shape = list(shape)
        length = len(shape)
        c_shape = (c_int * length)(*shape)
        self._api.LITE_tensor_reshape(self._tensor, c_shape, length)
        self.update()

    def slice(self, start, end, step=None):
        """
        slice the tensor with gaven start, end, step

        Args:
            start: silce begin index of each dim
            end: silce end index of each dim
            step: silce step of each dim
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
        slice_tensor = LiteTensor(physic_construct=False)
        self._api.LITE_tensor_slice(
            self._tensor, c_start, c_end, c_step, length, byref(slice_tensor._tensor),
        )
        slice_tensor.update()
        return slice_tensor

    def get_ctypes_memory(self):
        """
        get the memory of the tensor, return c_void_p of the tensor memory
        """
        mem = c_void_p()
        self._api.LITE_get_tensor_memory(self._tensor, byref(mem))
        return mem

    def set_data_by_share(self, data, length=0, layout=None):
        """
        share the data to the tensor

        Args:
            data: the data will shared to the tensor, it should be a
                numpy.ndarray or ctypes data
        """
        if isinstance(data, np.ndarray):
            assert data.flags[
                "C_CONTIGUOUS"
            ], "input numpy is not continuous, please call input = np.ascontiguousarray(input) before call set_data_by_share"
            assert (
                self.is_continue
            ), "set_data_by_share can only apply in continue tensor."
            assert (
                self.is_pinned_host or self.device_type == LiteDeviceType.LITE_CPU
            ), "set_data_by_share can only apply in cpu tensor or pinned tensor."

            c_type = _lite_dtypes_to_ctype[LiteDataType(self._layout.data_type)]

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

        Args:
            data: the data to copy to tensor, it should be list, numpy.ndarraya
                or ctypes with length
            data_length: length of data in bytes
            layout: layout of data
        """
        if layout is not None:
            self.layout = layout

        assert self.is_continue, "set_data_by_copy can only apply in continue tensor."

        c_type = _lite_dtypes_to_ctype[LiteDataType(self._layout.data_type)]

        cpu_tensor = LiteTensor(self._layout)
        tensor_length = self.nbytes

        if type(data) == list:
            length = len(data)
            assert (
                length * sizeof(c_type) <= tensor_length
            ), "the length of input data to set to the tensor is too large."
            cdata = (c_type * length)(*data)
            self._api.LITE_reset_tensor_memory(cpu_tensor._tensor, cdata, tensor_length)
            self.copy_from(cpu_tensor)

        elif type(data) == np.ndarray:
            assert data.flags[
                "C_CONTIGUOUS"
            ], "input numpy is not continuous, please call input = np.ascontiguousarray(input) before call set_data_by_copy"
            self.layout = LiteLayout(data.shape, data.dtype)
            cpu_tensor.layout = LiteLayout(data.shape, data.dtype)
            cdata = data.ctypes.data_as(POINTER(c_type))
            self._api.LITE_reset_tensor_memory(cpu_tensor._tensor, cdata, self.nbytes)
            self.copy_from(cpu_tensor)

        else:
            assert (
                data_length == self.nbytes or layout is not None
            ), "when input data is ctypes, the length of input data or layout must set"
            self._api.LITE_reset_tensor_memory(cpu_tensor._tensor, data, tensor_length)
            self.copy_from(cpu_tensor)

    def get_data_by_share(self):
        """
        get the data in the tensor, add share the data with a new numpy, and
            return the numpy arrray

        Note:
            Be careful, the data in numpy is valid before the tensor memory is
                write again, such as LiteNetwok forward next time.
        """

        self.update()
        buffer = c_void_p()
        self._api.LITE_get_tensor_memory(self._tensor, byref(buffer))
        buffer = self.np_array_type.from_address(buffer.value)
        return np.ctypeslib.as_array(buffer)

    def to_numpy(self):
        """
        get the buffer of the tensor
        """
        self.update()
        if self.nbytes <= 0:
            np_type = _lite_type_to_nptypes[LiteDataType(self._layout.data_type)]
            return np.array([], dtype=np_type)
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
    concat tensors at expected dim to one tensor

    Args:
        dim : the dim to act concat
        device_type: the result tensor device type
        device_id: the result tensor device id
    """
    api = _TensorAPI()._lib
    length = len(tensors)
    c_tensors = [t._tensor for t in tensors]
    c_tensors = (_Ctensor * length)(*c_tensors)
    result_tensor = LiteTensor(physic_construct=False)
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


def lite_dtype_2_numpy(dtype):
    """
    convert lite dtype to corresponding numpy dtype

    Args:
        dtype(LiteDataType): source dtype
    """
    assert isinstance(
        dtype, LiteDataType
    ), "input must be LiteDataType when using lite_dtype_2_numpy."
    return _lite_type_to_nptypes[dtype]
