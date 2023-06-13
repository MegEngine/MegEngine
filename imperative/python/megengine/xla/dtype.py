from functools import lru_cache, partial

import numpy as np

from ..tensor import Parameter as MgeParameter
from ..tensor import Tensor as MgeTensor
from .lib import xla_client as xc

_python_scalar_dtype_to_npdtypes = {
    bool: np.dtype("bool"),
    int: np.dtype("int64"),
    float: np.dtype("float64"),
    complex: np.dtype("complex128"),
}

_python_scalar_dtypes = list(_python_scalar_dtype_to_npdtypes.keys())

bfloat16 = xc.bfloat16
_bfloat16_dtype = np.dtype(bfloat16)
_float_types = [
    _bfloat16_dtype,
    np.dtype("float16"),
    np.dtype("float32"),
    np.dtype("float64"),
]

_numpy_scalar_types = {
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.complex64,
    np.complex128,
    np.bool_,
    np.longlong,
    np.intc,
} | set(np.dtype(dt).type for dt in _float_types)

_np_types = {np.ndarray} | _numpy_scalar_types

_dtype_to_32bit_dtype = {
    np.dtype("int64"): np.dtype("int32"),
    np.dtype("uint64"): np.dtype("uint32"),
    np.dtype("float64"): np.dtype("float32"),
    np.dtype("complex128"): np.dtype("complex64"),
}


def _scalar_type_to_dtype(typ, value):
    dtype = canonicalize_dtype(_python_scalar_dtype_to_npdtypes[typ])
    if typ is int and value is not None:
        if value < np.iinfo(dtype).min or value > np.iinfo(dtype).max:
            raise OverflowError(f"Python int {value} too large to convert to {dtype}")
    return dtype


# do not enable x64 because megengine only support x32
@lru_cache(maxsize=None)
def canonicalize_dtype(dtype, x64_enabled=False, allow_opaque_dtype=False):
    assert allow_opaque_dtype == False and x64_enabled == False
    try:
        dtype_ = np.dtype(dtype)
    except TypeError as e:
        raise TypeError(f"dtype {dtype!r} not understood") from e

    if x64_enabled:
        return dtype_
    else:
        return _dtype_to_32bit_dtype.get(dtype_, dtype_)


def _canonicalize_ndarray_dtype(x):
    return np.asarray(x, canonicalize_dtype(x.dtype))


def _canonicalize_python_scalar_dtype(typ, x):
    return np.asarray(x, canonicalize_dtype(_scalar_type_to_dtype(typ, x)))


def _canonicalize_mgetensor_dtype(x: MgeTensor):
    canonicalized = canonicalize_dtype(x.dtype)
    if canonicalized != x.dtype:
        return x.astype(canonicalized)
    return x


canonicalize_args_handlers = {}

canonicalize_args_handlers.update(
    (t, _canonicalize_ndarray_dtype) for t in _numpy_scalar_types
)
canonicalize_args_handlers[np.ndarray] = _canonicalize_ndarray_dtype
canonicalize_args_handlers.update(
    (t, partial(_canonicalize_python_scalar_dtype, t)) for t in _python_scalar_dtypes
)
canonicalize_args_handlers[MgeTensor] = _canonicalize_mgetensor_dtype
canonicalize_args_handlers[MgeParameter] = _canonicalize_mgetensor_dtype


def canonicalize_arg(x):
    typ = type(x)
    handler = canonicalize_args_handlers.get(typ)
    if handler:
        return handler(x)
    raise TypeError(f"No canonicalize_dtype handler for type: {type(x)}")
