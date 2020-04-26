# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import numpy as np

from .mgb import intb1, intb2, intb4

_metadata_dict = {
    "quint8": {
        "is_unsigned": True,
        "np_dtype_str": "uint8",
        "mgb_dtype": {"name": "Quantized8Asymm", "qmin": 0, "qmax": 255,},
    },
    "qint8": {
        "is_unsigned": False,
        "np_dtype_str": "int8",
        "mgb_dtype": {"name": "QuantizedS8", "qmin": -128, "qmax": 127,},
    },
    "quint4": {
        "is_unsigned": True,
        "np_dtype_str": "uint8",
        "mgb_dtype": {"name": "Quantized4Asymm", "qmin": 0, "qmax": 15,},
    },
    "qint4": {
        "is_unsigned": False,
        "np_dtype_str": "int8",
        "mgb_dtype": {"name": "QuantizedS4", "qmin": -8, "qmax": 7,},
    },
    "qint32": {
        "is_unsigned": False,
        "np_dtype_str": "int32",
        "mgb_dtype": {"name": "QuantizedS32", "qmin": -(2 ** 31), "qmax": 2 ** 31 - 1,},
    },
}


def is_quantize(dtype):
    return (
        hasattr(dtype, "metadata")
        and dtype.metadata is not None
        and "mgb_dtype" in dtype.metadata
    )


def is_lowbit(dtype):
    return (dtype is intb1) or (dtype is intb2) or (dtype is intb4)


def get_scale(dtype):
    assert is_quantize(dtype)
    return dtype.metadata["mgb_dtype"]["scale"]


def get_zero_point(dtype):
    assert is_quantize(dtype)
    metadata = dtype.metadata["mgb_dtype"]
    assert metadata["name"] in ("Quantized8Asymm", "Quantized4Asymm")
    return metadata["zero_point"]


def _check_zero_point(zp: int, dtype_str: str):
    qmin = _metadata_dict[dtype_str]["mgb_dtype"]["qmin"]
    qmax = _metadata_dict[dtype_str]["mgb_dtype"]["qmax"]
    if zp < qmin or zp > qmax:
        raise ValueError(
            "zero_point should be within [{}, {}] for {}".format(qmin, qmax, dtype_str)
        )


def _get_dtype(dtype_str: str, scale, zp):
    if zp is not None:
        if int(zp) != zp:
            raise ValueError("zero_point should be an integer")
        zp = int(zp)
        _check_zero_point(zp, dtype_str)
    metadata = _metadata_dict[dtype_str]["mgb_dtype"]
    np_dtype_str = _metadata_dict[dtype_str]["np_dtype_str"]
    return np.dtype(
        np_dtype_str,
        metadata={"mgb_dtype": {**metadata, "scale": float(scale), "zero_point": zp,}},
    )


def quint8(scale, zero_point):
    """
    Consturct a quantized unsigned int8 data type with ``scale`` (float) and
    ``zero_point`` (uint8). The real value represented by a quint8 data type is
    float_val = scale * (uint8_val - zero_point)
    """
    return _get_dtype("quint8", scale, zero_point)


def qint8(scale):
    """
    Construct a quantized int8 data type with ``scale`` (float). The real value
    represented by a qint8 data type is float_val = scale * int8_val
    """
    return _get_dtype("qint8", scale, None)


def qint32(scale):
    """
    Construct a quantized int32 data type with ``scale`` (float). The real value
    represented by a qint32 data type is float_val = scale * int32_val
    """
    return _get_dtype("qint32", scale, None)


def quint4(scale, zero_point):
    """
    Consturct a quantized unsigned int4 data type with ``scale`` (float) and
    ``zero_point`` (uint8). The real value represented by a quint4 data type is
    float_val = scale * (uint4_val - zero_point)
    """
    return _get_dtype("quint4", scale, zero_point)


def qint4(scale):
    """
    Construct a quantized int4 data type with ``scale`` (float). The real value
    represented by a qint4 data type is float_val = scale * int4_val
    """
    return _get_dtype("qint4", scale, None)


def _convert_to_dtype(arr: np.ndarray, dtype: np.dtype, dtype_str: str):
    metadata = _metadata_dict[dtype_str]["mgb_dtype"]
    arr_metadata = dtype.metadata["mgb_dtype"]
    if not isinstance(arr, np.ndarray):
        raise ValueError("arr parameter should be instance of np.ndarray")
    if not is_quantize(dtype) or arr_metadata["name"] != metadata["name"]:
        raise ValueError("dtype parameter should be a {} dtype".format(dtype_str))
    is_unsigned = _metadata_dict[dtype_str]["is_unsigned"]
    if is_unsigned:
        scale, zp = (
            arr_metadata["scale"],
            arr_metadata["zero_point"],
        )
        return (
            (np.round(arr / scale) + zp)
            .clip(metadata["qmin"], metadata["qmax"])
            .astype(dtype)
        )
    else:
        # don't trick to combine with is_unsigned for consistency with cpp interface
        scale = arr_metadata["scale"]
        return (
            np.round(arr / scale).clip(metadata["qmin"], metadata["qmax"]).astype(dtype)
        )


def _convert_from_dtype(arr: np.ndarray, dtype_str: str):
    metadata = _metadata_dict[dtype_str]["mgb_dtype"]
    arr_metadata = arr.dtype.metadata["mgb_dtype"]
    if not isinstance(arr, np.ndarray):
        raise ValueError("arr parameter should be instance of np.ndarray")
    if not is_quantize(arr.dtype) or arr_metadata["name"] != metadata["name"]:
        raise ValueError("arr's dtype should be a {} dtype".format(dtype_str))
    is_unsigned = _metadata_dict[dtype_str]["is_unsigned"]
    if is_unsigned:
        scale, zp = (
            arr_metadata["scale"],
            arr_metadata["zero_point"],
        )
        return (arr.astype(np.float32) - zp) * scale
    else:
        # don't trick to combine with is_unsigned for consistency with cpp interface
        scale = arr_metadata["scale"]
        return (arr.astype(np.float32)) * scale


def convert_to_quint8(arr: np.ndarray, q: np.dtype):
    """
    Quantize a float NumPy ndarray into a quint8 one with specified params.

    :param arr: Input ndarray.
    :param q: Target data type, should be a quint8.
    """
    return _convert_to_dtype(arr, q, "quint8")


def convert_from_quint8(arr: np.ndarray):
    """
    Dequantize a quint8 NumPy ndarray into a float one.

    :param arr: Input ndarray.
    """
    return _convert_from_dtype(arr, "quint8")


def convert_to_qint8(arr: np.ndarray, q: np.dtype):
    """
    Quantize a float NumPy ndarray into a qint8 one with specified params.

    :param arr: Input ndarray.
    :param q: Target data type, should be a qint8.
    """
    return _convert_to_dtype(arr, q, "qint8")


def convert_from_qint8(arr: np.ndarray):
    """
    Dequantize a qint8 NumPy ndarray into a float one.

    :param arr: Input ndarray.
    """
    return _convert_from_dtype(arr, "qint8")


def convert_to_qint32(arr: np.ndarray, q: np.dtype):
    """
    Quantize a float NumPy ndarray into a qint32 one with specified params.

    :param arr: Input ndarray.
    :param q: Target data type, should be a qint8.
    """
    return _convert_to_dtype(arr, q, "qint32")


def convert_from_qint32(arr):
    """
    Dequantize a qint32 NumPy ndarray into a float one.

    :param arr: Input ndarray.
    """
    return _convert_from_dtype(arr, "qint32")


def convert_to_quint4(arr: np.ndarray, q: np.dtype):
    """
    Quantize a float NumPy ndarray into a quint4 one with specified params.

    :param arr: Input ndarray.
    :param q: Target data type, should be a quint4.
    """
    return _convert_to_dtype(arr, q, "quint4")


def convert_from_quint4(arr: np.ndarray):
    """
    Dequantize a quint4 NumPy ndarray into a float one.

    :param arr: Input ndarray.
    """
    return _convert_from_dtype(arr, "quint4")


def convert_to_qint4(arr: np.ndarray, q: np.dtype):
    """
    Quantize a float NumPy ndarray into a qint4 one with specified params.

    :param arr: Input ndarray.
    :param q: Target data type, should be a qint4.
    """
    return _convert_to_dtype(arr, q, "qint4")


def convert_from_qint4(arr: np.ndarray):
    """
    Dequantize a qint4 NumPy ndarray into a float one.

    :param arr: Input ndarray.
    """
    return _convert_from_dtype(arr, "qint4")
