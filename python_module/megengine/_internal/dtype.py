# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import numpy as np

from .mgb import intb1, intb2, intb4


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
    assert metadata["name"] == "Quantized8Asymm"
    return metadata["zero_point"]


def quint8(scale, zero_point):
    """
    Consturct a quantized unsigned int8 data type with ``scale`` (float) and
    ``zero_point`` (uint8). The real value represented by a quint8 data type is
    float_val = scale * (uint8_val - zero_point)
    """
    int_zp = int(zero_point)
    assert int_zp == zero_point, "zero_point should be an integer"
    if int_zp < 0 or int_zp > 255:
        raise ValueError("zero_point should be within [0, 255] for quint8")
    return np.dtype(
        np.uint8,
        metadata={
            "mgb_dtype": {
                "name": "Quantized8Asymm",
                "scale": float(scale),
                "zero_point": int(zero_point),
            }
        },
    )


def qint8(scale):
    """
    Construct a quantized int8 data type with ``scale`` (float). The real value
    represented by a qint8 data type is float_val = scale * int8_val
    """
    return np.dtype(
        np.int8, metadata={"mgb_dtype": {"name": "QuantizedS8", "scale": float(scale)}}
    )


def qint32(scale):
    """
    Construct a quantized int32 data type with ``scale`` (float). The real value
    represented by a qint32 data type is float_val = scale * int32_val
    """
    return np.dtype(
        np.int32,
        metadata={"mgb_dtype": {"name": "QuantizedS32", "scale": float(scale)}},
    )


def convert_to_quint8(arr, q):
    """
    Quantize a float NumPy ndarray into a quint8 one with specified params.

    :param arr: Input ndarray.
    :type arr: :class:`np.ndarray`
    :param q: Target data type, should be a quint8.
    :type q: :class:`np.dtype`
    """
    assert isinstance(arr, np.ndarray)
    assert (
        "mgb_dtype" in q.metadata
        and q.metadata["mgb_dtype"]["name"] == "Quantized8Asymm"
    ), "q should be a quint8 dtype"
    scale, zp = q.metadata["mgb_dtype"]["scale"], q.metadata["mgb_dtype"]["zero_point"]
    return (np.round(arr / scale) + zp).clip(0, 255).astype(q)


def convert_from_quint8(arr):
    """
    Dequantize a quint8 NumPy ndarray into a float one.

    :param arr: Input ndarray.
    """
    assert isinstance(arr, np.ndarray)
    assert (
        "mgb_dtype" in arr.dtype.metadata
        and arr.dtype.metadata["mgb_dtype"]["name"] == "Quantized8Asymm"
    ), "arr should be a ndarray with quint8 dtype"
    scale, zp = (
        arr.dtype.metadata["mgb_dtype"]["scale"],
        arr.dtype.metadata["mgb_dtype"]["zero_point"],
    )
    return (arr.astype(np.float32) - zp) * scale


def convert_to_qint8(arr, q):
    """
    Quantize a float NumPy ndarray into a qint8 one with specified params.

    :param arr: Input ndarray.
    :type arr: :class:`np.ndarray`
    :param q: Target data type, should be a qint8.
    :type q: :class:`np.dtype`
    """
    assert isinstance(arr, np.ndarray)
    assert (
        "mgb_dtype" in q.metadata and q.metadata["mgb_dtype"]["name"] == "QuantizedS8"
    ), "q should be a qint8 dtype"
    scale = q.metadata["mgb_dtype"]["scale"]
    return (np.round(arr / scale)).clip(-128, 127).astype(q)


def convert_from_qint8(arr):
    """
    Dequantize a qint8 NumPy ndarray into a float one.

    :param arr: Input ndarray.
    """
    assert isinstance(arr, np.ndarray)
    assert (
        "mgb_dtype" in arr.dtype.metadata
        and arr.dtype.metadata["mgb_dtype"]["name"] == "QuantizedS8"
    ), "arr should be a ndarray with qint8 dtype"
    scale = arr.dtype.metadata["mgb_dtype"]["scale"]
    return arr.astype(np.float32) * scale


def convert_to_qint32(arr, q):
    """
    Quantize a float NumPy ndarray into a qint32 one with specified params.

    :param arr: Input ndarray.
    :type arr: :class:`np.ndarray`
    :param q: Target data type, should be a qint8.
    :type q: :class:`np.dtype`
    """
    assert isinstance(arr, np.ndarray)
    assert (
        "mgb_dtype" in q.metadata and q.metadata["mgb_dtype"]["name"] == "QuantizedS32"
    ), "q should be a qint32 dtype"
    scale = q.metadata["mgb_dtype"]["scale"]
    return (np.round(arr / scale)).clip(-(2 ** 31), 2 ** 31).astype(q)


def convert_from_qint32(arr):
    """
    Dequantize a qint32 NumPy ndarray into a float one.

    :param arr: Input ndarray.
    """
    assert isinstance(arr, np.ndarray)
    assert (
        "mgb_dtype" in arr.dtype.metadata
        and arr.dtype.metadata["mgb_dtype"]["name"] == "QuantizedS32"
    ), "arr should be a ndarray with qint8 dtype"
    scale = arr.dtype.metadata["mgb_dtype"]["scale"]
    return arr.astype(np.float32) * scale
