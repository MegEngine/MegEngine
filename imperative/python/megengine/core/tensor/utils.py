# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import collections
from typing import Iterable, Union

import numpy as np

from ..ops import builtin
from ..ops.special import Const
from ..tensor.core import OpBase, TensorBase, TensorWrapperBase, apply


def dtype_promotion(raw_inputs):
    def add_dtype(i):
        if type(i) == int:
            return np.array(i, dtype=np.int32)
        if type(i) == float:
            return np.array(i, dtype=np.float32)
        if type(i) == bool:
            return np.array(i, dtype=np.bool_)
        return None

    scalar_inputs = [
        add_dtype(i) for i in raw_inputs if not hasattr(i, "dtype") and add_dtype(i)
    ]
    inputs = [i for i in raw_inputs if hasattr(i, "dtype")]
    assert len(scalar_inputs + inputs) > 0
    dtype = np.result_type(*inputs)
    dtype_all = np.result_type(*(inputs + scalar_inputs))
    assert (
        dtype != np.float64 and dtype != np.int64
    ), "unsupport dtype {} by dtype_promotion, please use explict type convert".format(
        dtype
    )
    if dtype_all == np.bool_:
        for i in raw_inputs:
            if not hasattr(i, "dtype") or i.dtype != np.bool_:
                raise TypeError(
                    "bool dtype can not be operated with an element without bool dtype"
                )
    if dtype_all == np.float64:
        dtype_all = np.float32
    return dtype_all


def get_device(inputs):
    device = None
    for i in inputs:
        if isinstance(i, (TensorWrapperBase, TensorBase)):
            if device is None:
                device = i.device
            elif device != i.device:
                raise ValueError("ambiguous device: {} vs {}".format(device, i.device))
    assert device is not None
    return device


def concatenate(inputs, axis=0, *, device=None):
    dtype = dtype_promotion(inputs)
    device = get_device(inputs)

    def convert(x):
        return convert_single_value(x, inputs, dtype=dtype)

    inputs = tuple(map(convert, inputs))
    (result,) = apply(builtin.Concat(axis=axis, comp_node=device.to_c()), *inputs)
    return result


def astype(x, dtype):
    dtype = np.dtype(dtype)
    if x.dtype != dtype:
        (x,) = apply(builtin.TypeCvt(param=dtype), x)
    return x


def convert_single_value(v, inputs, *, dtype=None, device=None):
    tensors = [i for i in inputs if isinstance(i, (TensorBase, TensorWrapperBase))]
    assert len(tensors) > 0
    if isinstance(v, (TensorWrapperBase, TensorBase)):
        v = astype(v, dtype)
    else:
        (v,) = Const(v, dtype=dtype, device=device)(*tensors)
    return v


def convert_inputs(*args: TensorBase):
    dtype = dtype_promotion(args)
    device = get_device(args)

    def convert(value):
        if value is None:
            return value
        return convert_single_value(value, args, dtype=dtype, device=device)

    return tuple(map(convert, args))


def result_type(*args):
    dtypes = []
    for i in args:
        if isinstance(i, (TensorWrapperBase, TensorBase)):
            dtypes.append(i.dtype)
            continue
        try:
            dtypes.append(np.dtype(i))
        except TypeError:
            pass
    return np.result_type(*dtypes)


def isscalar(x):
    try:
        return x.ndim == 0
    except:
        pass
    return np.isscalar(x)


def astensor1d(x, *reference, dtype=None, device=None):
    """
    Convert something to 1D tensor. Support following types
    * sequence of scalar literal / tensor
    * numpy array
    * tensor (returned as is, regardless of dtype and device)
    """
    try:
        ndim = x.ndim
    except AttributeError:
        pass
    else:
        if ndim != 1:
            raise ValueError("ndim != 1: %d" % ndim)
        if not isinstance(x, (TensorBase, TensorWrapperBase)):
            (x,) = Const(x, dtype=dtype, device=device)(*reference)
        return x

    if not isinstance(x, collections.Sequence):
        raise TypeError

    if any(isinstance(i, (TensorBase, TensorWrapperBase)) for i in x):
        x = concatenate(x, device=device)
        if dtype is not None:
            x = astype(x, dtype)
        return x

    (x,) = Const(x, dtype=dtype, device=device)(*reference)
    return x
