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
from .dtype import is_equal, is_quantize


def dtype_promotion(inputs):
    """
    Returns the dtype that would result from performing an arithmetic
    operation on the provided input tensors and scalars.
    """
    # map numpy.dtype.kind to priority
    category_priority = {
        "f": 3,  # floating-point
        "i": 2,  # signed integer
        "u": 2,  # unsigned integer
        "b": 1,  # boolean
    }

    def scalar2dtype(x):
        """
        For scalar `x`, returns its corresponding type. A floating point scalar
        has dtype 'float32'. An integral non-boolean scalar has dtype 'int32'.
        A boolean scalar has dtype 'bool'.
        """
        if isinstance(x, bool):
            return np.bool_
        if isinstance(x, int):
            return np.int32
        if isinstance(x, float):
            return np.float32

    def promote_types(types, cat):
        """
        Returns the data type with sufficient size to hold all types of
        category `cat` in the list `types`.
        """
        used_types = [
            i for i in types if category_priority.get(np.dtype(i).kind, 0) == cat
        ]
        assert len(used_types) > 0
        res = used_types[0]
        for i in used_types:
            res = np.promote_types(res, i)
        return res

    def max_priority(types):
        """
        Returns the maximum value of the priority of each type in the list
        `types`.
        """
        if not types:
            return 0
        else:
            return max([category_priority.get(np.dtype(i).kind, 0) for i in types])

    scalars = []
    tensors = []

    for data in inputs:
        if hasattr(data, "dtype"):
            tensors.append(data.dtype)
        elif isinstance(data, (float, int, bool)):
            scalars.append(scalar2dtype(data))

    max_pri_scalars = max_priority(scalars)
    max_pri_tensors = max_priority(tensors)

    assert max_pri_scalars > 0 or max_pri_tensors > 0

    if max_pri_scalars > max_pri_tensors:
        return promote_types(scalars, max_pri_scalars)
    else:
        return promote_types(tensors, max_pri_tensors)


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
    if not is_equal(x.dtype, dtype):
        (x,) = apply(builtin.TypeCvt(param=dtype), x)
    return x


def convert_single_value(v, inputs, *, dtype=None, device=None):
    tensors = [i for i in inputs if isinstance(i, (TensorBase, TensorWrapperBase))]
    assert len(tensors) > 0
    if isinstance(v, (TensorWrapperBase, TensorBase)):
        v = astype(v, v.dtype if is_quantize(v.dtype) else dtype)
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

    if not isinstance(x, collections.abc.Sequence):
        raise TypeError

    if any(isinstance(i, (TensorBase, TensorWrapperBase)) for i in x):
        x = concatenate(x, device=device)
        if dtype is not None:
            x = astype(x, dtype)
        return x

    (x,) = Const(x, dtype=dtype, device=device)(*reference)
    return x


def _expand_int(s, i):
    if isinstance(i, (TensorBase, TensorWrapperBase)):
        s += list(i.numpy())
        return
    if isinstance(i, Iterable):
        for ii in i:
            _expand_int(s, ii)
        return
    if np.issubdtype(type(i), np.integer):
        s.append(i)
        return
    raise


def make_shape_tuple(shape):
    s = []
    _expand_int(s, shape)
    return tuple(s)
