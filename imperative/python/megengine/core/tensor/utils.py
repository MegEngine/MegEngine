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

from .._imperative_rt.core2 import Tensor, apply, dtype_promotion, get_device
from ..ops import builtin
from ..ops.special import Const
from .dtype import is_dtype_equal, is_quantize
from .megbrain_graph import VarNode

_enable_convert_inputs = True


def get_convert_inputs():
    """ get the curerent state of `_enable_convert_inputs` """
    return _enable_convert_inputs


def set_convert_inputs(flag):
    """ This function is a temporary workaround for reducing the overhead of operator
    invocations. The function `convert_inputs` is disabled if the global state
    `_enable_convert_inputs` is set to `False`, otherwise enabled. This function is for
    internal use only, and should be removed when the tensor-like system is refactored.
    """
    global _enable_convert_inputs
    backup = _enable_convert_inputs
    _enable_convert_inputs = flag
    return backup


def concatenate(inputs, axis=0, *, device=None):
    dtype = dtype_promotion(inputs)
    device = get_device(inputs)

    def convert(x):
        return convert_single_value(x, dtype=dtype, device=device)

    inputs = tuple(map(convert, inputs))
    (result,) = apply(builtin.Concat(axis=axis, comp_node=device), *inputs)
    return result


def astype(x, dtype):
    dtype = np.dtype(dtype)
    if not is_dtype_equal(x.dtype, dtype):
        isscalar = x.isscalar()
        (x,) = apply(builtin.TypeCvt(dtype=dtype), x)
        if isscalar:
            x.setscalar()
    return x


def convert_single_value(v, *, dtype=None, device=None):
    if isinstance(v, (Tensor, VarNode)):
        if not is_quantize(v.dtype):
            v = astype(v, dtype)
    else:
        (v,) = Const(v, dtype=dtype, device=device)()
    return v


def convert_inputs(*args: Tensor):
    if not _enable_convert_inputs:
        return args

    dtype = dtype_promotion(args)
    device = get_device(args)

    def convert(value):
        if value is None:
            return value
        return convert_single_value(value, dtype=dtype, device=device)

    return tuple(map(convert, args))


def result_type(*args):
    dtypes = []
    for i in args:
        if isinstance(i, Tensor):
            dtypes.append(i.dtype)
            continue
        try:
            dtypes.append(np.dtype(i))
        except TypeError:
            pass
    return np.result_type(*dtypes)


def isscalar(x):

    if isinstance(x, Tensor):
        return x.isscalar()

    return np.isscalar(x)


def setscalar(x):
    if isinstance(x, Tensor):
        x.setscalar()
    else:
        raise NotImplementedError("Unsupport type {}".format(type(x)))


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
        if ndim != 0 and ndim != 1:
            raise ValueError("ndim != 1 or 0, get : %d" % ndim)
        if not isinstance(x, Tensor):
            (x,) = Const(x, dtype=dtype, device=device)(*reference)
        return x

    if not isinstance(x, collections.abc.Sequence):
        raise TypeError

    if any(isinstance(i, Tensor) for i in x):
        x = concatenate(x, device=device)
        if dtype is not None:
            x = astype(x, dtype)
        return x
    (x,) = Const(x, dtype=dtype, device=device)(*reference)
    return x


def _expand_int(s, i):
    if isinstance(i, Tensor):
        i_np = i.numpy()
        if i_np.ndim == 0:
            s.append(int(i_np))
        else:
            s += list(i_np)
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
