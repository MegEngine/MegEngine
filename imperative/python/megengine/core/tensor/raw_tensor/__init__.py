# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import functools

import numpy as np

from ..._imperative_rt import CompNode, DeviceTensorND
from ..._imperative_rt.imperative import (
    _get_dev_tensor,
    apply_op,
    delete,
    get_device,
    get_dtype,
    get_shape,
    get_value,
    put,
)
from ..._wrap import device as as_device
from ...ops.builtin import Copy, OpDef, TypeCvt
from ...ops.special import Const
from ..core import OpBase, TensorBase, apply


class RawTensor(TensorBase):

    _init_cb = None
    _del_cb = None

    def __init__(self, handle):
        self._handle = handle
        if self._init_cb:
            self._init_cb()

    @property
    def dtype(self):
        return get_dtype(self._handle)

    @property
    def device(self):
        return as_device(get_device(self._handle))

    @property
    def shape(self):
        return get_shape(self._handle)

    def numpy(self):
        return get_value(self._handle)

    def _dev_tensor(self):
        return _get_dev_tensor(self._handle)

    def __repr__(self):
        return "{}({}, device='{}')".format(
            type(self).__qualname__, repr(self.numpy()), self.device
        )

    def __del__(self):
        if self._del_cb:
            self._del_cb()
        delete(self._handle)


@apply.register()
def _(op: OpDef, *args: RawTensor):
    outputs = apply_op(op, tuple(i._handle for i in args))
    return tuple(map(RawTensor, outputs))


@apply.register()
def _(op: Const, *args: RawTensor):
    dtype = op.dtype
    device = as_device(op.device).to_c()
    return (as_raw_tensor(op.value, dtype=dtype, device=device),)


@functools.singledispatch
def as_raw_tensor(obj, dtype=None, device=None):
    obj = np.asarray(obj, dtype=dtype)
    if obj.dtype == np.float64:
        obj = obj.astype(np.float32)
    if obj.dtype == np.int64:
        obj = obj.astype(np.int32)
    return as_raw_tensor(obj, device=device)


@as_raw_tensor.register(np.ndarray)
def _(array: np.ndarray, dtype=None, device=None):
    device = None if device is None else as_device(device).to_c()
    return RawTensor(put(array, dtype=dtype, device=device))


@as_raw_tensor.register(RawTensor)
def _(tensor: RawTensor, dtype=None, device=None):
    if dtype is not None:
        dtype = np.dtype(dtype)
        if dtype != tensor.dtype:
            (tensor,) = apply(TypeCvt(dtype=dtype), tensor)
    if device is not None:
        device = as_device(device)
        if device != tensor.device:
            (tensor,) = apply(Copy(comp_node=device.to_c()), tensor)
    return tensor
