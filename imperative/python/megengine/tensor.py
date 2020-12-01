# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.


import collections

import numpy as np

from .core._imperative_rt import CompNode
from .core._imperative_rt.core2 import Tensor as _Tensor
from .core._imperative_rt.core2 import apply
from .core._trace_option import use_symbolic_shape
from .core.ops.builtin import Copy, GetVarShape
from .core.tensor.raw_tensor import as_device
from .core.tensor.tensor_wrapper import ArrayMethodMixin
from .device import _valid_device, get_default_device
from .utils.deprecation import deprecated


class Tensor(_Tensor, ArrayMethodMixin):
    grad = None
    dmap_callback = None
    q_dict = {"mode": None, "scale": None, "zero_point": None}

    def __new__(cls, data, dtype=None, device=None):
        if device is None:
            cn = get_default_device()
        elif isinstance(device, str):
            if cls.dmap_callback is not None:
                cn = CompNode(cls.dmap_callback(device))
            else:
                cn = CompNode(device)
        else:
            assert isinstance(device, CompNode)
            cn = device

        if isinstance(data, _Tensor):
            obj = _Tensor.__new__(cls, data)
        else:
            obj = _Tensor.__new__(cls, data, dtype, cn)
        return obj

    @property
    def shape(self):
        shape = super().shape
        if shape == () or not use_symbolic_shape():
            return shape
        return apply(GetVarShape(), self)[0]

    @property
    def _tuple_shape(self):
        return super().shape

    def __repr__(self):
        piece = "Tensor("
        with np.printoptions(precision=4, suppress=True):
            piece += "{}".format(str(self.numpy()))
        if self.dtype != np.float32:
            piece += ", dtype={}".format(np.dtype(self.dtype).name)
        piece += ", device={}".format(self.device) + ")"
        return piece

    @deprecated(version="1.0", reason="no need to reuse an existing tensor since 1.0")
    def set_value(self, value):
        if not isinstance(value, _Tensor):
            value = Tensor(value, dtype=self.dtype, device=self.device)
        self._reset(value)

    @deprecated(version="1.0", reason="use *= 0 instead")
    def reset_zero(self):
        self *= 0

    def to(self, device):
        if isinstance(device, str) and not _valid_device(device):
            raise ValueError(
                "invalid device name {}. For the correct format of the device name, please refer to the instruction of megengine.device.set_default_device()".format(
                    device
                )
            )
        cn = as_device(device).to_c()
        return apply(Copy(comp_node=cn), self)[0]

    @property
    def requires_grad(self):
        raise AttributeError("requires_grad is reserved for future use")

    @requires_grad.setter
    def requires_grad(self, value):
        raise AttributeError("requires_grad is reserved for future use")

    @requires_grad.deleter
    def requires_grad(self):
        raise AttributeError("requires_grad is reserved for future use")

    def __hash__(self):
        return id(self)

    def __getnewargs__(self):
        r""" __getnewargs__ will be called for pickle serialization or deep copy
        """
        return (self.numpy(), self.dtype, self.device.logical_name)

    def __getstate__(self):
        r""" __getstate__ will be called for pickle serialization or deep copy
        """

        state = {
            "qdict": self.q_dict,
        }
        return state

    def __setstate__(self, state):
        self.q_dict = state.pop("qdict")

    def detach(self):
        r"""
        Returns a new tensor sharing the same data memory, which is treated as a constant
        during backward gradient calcuation, i.e. its gradient is zero.
        """
        Wrapper = type(self)
        return Wrapper(self)


tensor = Tensor


class Parameter(Tensor):
    r"""
    A kind of Tensor that is to be considered a module parameter.
    """
