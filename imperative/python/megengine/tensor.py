# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.


import collections

from .core import Tensor as _Tensor
from .core.ops.builtin import Copy
from .core.tensor.core import apply
from .core.tensor.raw_tensor import as_device
from .device import _valid_device, get_default_device
from .utils.deprecation import deprecated


class Tensor(_Tensor):
    grad = None
    dmap_callback = None

    def __init__(self, data, dtype=None, device=None):
        if device is None:
            device = get_default_device()
        self.q_dict = {"mode": None, "scale": None, "zero_point": None}
        super().__init__(data, dtype=dtype, device=device)

    @deprecated(version="1.0", reason="no need to reuse an existing tensor since 1.0")
    def set_value(self, value):
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

    def __getstate__(self):
        r""" __getstate__ will be called for pickle serialization or deep copy
        """

        state = {
            "data": self.numpy(),
            "device": self.device.logical_name,
            "dtype": self.dtype,
            "qdict": self.q_dict,
        }
        return state

    def __setstate__(self, state):
        data = state.pop("data")
        logical_device = state.pop("device")
        if self.dmap_callback is not None:
            assert isinstance(logical_device, str)
            logical_device = self.dmap_callback(logical_device)
        dtype = state.pop("dtype")
        self.q_dict = state.pop("qdict")
        super().__init__(data, dtype=dtype, device=logical_device)

    def detach(self):
        r"""
        Returns a new tensor sharing the same data memory, which is treated as a constant
        during backward gradient calcuation, i.e. its gradient is zero.
        """
        Wrapper = type(self)
        Tensor = type(self.__wrapped__)
        return Wrapper(Tensor(self.__wrapped__._data))


tensor = Tensor


class Parameter(Tensor):
    r"""A kind of Tensor that is to be considered a module parameter.
    """
