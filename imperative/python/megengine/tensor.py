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
from .device import get_default_device


class Tensor(_Tensor):
    requires_grad = False
    dmap_callback = None

    def __init__(self, data, dtype=None, device=None):
        if device is None:
            device = get_default_device()
        self.q_dict = {"mode": None, "scale": None, "zero_point": None}
        super().__init__(data, dtype=dtype, device=device)

    def set_value(self, value):
        self._reset(value)

    def reset_zero(self):
        self *= 0

    def __getstate__(self):
        r""" __getstate__ will be called for pickle serialization or deep copy
        """

        state = {
            "data": self.numpy(),
            "device": str(self.device),
            "dtype": self.dtype,
            "qdict": self.q_dict,
        }
        return state

    def __setstate__(self, state):
        data = state.pop("data")
        device = state.pop("device")
        if self.dmap_callback is not None:
            assert isinstance(device, str)
            device = self.dmap_callback(device)
        dtype = state.pop("dtype")
        self.q_dict = state.pop("qdict")
        super().__init__(data, dtype=dtype, device=device)

    def detach(self):
        r"""
        Returns a new tensor which is treated as constant during backward gradient calcuation,
        i.e. its gradient is zero.

        :param inp: input tensor

        """
        Wrapper = type(self)
        Tensor = type(self.__wrapped__)
        return Wrapper(Tensor(self.__wrapped__._data))


tensor = Tensor


class Dict(collections.MutableMapping):
    def __init__(self, *args, key=None, **kwargs):
        self.data = {}
        if key:
            self.keyfn = key
        for i in args:
            self.update(i)
        self.update(**kwargs)

    @staticmethod
    def keyfn(key):  # pylint: disable=method-hidden
        return key

    def __getitem__(self, key):
        _, v = self.data[self.keyfn(key)]
        return v

    def __setitem__(self, key, value):
        self.data[self.keyfn(key)] = key, value

    def __delitem__(self, key):
        del self.data[self.keyfn(key)]

    def __iter__(self):
        for _, (k, _) in self.data.items():
            yield k

    def __len__(self):
        return len(self.data)


class TensorDict(Dict):  # pylint: disable=too-many-ancestors
    class keyfn:
        def __new__(cls, x: Tensor):
            if not isinstance(x, Tensor):
                return x
            return super().__new__(cls)

        def __init__(self, x: Tensor):
            self._data = x  # do not save id directly to make pickle work

        def __hash__(self):
            return id(self._data)

        def __eq__(self, other):
            # pylint: disable=undefined-variable
            return isinstance(other, __class__) and id(self._data) == id(other._data)

    def __init__(self, *args):
        super().__init__(*args)
