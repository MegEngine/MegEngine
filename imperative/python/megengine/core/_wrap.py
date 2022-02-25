# -*- coding: utf-8 -*-
import numpy as np

from ._imperative_rt import CompNode
from ._imperative_rt.core2 import set_py_device_type


class Device:
    def __init__(self, device=None):
        if device is None:
            self._cn = CompNode()
        elif isinstance(device, Device):
            self._cn = device._cn
        elif isinstance(device, CompNode):
            self._cn = device
        else:
            self._cn = CompNode(device)

        self._logical_name = None

    @property
    def logical_name(self):
        if self._logical_name:
            return self._logical_name
        self._logical_name = self._cn.logical_name
        return self._logical_name

    def to_c(self):
        return self._cn

    def __repr__(self):
        return "{}({})".format(type(self).__qualname__, repr(self._cn))

    def __str__(self):
        return str(self._cn)

    def __hash__(self):
        return hash(str(self._cn))

    def __eq__(self, rhs):
        if not isinstance(rhs, Device):
            rhs = Device(rhs)
        return self._cn == rhs._cn


def as_device(obj):
    if isinstance(obj, Device):
        return obj
    return Device(obj)


set_py_device_type(Device)
