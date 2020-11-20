# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

from ._imperative_rt import CompNode


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


def device(obj):
    if isinstance(obj, Device):
        return obj
    return Device(obj)
