# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from .._imperative_rt.core2 import (
    getitem_cpp,
    set_cpp_astensor1d,
    set_cpp_use_symbolic_shape,
    setitem_cpp,
)
from .._trace_option import use_symbolic_shape
from .utils import astensor1d


def getitem(tensor, index):
    return getitem_cpp(tensor, index)


def setitem(tensor, index, value):
    return setitem_cpp(tensor, index, value)


set_cpp_use_symbolic_shape(use_symbolic_shape)
set_cpp_astensor1d(astensor1d)
