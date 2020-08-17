# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import collections
import functools
import inspect
import sys
import typing
from abc import ABC

from .multipledispatch import Dispatcher


class OpBase(ABC):
    def __call__(self, *args):
        return apply(self, *args)


class TensorBase:
    pass


class TensorWrapperBase:
    pass


apply = Dispatcher("apply")

OpBase.apply = apply


@apply.register()
def _(op: OpBase, *args: TensorBase):
    raise NotImplementedError


@apply.register()
def _(op: OpBase, *args: TensorWrapperBase):
    assert args
    Wrapper = type(args[0])
    outputs = apply(op, *(i.__wrapped__ for i in args))
    assert isinstance(outputs, tuple)
    return tuple(map(Wrapper, outputs))
