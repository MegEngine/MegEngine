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

import multipledispatch


class OpBase(ABC):
    def __call__(self, *args):
        return apply(self, *args)


class TensorBase:
    pass


class TensorWrapperBase:
    pass


class Dispatcher(multipledispatch.Dispatcher):
    def add(self, f, g=None):
        if g is None:
            super().add(get_signature(f), f)
        else:
            super().add(f, g)

        return f

    def __get__(self, instance, owner=None):
        if instance is not None:
            return self
        return functools.partial(self, instance)


if sys.version_info < (3, 6):

    def parse_union(ann):
        if type(ann) is not typing.UnionMeta:
            return
        return ann.__union_params__


elif sys.version_info < (3, 7):

    def parse_union(ann):
        if type(ann) is not typing._Union:
            return
        return ann.__args__


elif sys.version_info < (3, 8):

    def parse_union(ann):
        if type(ann) is not typing._GenericAlias:
            if type(ann) is not typing.Union:
                return
        else:
            if ann.__origin__ is not typing.Union:
                return
        return ann.__args__


else:

    def parse_union(ann):
        if typing.get_origin(ann) is not typing.Union:
            return
        return typing.get_args(ann)


def get_signature(function, op_type=None):
    sig = inspect.signature(function)
    types = []
    for p in sig.parameters.values():
        ann = p.annotation
        ann = parse_union(ann) or ann
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            types.append(ann)
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            types.append([ann])
    return tuple(types)


apply = Dispatcher("apply")

OpBase.apply = apply


@apply.add
def _(op: OpBase, *args: TensorBase):
    raise NotImplementedError


@apply.add
def _(op: OpBase, *args: TensorWrapperBase):
    assert args
    Wrapper = type(args[0])
    outputs = apply(op, *(i.__wrapped__ for i in args))
    assert isinstance(outputs, tuple)
    return tuple(map(Wrapper, outputs))
