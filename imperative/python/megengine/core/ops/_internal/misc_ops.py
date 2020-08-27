# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import collections
import ctypes

from ..._imperative_rt import OperatorNodeConfig as Config
from . import param_defs
from .helper import PodOpVisitor, make_param

__all__ = ["ConvolutionBackwardData", "Dimshuffle", "Reshape", "AxisAddRemove"]


class TensorShape:
    MAX_NDIM = 7


class ConvolutionBackwardData(PodOpVisitor):
    param_names = (
        "param",
        "execution_polity",
    )
    name = "ConvolutionBackwardDataV1"

    def __init__(
        self,
        *,
        param=None,
        execution_polity=None,
        name=None,
        comp_node=None,
        config=None,
        dtype=None,
        **kwargs
    ):
        config = config or Config()
        if name:
            config.name = name
        if comp_node:
            config.comp_node = comp_node
        if dtype:
            config.dtype = dtype
        self.config = config

        self.param = make_param(param, param_defs.Convolution, kwargs)
        self.execution_polity = make_param(
            execution_polity, param_defs.ExecutionPolicy, kwargs
        )
        assert not kwargs, "extra kwargs: {}".format(kwargs)


class Dimshuffle(PodOpVisitor):
    name = "Dimshuffle"
    param_names = ("pattern",)

    class Pattern(ctypes.Structure):
        Pattern_Array = ctypes.c_int32 * TensorShape.MAX_NDIM
        _fields_ = [
            ("length", ctypes.c_uint32),
            ("pattern", Pattern_Array),
            ("ndim", ctypes.c_uint32),
        ]

        def serialize(self):
            return bytes(ctypes.c_uint32(0)) + bytes(self)

    def __init__(self, pattern, ndim=0):
        assert isinstance(pattern, collections.Iterable)
        assert len(pattern) <= TensorShape.MAX_NDIM
        pattern_array = Dimshuffle.Pattern.Pattern_Array()
        for idx, v in enumerate(pattern):
            pattern_array[idx] = ctypes.c_int32(-1 if v == "x" else int(v))
        self.pattern = Dimshuffle.Pattern(len(pattern), pattern_array, ndim)


class Reshape(PodOpVisitor):
    name = "ReshapeV1"
    param_names = ("unspec_axis",)

    def __init__(self, unspec_axis=None):
        if unspec_axis is None:
            self.unspec_axis = param_defs.OptionalAxisV1()
        else:
            self.unspec_axis = param_defs.OptionalAxisV1(unspec_axis)


class AxisNum(ctypes.Structure):
    _fields_ = [
        ("m_num", ctypes.c_int),
    ]


class AxisDesc(ctypes.Structure):
    class Method(ctypes.c_int):
        ADD_1 = 0
        REMOVE = 1

    _fields_ = [
        ("method", Method),
        ("axis", AxisNum),
    ]

    @classmethod
    def make_add(cls, axis):
        return cls(cls.Method.ADD_1, AxisNum(axis))

    @classmethod
    def make_remove(cls, axis):
        return cls(cls.Method.REMOVE, AxisNum(axis))


class AxisAddRemove(PodOpVisitor):
    name = "AxisAddRemove"
    param_names = ("param",)

    AxisDesc = AxisDesc

    class Param(ctypes.Structure):
        MAX_DESC_SIZE = TensorShape.MAX_NDIM * 2

        _fields_ = [("nr_desc", ctypes.c_uint32), ("desc", AxisDesc * MAX_DESC_SIZE)]

        def __init__(self, *args):
            super().__init__()
            self.nr_desc = len(args)
            for i, a in enumerate(args):
                self.desc[i] = a

        def serialize(self):
            return bytes(ctypes.c_uint32(0)) + bytes(self)

    def __init__(self, param):
        assert isinstance(param, self.Param)
        self.param = param


del AxisDesc


class IndexingOpBase(PodOpVisitor):
    param_names = ("index_desc",)

    class IndexDescMaskDump(ctypes.Structure):
        class Item(ctypes.Structure):
            _fields_ = [
                ("axis", ctypes.c_int8),
                ("begin", ctypes.c_bool),
                ("end", ctypes.c_bool),
                ("step", ctypes.c_bool),
                ("idx", ctypes.c_bool),
            ]

        Item_Array = Item * TensorShape.MAX_NDIM

        _fields_ = [("nr_item", ctypes.c_uint8), ("items", Item_Array)]

        def serialize(self):
            return bytes(ctypes.c_uint32(0)) + bytes(self)

    def __init__(self, items):
        nr_item = len(items)
        assert nr_item <= TensorShape.MAX_NDIM
        item_array = IndexingOpBase.IndexDescMaskDump.Item_Array()
        for idx, item in enumerate(items):
            assert isinstance(item, (tuple, list)) and len(item) == 5
            item_array[idx] = IndexingOpBase.IndexDescMaskDump.Item(*item)
        self.index_desc = IndexingOpBase.IndexDescMaskDump(nr_item, item_array)


def _gen_indexing_defs(*names):
    for name in names:
        globals()[name] = type(name, (IndexingOpBase,), dict(name=name))
        __all__.append(name)


_gen_indexing_defs(
    "Subtensor",
    "SetSubtensor",
    "IncrSubtensor",
    "IndexingMultiAxisVec",
    "IndexingSetMultiAxisVec",
    "IndexingIncrMultiAxisVec",
    "MeshIndexing",
    "IncrMeshIndexing",
    "SetMeshIndexing",
    "BatchedMeshIndexing",
    "BatchedIncrMeshIndexing",
    "BatchedSetMeshIndexing",
)
