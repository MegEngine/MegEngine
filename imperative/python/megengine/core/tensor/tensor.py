# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import contextlib
import copy

from .core import Dispatcher, OpBase, TensorBase, apply


class Tensor(TensorBase):
    def __init__(self, data: TensorBase):
        self._data = data
        # _extra_data is set up in Grad.wrt
        self._extra_data = {}
        self._user_data = {}

    def __getattr__(self, name):
        if name in self._user_data:
            return self._user_data[name]
        raise AttributeError(name)

    def reset(self, other):
        assert isinstance(other, __class__)
        self.__dict__.clear()
        self._data = other.data
        self._extra_data = other._extra_data.copy()
        self._user_data = other._user_data.copy()

    def copy(self):
        other = object.__new__(type(self))
        other.reset(self)
        return other

    # tensor interface

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def device(self):
        return self._data.device

    def numpy(self):
        return self._data.numpy()


class ApplyContext:
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.key = None


_context = None


@contextlib.contextmanager
def push_context():
    global _context
    backup = _context
    try:
        _context = ApplyContext()
        yield _context
    finally:
        _context = backup


def get_context():
    return _context


@apply.add
def tensor_apply(op: OpBase, *args: Tensor):
    data = tuple(i._data if isinstance(i, Tensor) else i for i in args)
    # type(Tensor._data) is RawTensor
    # dispached to apply.add@RawTensor.py if passed Tensor args
    outputs = apply(op, *data)
    ret = tuple(map(Tensor, outputs))

    with push_context() as ctx:
        ctx.inputs = args
        ctx.outputs = ret
        for k in set().union(*(i._extra_data for i in args if isinstance(i, Tensor))):
            ctx.key = k
            data = tuple(
                i._extra_data.get(k) if isinstance(i, Tensor) else i for i in args
            )
            # data are instances of Tracer
            # dispatched to apply.add@grad.py
            outputs = apply(op, *data)
            if outputs is not None:
                assert len(outputs) == len(ret)
                for t, i in zip(ret, outputs):
                    t._extra_data[k] = i

    return ret
