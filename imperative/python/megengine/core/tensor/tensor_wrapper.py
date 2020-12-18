# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import abc
import collections
from typing import Union

import numpy as np

from .._imperative_rt.common import CompNode
from .._imperative_rt.core2 import Tensor, apply
from .._trace_option import use_symbolic_shape
from ..ops import builtin
from ..ops.builtin import Elemwise, GetVarShape
from ..ops.special import Const
from . import utils
from .indexing import getitem as _getitem
from .indexing import setitem as _setitem
from .utils import isscalar
from .utils import make_shape_tuple as _make_shape_tuple
from .utils import setscalar

_ElwMod = Elemwise.Mode


def _elwise(*args, mode):
    op = builtin.Elemwise(mode)
    if mode in (_ElwMod.TRUE_DIV, _ElwMod.POW):
        args = tuple(
            map(
                lambda x: x.astype("float32")
                if hasattr(x, "dtype") and x.dtype != np.float32
                else x,
                args,
            )
        )
    args = utils.convert_inputs(*args)
    (result,) = apply(op, *args)

    _isscalar = True
    for i in args:
        if isscalar(i) == False:
            _isscalar = False
            break
    if _isscalar:
        setscalar(result)
    return result


def _matmul(inp1, inp2):
    op = builtin.MatrixMul(
        transposeA=False, transposeB=False, compute_mode="DEFAULT", format="DEFAULT"
    )
    inp1, inp2 = utils.convert_inputs(inp1, inp2)
    (result,) = apply(op, inp1, inp2)
    return result


def _transpose(data, axes):
    op = builtin.Dimshuffle(axes)
    (data,) = utils.convert_inputs(data)
    (result,) = apply(op, data)
    return result


def _broadcast(inp, shape):
    shape = utils.astensor1d(shape, inp, dtype="int32", device=inp.device)
    (result,) = apply(builtin.Broadcast(), inp, shape)
    return result


def _reshape(x, shape):
    shape_tuple = _make_shape_tuple(shape)
    unspec_axis = None
    # XXX: assume unspec_axis is not changed in trace
    for i, s in enumerate(shape_tuple):
        if s < 0:
            if s != -1:
                raise ValueError("expect shape[{}] >= -1, got {}".format(i, s))
            if unspec_axis is not None:
                raise ValueError("multiple -1 in shape: {} & {}".format(unspec_axis, i))
            unspec_axis = i
    shape = utils.astensor1d(shape, x, dtype="int32", device=x.device)
    if unspec_axis is None:
        op = builtin.Reshape()
    else:
        op = builtin.Reshape(axis=unspec_axis)
    (x,) = apply(op, x, shape)
    return x


def _unary_elwise(mode):
    def f(self):
        return _elwise(self, mode=mode)

    return f


def _binary_elwise(mode, rev=False):
    if not rev:

        def f(self, value):
            return _elwise(self, value, mode=mode)

    else:

        def f(self, value):
            return _elwise(value, self, mode=mode)

    return f


def _logical_unary_elwise(mode, rev=False):
    def f(self):
        if self.dtype != np.bool_:
            raise TypeError("{} requires a bool tensor".format(mode))
        return _elwise(self, mode=mode)

    return f


def _logical_binary_elwise(mode, rev=False):
    if not rev:

        def f(self, value):
            if self.dtype != np.bool_ or value.dtype != np.bool_:
                raise TypeError("{} requires 2 bool tensors".format(mode))
            return _elwise(self, value, mode=mode)

    else:

        def f(self, value):
            if self.dtype != np.bool_ or value.dtype != np.bool_:
                raise TypeError("{} requires 2 bool tensors".format(mode))
            return _elwise(value, self, mode=mode)

    return f


def _remove_axis(inp: Tensor, axis) -> Tensor:
    def get_axes():
        if axis is None:
            return [i for i, s in enumerate(inp.shape) if s == 1]
        try:
            return [int(axis)]
        except (TypeError, ValueError):
            pass
        return list(map(int, axis))

    axis = get_axes()
    axis = sorted(i + inp.ndim if i < 0 else i for i in axis)
    axis = [a - i for i, a in enumerate(axis)]

    op = builtin.RemoveAxis(axis=axis)
    (result,) = apply(op, inp)
    if len(axis) == inp.ndim:
        setscalar(result)
    return result


def _reduce(mode):
    def f(self, axis=None, keepdims: bool = False):
        data = self
        (data,) = utils.convert_inputs(data)
        if mode == "MEAN":
            data = data.astype("float32")
        elif self.dtype == np.bool_:
            data = data.astype("int32")
        if axis is None:
            data = data.reshape(-1)
            assert not keepdims, "can not set axis=None and keepdims=True"

            op = builtin.Reduce(mode=mode, axis=0)
            (result,) = apply(op, data)
        elif isinstance(axis, collections.abc.Iterable):
            axis = list(axis)
            axis.sort(reverse=True)
            for ai in axis:
                op = builtin.Reduce(mode=mode, axis=ai)
                (data,) = apply(op, data)
                if not keepdims:
                    data = _remove_axis(data, ai)
            result = data
        else:
            op = builtin.Reduce(mode=mode, axis=axis)
            (result,) = apply(op, data)

            if not keepdims:
                result = _remove_axis(result, axis)
        if self.dtype == np.bool_:
            if mode in ["MIN", "MAX"]:
                result = result.astype("bool")
        if axis is None or self.ndim == 1:
            setscalar(result)
        return result

    return f


def _inplace(f):
    def g(self, value):
        result = f(self, value)
        if result is NotImplemented:
            raise NotImplementedError
        self._reset(result)
        return self

    return g


def _todo(*_):
    raise NotImplementedError


def _expand_args(args):
    if len(args) == 1:
        if isinstance(args[0], (collections.abc.Sequence, Tensor, np.ndarray),):
            args = args[0]
    return args


class ArrayMethodMixin(abc.ABC):

    # enable tensor to be converted to numpy array
    __array_priority__ = 1001

    def __array__(self, dtype=None):
        if dtype == None:
            return self.numpy()
        return self.numpy().astype(dtype)

    def __array_wrap__(self, array):
        Wrapper = type(self)
        return Wrapper(array, dtype=array.dtype, device=self.device)

    @abc.abstractmethod
    def _reset(self, other):
        pass

    @abc.abstractproperty
    def dtype(self) -> np.dtype:
        pass

    @abc.abstractproperty
    def shape(self) -> Union[tuple, Tensor]:
        pass

    @abc.abstractproperty
    def _tuple_shape(self) -> tuple:
        pass

    @abc.abstractmethod
    def numpy(self) -> np.ndarray:
        pass

    __hash__ = None  # due to __eq__ diviates from python convention

    __lt__ = lambda self, value: _elwise(self, value, mode=_ElwMod.LT).astype("bool")
    __le__ = lambda self, value: _elwise(self, value, mode=_ElwMod.LEQ).astype("bool")
    __gt__ = lambda self, value: _elwise(value, self, mode=_ElwMod.LT).astype("bool")
    __ge__ = lambda self, value: _elwise(value, self, mode=_ElwMod.LEQ).astype("bool")
    __eq__ = lambda self, value: _elwise(self, value, mode=_ElwMod.EQ).astype("bool")
    __ne__ = lambda self, value: _elwise(
        _elwise(self, value, mode=_ElwMod.EQ).astype("bool"), mode=_ElwMod.NOT,
    )

    __neg__ = _unary_elwise(_ElwMod.NEGATE)
    __pos__ = lambda self: self
    __abs__ = _unary_elwise(_ElwMod.ABS)
    __invert__ = _logical_unary_elwise(_ElwMod.NOT)
    __round__ = _unary_elwise(_ElwMod.ROUND)
    __trunc__ = _todo
    __floor__ = _unary_elwise(_ElwMod.FLOOR)
    __ceil__ = _unary_elwise(_ElwMod.CEIL)

    __add__ = _binary_elwise(_ElwMod.ADD)
    __sub__ = _binary_elwise(_ElwMod.SUB)
    __mul__ = _binary_elwise(_ElwMod.MUL)
    __matmul__ = lambda self, other: _matmul(self, other)
    __truediv__ = _binary_elwise(_ElwMod.TRUE_DIV)
    __floordiv__ = _binary_elwise(_ElwMod.FLOOR_DIV)
    __mod__ = _binary_elwise(_ElwMod.MOD)
    # __divmode__
    __pow__ = _binary_elwise(_ElwMod.POW)
    __lshift__ = _binary_elwise(_ElwMod.SHL)
    __rshift__ = _binary_elwise(_ElwMod.SHR)
    __and__ = _logical_binary_elwise(_ElwMod.AND)
    __or__ = _logical_binary_elwise(_ElwMod.OR)
    __xor__ = _logical_binary_elwise(_ElwMod.XOR)

    __radd__ = _binary_elwise(_ElwMod.ADD, rev=1)
    __rsub__ = _binary_elwise(_ElwMod.SUB, rev=1)
    __rmul__ = _binary_elwise(_ElwMod.MUL, rev=1)
    __rmatmul__ = lambda self, other: _matmul(other, self)
    __rtruediv__ = _binary_elwise(_ElwMod.TRUE_DIV, rev=1)
    __rfloordiv__ = _binary_elwise(_ElwMod.FLOOR_DIV, rev=1)
    __rmod__ = _binary_elwise(_ElwMod.MOD, rev=1)
    # __rdivmode__
    __rpow__ = _binary_elwise(_ElwMod.POW, rev=1)
    __rlshift__ = _binary_elwise(_ElwMod.SHL, rev=1)
    __rrshift__ = _binary_elwise(_ElwMod.SHR, rev=1)
    __rand__ = _logical_binary_elwise(_ElwMod.AND, rev=1)
    __ror__ = _logical_binary_elwise(_ElwMod.OR, rev=1)
    __rxor__ = _logical_binary_elwise(_ElwMod.XOR, rev=1)

    __iadd__ = _inplace(__add__)
    __isub__ = _inplace(__sub__)
    __imul__ = _inplace(__mul__)
    __imatmul__ = _inplace(__matmul__)
    __itruediv__ = _inplace(__truediv__)
    __ifloordiv__ = _inplace(__floordiv__)
    __imod__ = _inplace(__mod__)
    __ipow__ = _inplace(__pow__)
    __ilshift__ = _inplace(__lshift__)
    __irshift__ = _inplace(__rshift__)
    __iand__ = _inplace(__and__)
    __ior__ = _inplace(__or__)
    __ixor__ = _inplace(__xor__)

    __index__ = lambda self: self.item().__index__()
    __bool__ = lambda self: bool(self.item())
    __int__ = lambda self: int(self.item())
    __float__ = lambda self: float(self.item())
    __complex__ = lambda self: complex(self.item())

    def __len__(self):
        shape = self._tuple_shape
        if shape:
            return int(shape[0])
        raise TypeError("ndim is 0")

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        return _getitem(self, index)

    def __setitem__(self, index, value):
        if index is not Ellipsis:
            value = _setitem(self, index, value)
        self._reset(value)

    __contains__ = _todo

    @property
    def ndim(self):
        shape = self._tuple_shape
        if shape is None:
            raise ValueError("unkown ndim")
        return len(shape)

    @property
    def size(self):
        shape = self.shape
        if shape.__class__ is tuple:
            return np.prod(self.shape).item()
        return shape.prod()

    @property
    def T(self):
        return self.transpose()

    def item(self, *args):
        if not args:
            if isinstance(self.size, int):
                assert self.size == 1
            return self.numpy().item()
        return self[args].item()

    def tolist(self):
        return self.numpy().tolist()

    def astype(self, dtype):
        return utils.astype(self, dtype)

    def reshape(self, *args):
        return _reshape(self, _expand_args(args))

    # FIXME: remove this method
    def _broadcast(self, *args):
        return _broadcast(self, _expand_args(args))

    def transpose(self, *args):
        if self.ndim == 0:
            assert (
                len(args) == 0
            ), "transpose for scalar does not accept additional args"
            ret = self.to(self.device)
            setscalar(ret)
            return ret
        if not args:
            args = range(self.ndim)[::-1]
        return _transpose(self, _expand_args(args))

    def flatten(self):
        return self.reshape(-1)

    def sum(self, axis=None, keepdims: bool = False):
        r"""
        Returns the sum of each row of the input tensor in the given dimension ``axis``.
        If ``axis`` is a list of axises, reduce over all of them.

        If ``keepdims`` is ``True``, the shape of output tensor is the same as the input tensor, except in the dimension(s) ``axis`` where it is of size 1. Otherwise, ``axis`` is squeezed(see :meth:`~.functional.tensor.squeeze`).

        Same for prod/mean/max/min.

        :param axis: the dimension or dimensions to reduce.
        :param keepdim: whether the output tensor has ndim retained or not.
        :return: output tensor.

        Examples:

        .. testcode::

            from megengine import tensor
            a = tensor([False, True, True, False])
            b = tensor([1.0, 2.0, 3.0, 4.0])
            print(a.sum().numpy())
            print(b.sum().numpy())

        Outputs:

        .. testoutput::

            2
            10.

        """
        return _reduce("SUM")(self, axis, keepdims)

    prod = _reduce("PRODUCT")
    min = _reduce("MIN")
    max = _reduce("MAX")
    mean = _reduce("MEAN")
