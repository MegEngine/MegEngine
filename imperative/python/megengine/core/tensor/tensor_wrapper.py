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

import numpy as np

from .._trace_option import use_tensor_shape
from ..ops import builtin
from ..ops.builtin import GetVarShape
from ..ops.special import Const
from . import utils
from .core import OpBase, TensorBase, TensorWrapperBase, apply
from .indexing import getitem as _getitem
from .indexing import setitem as _setitem
from .raw_tensor import RawTensor, as_raw_tensor
from .tensor import Tensor
from .utils import make_shape_tuple as _make_shape_tuple


def _elwise(*args, mode):
    op = builtin.Elemwise(mode=mode)
    if mode in ("TRUE_DIV", "POW"):
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
    def valid_broadcast(src, tar):
        def failed():
            raise ValueError(
                "the input shape {} can not be broadcasted to target shape {}".format(
                    src, tar
                )
            )

        if isinstance(src, (Tensor, TensorWrapperBase)):
            src = src.numpy()

        if isinstance(tar, (Tensor, TensorWrapperBase)):
            tar = tar.numpy()

        if len(src) > len(tar):
            failed()

        for i in range(min(len(src), len(tar))):
            if src[-i - 1] != 1 and src[-i - 1] != tar[-i - 1]:
                failed()

    valid_broadcast(inp.shape, shape)
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
        op = builtin.Reshape(unspec_axis=unspec_axis)
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
    Param = builtin.AxisAddRemove.Param

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

    param = Param(*map(builtin.AxisAddRemove.AxisDesc.make_remove, axis))
    op = builtin.AxisAddRemove(param=param)
    (result,) = apply(op, inp)
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
        if isinstance(
            args[0],
            (collections.abc.Sequence, TensorBase, TensorWrapperBase, np.ndarray),
        ):
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
        return TensorWrapper(
            as_raw_tensor(array, dtype=array.dtype, device=self.device)
        )

    @abc.abstractmethod
    def _reset(self, other):
        pass

    @abc.abstractproperty
    def dtype(self) -> np.dtype:
        pass

    @abc.abstractproperty
    def shape(self) -> tuple:
        pass

    @abc.abstractmethod
    def numpy(self) -> np.ndarray:
        pass

    __hash__ = None  # due to __eq__ diviates from python convention

    __lt__ = lambda self, value: _elwise(self, value, mode="LT").astype("bool")
    __le__ = lambda self, value: _elwise(self, value, mode="LEQ").astype("bool")
    __gt__ = lambda self, value: _elwise(value, self, mode="LT").astype("bool")
    __ge__ = lambda self, value: _elwise(value, self, mode="LEQ").astype("bool")
    __eq__ = lambda self, value: _elwise(self, value, mode="EQ").astype("bool")
    __ne__ = lambda self, value: _elwise(
        _elwise(self, value, mode="EQ").astype("bool"), mode="NOT"
    )

    __neg__ = _unary_elwise("NEGATE")
    __pos__ = lambda self: self
    __abs__ = _unary_elwise("ABS")
    __invert__ = _logical_unary_elwise("NOT")
    __round__ = _unary_elwise("ROUND")
    __trunc__ = _todo
    __floor__ = _unary_elwise("FLOOR")
    __ceil__ = _unary_elwise("CEIL")

    __add__ = _binary_elwise("ADD")
    __sub__ = _binary_elwise("SUB")
    __mul__ = _binary_elwise("MUL")
    __matmul__ = lambda self, other: _matmul(self, other)
    __truediv__ = _binary_elwise("TRUE_DIV")
    __floordiv__ = _binary_elwise("FLOOR_DIV")
    __mod__ = _binary_elwise("MOD")
    # __divmode__
    __pow__ = _binary_elwise("POW")
    __lshift__ = _binary_elwise("SHL")
    __rshift__ = _binary_elwise("SHR")
    __and__ = _logical_binary_elwise("AND")
    __or__ = _logical_binary_elwise("OR")
    __xor__ = _logical_binary_elwise("XOR")

    __radd__ = _binary_elwise("ADD", rev=1)
    __rsub__ = _binary_elwise("SUB", rev=1)
    __rmul__ = _binary_elwise("MUL", rev=1)
    __rmatmul__ = lambda self, other: _matmul(other, self)
    __rtruediv__ = _binary_elwise("TRUE_DIV", rev=1)
    __rfloordiv__ = _binary_elwise("FLOOR_DIV", rev=1)
    __rmod__ = _binary_elwise("MOD", rev=1)
    # __rdivmode__
    __rpow__ = _binary_elwise("POW", rev=1)
    __rlshift__ = _binary_elwise("SHL", rev=1)
    __rrshift__ = _binary_elwise("SHR", rev=1)
    __rand__ = _logical_binary_elwise("AND", rev=1)
    __ror__ = _logical_binary_elwise("OR", rev=1)
    __rxor__ = _logical_binary_elwise("XOR", rev=1)

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
        shape = self.shape
        if use_tensor_shape():
            shape = shape.numpy()
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
        shape = self.shape
        # XXX: assume ndim is not changed during trace
        if isinstance(shape, self.__class__):
            shape = shape.numpy()
        return len(shape)

    @property
    def size(self):
        if use_tensor_shape():
            return self.shape.prod()
        return np.prod(self.shape).item()

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

    def broadcast(self, *args):
        return _broadcast(self, _expand_args(args))

    def transpose(self, *args):
        if not args:
            args = range(self.ndim)[::-1]
        return _transpose(self, _expand_args(args))

    def flatten(self):
        return self.reshape(-1)

    def sum(self, axis=None, keepdims: bool = False):
        r"""Returns the sum of each row of the input tensor in the given dimension ``axis``.
        If ``axis`` is a list of axises, reduce over all of them.

        If ``keepdims`` is ``True``, the shape of output tensor is the same as the input tensor, except in the dimension(s) ``axis`` where it is of size 1. Otherwise, ``axis`` is squeezed(see :meth:`~.functional.tensor.remove_axis`).

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

            [2]
            [10.]

        """
        return _reduce("SUM")(self, axis, keepdims)

    prod = _reduce("PRODUCT")
    min = _reduce("MIN")
    max = _reduce("MAX")
    mean = _reduce("MEAN")


class GenericTensorWrapper(ArrayMethodMixin, TensorWrapperBase):
    def __init__(self, data):
        self.__wrapped__ = data

    def _reset(self, other):
        if not isinstance(other, __class__):
            raise TypeError(type(other))
        self.__wrapped__ = other.__wrapped__
        return self

    @property
    def dtype(self):
        return self.__wrapped__.dtype

    @property
    def shape(self):
        if use_tensor_shape():
            return apply(GetVarShape(), self)[0]
        else:
            return self.__wrapped__.shape

    @property
    def device(self):
        return self.__wrapped__.device

    def numpy(self):
        return self.__wrapped__.numpy()


class TensorWrapper(GenericTensorWrapper):
    def __init__(self, data, dtype=None, device=None):
        if isinstance(data, TensorWrapperBase):
            data = data.__wrapped__
        elif not isinstance(data, TensorBase):
            assert data is not None, "Cannot init a tensor with data as None"
            data = Tensor(as_raw_tensor(data, dtype=dtype, device=device))
        super().__init__(data)

    def _reset(self, other):
        if isinstance(other, TensorWrapperBase):
            self.__wrapped__ = other.__wrapped__
        elif isinstance(other, TensorBase):
            self.__wrapped__ = other
        else:
            self._reset(type(self)(other, dtype=self.dtype, device=self.device))

    def __repr__(self):
        piece = "Tensor("
        with np.printoptions(precision=4, suppress=True):
            piece += "{}".format(str(self.numpy()))
        if self.dtype != np.float32:
            piece += ", dtype={}".format(np.dtype(self.dtype).name)
        piece += ", device={}".format(self.device) + ")"
        return piece
