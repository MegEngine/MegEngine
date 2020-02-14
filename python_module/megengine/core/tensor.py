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
import itertools
from typing import Union

import numpy as np

import megengine._internal as mgb

from .graph import _use_default_if_none, get_default_graph


def wrap_io_tensor(func):
    r"""A wrapper to make ``func`` compatible with functions in ``_internal.opr``.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        comp_graph = None
        for i in itertools.chain(args, kwargs.values()):
            if isinstance(i, Tensor) and i._comp_graph:
                comp_graph = i._comp_graph
                break
        else:

            comp_graph = get_default_graph()
        new_args = (
            arg._attach(comp_graph) if isinstance(arg, Tensor) else arg for arg in args
        )
        new_kwargs = {
            k: v._attach(comp_graph) if isinstance(v, Tensor) else v
            for k, v in kwargs.items()
        }
        ret = func(*new_args, **new_kwargs)
        if isinstance(ret, mgb.SymbolVar):
            ret = Tensor(ret)
        elif isinstance(ret, list):
            ret = [Tensor(t) if isinstance(t, mgb.SymbolVar) else t for t in ret]
        elif isinstance(ret, tuple):
            ret = tuple(Tensor(t) if isinstance(t, mgb.SymbolVar) else t for t in ret)
        return ret

    return wrapper


def _wrap_symbolvar_binary_op(f):
    @functools.wraps(f)
    def wrapped(self, other):
        comp_graph = (
            isinstance(other, Tensor)
            and other._comp_graph
            or self._comp_graph
            or get_default_graph()
        )
        if isinstance(other, Tensor):
            other = other._attach(comp_graph)
        return Tensor(f(self._attach(comp_graph), other))

    return wrapped


def wrap_slice(inp):
    start = inp.start._symvar if isinstance(inp.start, Tensor) else inp.start
    stop = inp.stop._symvar if isinstance(inp.stop, Tensor) else inp.stop
    step = inp.step._symvar if isinstance(inp.step, Tensor) else inp.step
    return slice(start, stop, step)


def wrap_idx(idx):
    if not isinstance(idx, tuple):
        idx = (idx,)

    idx = tuple(i._symvar if isinstance(i, Tensor) else i for i in idx)
    idx = tuple(wrap_slice(i) if isinstance(i, slice) else i for i in idx)
    return idx


class MGBIndexWrapper:
    def __init__(self, dest, mgb_index, val=None):
        self.dest = dest
        self.val = val
        self.mgb_index = mgb_index

    def __getitem__(self, idx):
        if self.val is None:
            return wrap_io_tensor(self.mgb_index(self.dest._symvar).__getitem__)(
                wrap_idx(idx)
            )
        else:
            return wrap_io_tensor(
                self.mgb_index(self.dest._symvar, self.val._symvar).__getitem__
            )(wrap_idx(idx))


class Tensor:
    r"""The main data container in MegEngine.
    Use :func:`~.tensor` to create a Tensor with existed data.
    """
    requires_grad = False
    grad = None

    def __init__(self, val=None, *, requires_grad=None):
        self._reset(val, requires_grad=requires_grad)

    def _reset(self, val=None, *, requires_grad=None):
        if val is None:
            self.__val = None
            self.__sym = None
        elif isinstance(val, mgb.SharedND):
            self.__val = val
            self.__sym = None
        elif isinstance(val, mgb.SymbolVar):
            self.__val = None
            self.__sym = val
        else:
            raise TypeError("must be initialized with SymbolVar or SharedND")
        self.requires_grad = requires_grad

    def _as_tensor(self, obj):
        r"""Convert the data into a ``Tensor``. If the data is already a Tensor
        with the same dtype and device, no copy will be performed. Otherwise a
        new Tensor will be returned with computational graph retained.

        """
        if isinstance(obj, Tensor):
            return obj
        if isinstance(obj, mgb.SymbolVar):
            return Tensor(obj)
        if isinstance(obj, mgb.SharedScalar):
            return Tensor(obj._as_sym_var(self._comp_graph, self._comp_node))
        return tensor(data=obj, device=self.device)

    def numpy(self):
        r"""Return the tensor value in ndarray format.
        """
        if self.__val is not None:
            assert self.__sym is None
            return self.__val.get_value()
        if self.__sym is None:
            raise ValueError("uninitialized")
        if self.__sym.eager_val is not None:
            return self.__sym.eager_val.get_value()
        return self.__sym.inferred_value

    def item(self):
        return self.numpy().item()

    def _attach(self, comp_graph, *, volatile=True):
        if self.__val:
            return self.__val.symvar(comp_graph, volatile=volatile)
        if self.__sym:
            if self.__sym.owner_graph != comp_graph:
                raise RuntimeError("internal error")
            return self.__sym
        else:
            raise ValueError("uninitialized")

    @property
    def _symvar(self):
        if self.__sym:
            assert not self.__val
            return self.__sym
        if not self.__val:
            raise ValueError("uninitialized")

        return self._attach(get_default_graph())

    def __mgb_symvar__(self, comp_graph=None, **_):
        if self.__val and comp_graph:
            return self._attach(comp_graph)
        return self._symvar  # read by mgb.opr

    @property
    def dtype(self):
        r"""Return the data type of the tensor.
        """
        if self.__val is not None:
            return self.__val.dtype
        return self._symvar.dtype

    @property
    def _comp_node(self):
        if self.__val is not None:
            return self.__val.comp_node
        return self._symvar.comp_node

    device = _comp_node

    @property
    def _comp_graph(self):
        if self.__sym is not None:
            return self.__sym.owner_graph
        return None

    @property
    def shape(self):
        r"""Return an int tuple that is the shape/layout of the tensor.
        Could be invalid in static graph mode.
        """
        from ..jit import trace

        if trace._active_instance:  # pylint: disable=protected-access
            # NOTE: this is an hack
            shape = mgb.opr.get_var_shape(self._symvar)
            return tuple(Tensor(shape[i]) for i in range(self.ndim))
        return self._symvar.imm_shape

    def set_value(self, value, *, sync=True, inplace=False, share=False):
        r"""Set value to the tensor.
        """
        if not self.__val:
            raise ValueError("not detached")
        if isinstance(value, Tensor):
            value = value.__val or value.__sym.eager_val
        self.__val.set_value(value, sync=sync, inplace=inplace, share=share)

    def fill(self, value):
        r"""Fills the tensor with the specified value.
        """
        self.set_value(np.full(self.shape, value, dtype=self.dtype))

    def reset_zero(self):
        r"""Reset the tensor and fills with zeros.
        """
        if not self.__val:
            raise ValueError("not detached")
        self.__val.reset_zero()

    def to(self, device):
        return wrap_io_tensor(mgb.opr.copy)(self, comp_node=device)

    # https://docs.python.org/3/reference/datamodel.html#object.__hash__
    # > If a class does not define an __eq__() method it should not define a
    # > __hash__() operation either
    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, rhs):
        rhs = self._as_tensor(rhs)
        return Tensor(self._symvar._binary_opr("EQ", rhs._symvar))

    def __ne__(self, rhs):
        return 1 - self.__eq__(rhs)

    def __len__(self):
        if self._symvar.eager_val is not None:
            return self._symvar.eager_val.shape[0]
        raise TypeError(
            "__len__ and __iter__ is not available for tensors on non eager graph."
        )

    __add__ = _wrap_symbolvar_binary_op(mgb.SymbolVar.__add__)
    __radd__ = _wrap_symbolvar_binary_op(mgb.SymbolVar.__radd__)
    __sub__ = _wrap_symbolvar_binary_op(mgb.SymbolVar.__sub__)
    __rsub__ = _wrap_symbolvar_binary_op(mgb.SymbolVar.__rsub__)
    __mul__ = _wrap_symbolvar_binary_op(mgb.SymbolVar.__mul__)
    __rmul__ = _wrap_symbolvar_binary_op(mgb.SymbolVar.__rmul__)
    __matmul__ = _wrap_symbolvar_binary_op(mgb.SymbolVar.__matmul__)
    __rmatmul__ = _wrap_symbolvar_binary_op(mgb.SymbolVar.__rmatmul__)
    __lshift__ = _wrap_symbolvar_binary_op(mgb.SymbolVar.__lshift__)
    __rshift__ = _wrap_symbolvar_binary_op(mgb.SymbolVar.__rshift__)
    __truediv__ = _wrap_symbolvar_binary_op(mgb.SymbolVar.__truediv__)
    __rtruediv__ = _wrap_symbolvar_binary_op(mgb.SymbolVar.__rtruediv__)
    __floordiv__ = _wrap_symbolvar_binary_op(mgb.SymbolVar.__floordiv__)
    __rfloordiv__ = _wrap_symbolvar_binary_op(mgb.SymbolVar.__rfloordiv__)
    __mod__ = _wrap_symbolvar_binary_op(mgb.SymbolVar.__mod__)
    __rmod__ = _wrap_symbolvar_binary_op(mgb.SymbolVar.__rmod__)
    __pow__ = _wrap_symbolvar_binary_op(mgb.SymbolVar.__pow__)
    __rpow__ = _wrap_symbolvar_binary_op(mgb.SymbolVar.__rpow__)
    __lt__ = _wrap_symbolvar_binary_op(mgb.SymbolVar.__lt__)
    __gt__ = _wrap_symbolvar_binary_op(mgb.SymbolVar.__gt__)
    __le__ = _wrap_symbolvar_binary_op(mgb.SymbolVar.__le__)
    __ge__ = _wrap_symbolvar_binary_op(mgb.SymbolVar.__ge__)
    __neg__ = wrap_io_tensor(mgb.SymbolVar.__neg__)
    sum = wrap_io_tensor(mgb.SymbolVar.sum)
    """
    Sum up the given tensors.
    """
    max = wrap_io_tensor(mgb.SymbolVar.max)
    """
    Return the maximum value of given tensor.
    """
    min = wrap_io_tensor(mgb.SymbolVar.min)
    """
    Return the minimum value of given tensor.
    """
    prod = wrap_io_tensor(mgb.SymbolVar.prod)
    """
    Return the product value of the given tensor.
    """
    mean = wrap_io_tensor(mgb.SymbolVar.mean)
    """
    Return the mean value of the given tensor.
    """
    dimshuffle = wrap_io_tensor(mgb.SymbolVar.dimshuffle)
    """
    See more details in :func:`~.functional.tensor.dimshuffle`.
    """
    astype = wrap_io_tensor(mgb.SymbolVar.astype)
    """
    Cast the tensor to a specified type.
    """

    def reshape(self, *target_shape):
        r"""Return a tensor which has given target shape

        Examples:

        .. testcode::

            import numpy as np
            from megengine import tensor

            inp = tensor(np.arange(1, 17, dtype=np.int32).reshape(4,4))
            out = tensor(np.arange(100, 116, dtype=np.int32).reshape(1,16))
            out = out.reshape(inp.shape)
            print(out.numpy())

        .. testoutput::

           [[100 101 102 103]
            [104 105 106 107]
            [108 109 110 111]
            [112 113 114 115]]
        """

        if isinstance(target_shape[0], tuple):
            if len(target_shape) > 1:
                raise ValueError("Only single tuple is accepted in reshape")
            target_shape = target_shape[0]
        target_shape = (t._symvar if isinstance(t, Tensor) else t for t in target_shape)
        return Tensor(mgb.SymbolVar.reshape(self._symvar, *target_shape))

    def broadcast(self, *target_shape):
        r"""Return a tesnor broadcasted by current tensor to given target shape

        Examples:

        .. testcode::

            import numpy as np
            from megengine import tensor

            data = tensor(np.arange(100, 104, dtype=np.int32).reshape(1,4))
            data = data.broadcast((4,4))
            print(data.numpy())

        .. testoutput::

            [[100 101 102 103]
             [100 101 102 103]
             [100 101 102 103]
             [100 101 102 103]]
        """

        if isinstance(target_shape[0], tuple):
            if len(target_shape) > 1:
                raise ValueError("Only single tuple is accepted in broadcast")
            target_shape = target_shape[0]
        target_shape = (t._symvar if isinstance(t, Tensor) else t for t in target_shape)
        return Tensor(mgb.SymbolVar.broadcast(self._symvar, *target_shape))

    # Prefer operators on Tensor instead of convert to numpy
    __array_priority__ = 1000

    # mgb indexing family
    def __getitem__(self, idx):
        return wrap_io_tensor(self._symvar.__getitem__)(wrap_idx(idx))

    def set_subtensor(self, val):
        return MGBIndexWrapper(self, mgb.opr.set_subtensor, val)

    def incr_subtensor(self, val):
        return MGBIndexWrapper(self, mgb.opr.incr_subtensor, val)

    @property
    def ai(self):
        return MGBIndexWrapper(self, mgb.opr.advanced_indexing)

    def set_ai(self, val):
        return MGBIndexWrapper(self, mgb.opr.set_advanced_indexing, val)

    def incr_ai(self, val):
        return MGBIndexWrapper(self, mgb.opr.incr_advanced_indexing, val)

    @property
    def mi(self):
        return MGBIndexWrapper(self, mgb.opr.mesh_indexing)

    def set_mi(self, val):
        return MGBIndexWrapper(self, mgb.opr.set_mesh_indexing, val)

    def incr_mi(self, val):
        return MGBIndexWrapper(self, mgb.opr.incr_mesh_indexing, val)

    @property
    def batched_mi(self):
        return MGBIndexWrapper(self, mgb.opr.batched_mesh_indexing)

    def batched_set_mi(self, val):
        return MGBIndexWrapper(self, mgb.opr.batched_set_mesh_indexing, val)

    def batched_incr_mi(self, val):
        return MGBIndexWrapper(self, mgb.opr.batched_incr_mesh_indexing, val)

    def __array__(self, dtype=None):
        if dtype is None:
            return self.numpy()
        else:
            return self.numpy().astype(dtype, copy=False)

    def __int__(self):
        return int(self.item())

    def __index__(self):
        return int(self.item())

    def __round__(self, ndigits=0):
        if ndigits != 0:
            raise ValueError("ndigits must be 0 for Tensor.round")
        return Tensor(mgb.opr.elemwise([self._symvar], mode="ROUND"))

    round = __round__

    def sqrt(self):
        r"""Return a tensor that each element is the square root of its
        original value.

        """
        return Tensor(mgb.opr.sqrt(self._symvar))

    def shapeof(self, axis=None):
        r"""Return a Tensor that represent the shape of the tensor.
        """
        return Tensor(mgb.opr.get_var_shape(self._symvar, axis=axis))

    @property
    def ndim(self):
        r"""Return the number of dimensions of the tensor.
        """
        return len(self._symvar.imm_shape)

    def __repr__(self):
        piece = "Tensor("
        with np.printoptions(precision=4, suppress=True):
            piece += "{}".format(str(self.numpy()))
        if self.dtype != np.float32:
            piece += ", dtype={}".format(np.dtype(self.dtype).name)
        if self._comp_node.locator_logical != ("XPU", -1, 0):
            piece += ", device={}".format(self.device)
        piece += ")"
        return piece

    def __bool__(self):
        raise RuntimeError(
            "Tensor object should not be converted to bool or used in a if statement. Use .numpy(), int() or float() if you want to use its value in if statement, be aware that this may lead to incorrect result in non-eager mode."
        )

    def __getstate__(self):
        assert (self.__val is not None) and (self.__sym is None)
        metadata = {"requires_grad": self.requires_grad}
        state = {
            "data": self.numpy(),
            "device": self.device,
            "dtype": self.dtype,
            "metadata": metadata,
        }
        return state

    def __setstate__(self, state):
        data = state.pop("data")
        device = state.pop("device")
        dtype = state.pop("dtype")
        metadata = state.pop("metadata", {})
        requires_grad = metadata.pop("requires_grad", None)
        snd = mgb.make_shared(device, value=data, dtype=dtype)
        self._reset(snd, requires_grad=requires_grad)


def tensor(
    data: Union[list, np.ndarray] = None,
    *,
    dtype: str = None,
    device: mgb.CompNode = None,
    requires_grad: bool = None
):
    r"""A helper function to create a :class:`~.Tensor` using existing data.

    :param data: an existing data array, must be Python list, NumPy array or None.
    :param dtype: target Tensor data type, one of ``("uint8", "int8", "int16", "int32", "float32", "float16")``.
    :param device: target device for Tensor storing.
    :param requires_grad: whether its gradiant will be calculated during :meth:`~.Optimizer.backward`
    """
    supported_dtypes = ("uint8", "int8", "int16", "int32", "float32", "float16")
    if isinstance(data, Tensor):
        raise NotImplementedError
    if dtype is not None and np.dtype(dtype).name not in supported_dtypes:
        raise TypeError("unsupported dtype {}".format(dtype))
    if data is not None:
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=dtype)
            # In order to accept tensor([1]),
            # Automaticlly convert to  32-bit number instead of numpy's default 64-bit when input data is not nparray.
            dtype = mgb.to_mgb_supported_dtype(data.dtype)
        if dtype is None:
            if data.dtype.name not in supported_dtypes:
                raise TypeError("unsupported dtype {}".format(data.dtype))

    device, _ = _use_default_if_none(device, None)
    shared_nd = mgb.make_shared(device, value=data, dtype=dtype)
    return Tensor(shared_nd, requires_grad=requires_grad)


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
