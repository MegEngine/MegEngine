# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Union

import numpy as np

from .core._imperative_rt import CompNode
from .core._imperative_rt.core2 import Tensor as _Tensor
from .core._imperative_rt.core2 import apply
from .core._trace_option import use_symbolic_shape
from .core._wrap import device as as_device
from .core.ops.builtin import Copy, GetVarShape
from .core.tensor.array_method import ArrayMethodMixin
from .device import _valid_device, get_default_device
from .logger import get_logger
from .utils.deprecation import deprecated
from .utils.naming import AutoNaming

logger = get_logger(__name__)


class Tensor(_Tensor, ArrayMethodMixin):
    r"""
    A tensor object represents a multidimensional, homogeneous array of fixed-size items.

    :param data: The value of returned Tensor.
    :param dtype: The dtype of returned Tensor. Uses data's dtype if not specified.
    :param device: The desired device of returned Tensor. Uses :func:`get_default_device` if not specified.
    :param is_const: Whether make it a ``ImutableTensor`` in tracing mode.
    :param no_cache: Whether cache it for memory sharing.
    :param name: Used to improve convenience in graph operation on dumped model.
    """

    grad = None
    dmap_callback = None
    _qparams = None

    def __new__(
        cls,
        data: Union["Tensor", np.ndarray, list, "scalar"] = None,
        dtype: np.dtype = None,
        device: str = None,
        is_const: bool = False,
        no_cache: bool = False,
        name: str = None,
    ):
        if data is None:
            data = []
        if device is None:
            cn = get_default_device()
        elif isinstance(device, str):
            if cls.dmap_callback is not None:
                cn = CompNode(cls.dmap_callback(device))
            else:
                cn = CompNode(device)
        else:
            if isinstance(device, CompNode):
                cn = device
            else:
                cn = device._cn

        if isinstance(data, _Tensor):
            obj = _Tensor.__new__(cls, data)
        else:
            if isinstance(data, np.ndarray):
                if 0 in data.strides:
                    data = data.squeeze().reshape(data.shape)
            obj = _Tensor.__new__(cls, data, dtype, cn, is_const, no_cache, name)
        return obj

    def __init__(
        self,
        data: Union["Tensor", np.ndarray, list, "scalar"],
        dtype: np.dtype = None,
        device: str = None,
        is_const: bool = False,
        no_cache: bool = False,
        name: str = None,
    ):
        pass

    @property
    def shape(self) -> Union[tuple, "Tensor"]:
        r"""
        Returns a :class:`tuple` or a :class:`~.Tensor` represents tensor dimensions.

        .. note::

           The shape of a tensor was usually represented by a :class:`tuple`.
           But if a tensor was treated as symbolic placeholder with tracing, 
           it's shape could also be a :class:`~.Tensor`. See :class:`~.trace` for more details.

        The shape property is usually used to get the current shape of a tensor, 
        but may also be used to reshape the tensor in-place by assigning a tuple of tensor dimensions to it. 
        As with :func:`~.reshape`, one of the new shape dimensions can be -1, 
        in which case its value is inferred from the size of the tensor and the remaining dimensions.
        """
        shape = super().shape
        if shape == () or not use_symbolic_shape():
            return shape
        return apply(GetVarShape(), self)[0]

    @property
    def _tuple_shape(self):
        return super().shape

    @property
    def device(self) -> CompNode:
        r"""
        Returns a string represents the device a :class:`~.Tensor` storaged on. 
        """
        return super().device

    @property
    def dtype(self) -> np.dtype:
        r"""
        Returns a :class:`numpy.dtype` object represents the data type of a :class:`~.Tensor`.
        """
        return super().dtype

    @property
    def qparams(self):
        r"""
        Returns a :class:`~.QParams` object containing quantization params of a :class:`~.Tensor`.
        """
        from .quantization.utils import create_qparams  # pylint: disable=all

        if self._qparams is None:
            self._qparams = create_qparams()
        return self._qparams

    def numpy(self) -> np.ndarray:
        r"""
        Returns self :class:`~.Tensor` as a :class:`numpy.ndarray`.
        """
        return super().numpy()

    def detach(self):
        r"""
        Returns a new :class:`~.Tensor`, detached from the current graph.
        """
        return super().detach()

    def _reset(self, other):
        if not isinstance(other, _Tensor):
            other = Tensor(other, dtype=self.dtype, device=self.device)
        super()._reset(other)

    def __repr__(self):
        piece = "{}(".format(self.__class__.__name__)
        with np.printoptions(precision=4, suppress=True):
            piece += "{}".format(str(self.numpy()))
        if self.dtype != np.float32:
            piece += ", dtype={}".format(np.dtype(self.dtype).name)
        piece += ", device={}".format(self.device) + ")"
        return piece

    @property
    def name(self):
        return self.c_name

    @name.setter
    def name(self, name):
        self.c_name = name
        AutoNaming.record_var_name(self._mixin_handle, name)

    @deprecated(version="1.0", reason="no need to reuse an existing tensor since 1.0")
    def set_value(self, value):
        self._reset(value)

    @deprecated(version="1.0", reason="use *= 0 instead")
    def reset_zero(self):
        self *= 0

    def to(self, device):
        r"""
        Copy self :class:`~.Tensor` to specified device. See :func:`~.copy`
        """
        if isinstance(device, str) and not _valid_device(device):
            raise ValueError(
                "invalid device name {}. For the correct format of the device name, please refer to the instruction of megengine.device.set_default_device()".format(
                    device
                )
            )
        cn = as_device(device).to_c()
        return apply(Copy(comp_node=cn), self)[0]

    @property
    def requires_grad(self):
        raise AttributeError("requires_grad is reserved for future use")

    @requires_grad.setter
    def requires_grad(self, value):
        raise AttributeError("requires_grad is reserved for future use")

    @requires_grad.deleter
    def requires_grad(self):
        raise AttributeError("requires_grad is reserved for future use")

    def __hash__(self):
        return id(self)

    def __getnewargs__(self):
        r""" __getnewargs__ will be called for pickle serialization or deep copy
        """
        return (self.numpy(), self.dtype, self.device.logical_name)

    def __getstate__(self):
        r""" __getstate__ will be called for pickle serialization or deep copy
        """
        state = {}
        if self._qparams is not None:
            state["qparams"] = self._qparams
        return state

    def __setstate__(self, state):
        # for compatibility with old version not using fastcore
        if "data" in state:
            data = state.pop("data")
            device = state.pop("device")
            dtype = state.pop("dtype")
            self._reset(Tensor(data, dtype=dtype, device=device))

        # quantize related state for deepcopy
        if "qdict" in state:
            qparams = state.pop("qdict")
            logger.warning(
                "Tensor's 'qdict' state is depreciated. Use 'qparams' instead"
            )
        elif "qparams" in state:
            qparams = state.pop("qparams")
        else:
            qparams = None
        self._qparams = qparams


tensor = Tensor


class Parameter(Tensor):
    r"""
    A kind of Tensor that is to be considered a module parameter.
    """
