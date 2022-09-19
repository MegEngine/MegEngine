# -*- coding: utf-8 -*-
from typing import Union

import numpy as np

from .core._imperative_rt import CompNode
from .core._imperative_rt.core2 import FormatType
from .core._imperative_rt.core2 import Tensor as _Tensor
from .core._imperative_rt.core2 import apply, set_py_tensor_type
from .core._trace_option import use_symbolic_shape
from .core._wrap import as_device
from .core.ops.builtin import Borrow, Copy, GetVarShape
from .core.tensor.array_method import ArrayMethodMixin
from .device import _valid_device, get_default_device
from .logger import get_logger
from .utils.deprecation import deprecated

logger = get_logger(__name__)


class Tensor(_Tensor, ArrayMethodMixin):
    r"""A tensor object represents a multidimensional, homogeneous array of fixed-size items.

    Tensor is the primary MegEngine data structure.
    Data type(dtype) describes the format of each element, such as ``float32``, ``int8`` and so on,
    see :ref:`tensor-dtype` for more details.
    It is similar to :class:`numpy.ndarray` but not the same in the design.
    For example, GPU devices can be used to store Tensors and execute calculations in MegEngine.
    The concept of `view <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.view.html>`_
    does not exist in MegEngine so indexing and other behaviors might be different with NumPy.
    All manipulations and operations on/between Tensors could be found in the :mod:`~.megengine.functional` module.
    Keep in mind that they are **not in-place**, a new Tensor will always be returned and
    the original data will remain constant.

    For more information, refer to the :ref:`tensor-guide` topic.

    Args:
        data(Tensor, :class:`~.numpy.ndarray`, :class:`list` or Python number): 
            The data used for construcing Tensor.
            Tensor could be constructed from a Python :class:`list` / :class:`tuple` or sequence;
            a NumPy :class:`~.numpy.ndarray` data structure; MegEngine builtin methods and so on.
            Refer to :ref:`tensor-creation` for more details.

        dtype(:attr:`~.Tensor.dtype`): The data type of returned Tensor. Infer from ``data`` if not specified.
        device(:attr:`~.Tensor.device`): The desired device of returned Tensor. Uses :func:`get_default_device` if not specified.
        is_const: Whether make it a ``ImutableTensor`` in tracing mode, refer to :class:`.jit.trace`.
        no_cache: Whether cache it for memory sharing.
        name: Used to improve convenience in graph operation on dumped model.
        format: Used to indicate which memory format Tensor uses. It will not affect actual memory order or stride,
            but may affect some operators related to indexing and dimension. Only support "default", "nchw" and "nhwc".

    .. note::

       There are some methods like :meth:`~.Tensor.reshape` / :meth:`~.Tensor.flatten` / 
       :meth:`~.Tensor.transpose` / :meth:`~.Tensor.min` / :meth:`~.Tensor.max` /
       :meth:`~.Tensor.mean` / :meth:`~.Tensor.sum` / :meth:`~.Tensor.prod` implemented
       in ``Tensor`` class for convenience and historical reasons.
       But other methods implemented in the :mod:`~.megengine.functional` module will not be added here anymore,
       it is hard for maintaining and too many candidates will affect code completion experience.

    """

    grad = None  #: gradient of this tensor, see :mod:`~.autodiff`.
    dmap_callback = None  #: callback for device mapping, see :func:`~.load`.
    _qparams = None
    _custom_name = ""
    _name = None
    _short_name = None
    _prefix = None

    def __init__(
        self,
        data: Union["Tensor", np.ndarray, list, int, float],
        dtype: np.dtype = None,
        device: str = None,
        is_const: bool = False,
        no_cache: bool = False,
        name: str = None,
        format: str = "default",
    ):
        if name is None:
            name = ""
        else:
            self._set_name(name)
        self._custom_name = name
        self._name = name
        self._short_name = name
        self._prefix = None

    @property
    def shape(self) -> Union[tuple, "Tensor"]:
        r"""Returns a :class:`tuple` or a :class:`~.Tensor` represents tensor dimensions.

        Note:
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
    def device(self):
        r"""Returns a string represents the device a :class:`~.Tensor` storaged on.
        
        .. seealso:: see :ref:`tensor-device` for more details.
        """
        return super().device

    @property
    def dtype(self) -> np.dtype:
        r"""Returns a :class:`numpy.dtype` object represents the data type of a :class:`~.Tensor`.
        
        .. seealso:: see :ref:`tensor-dtype` for more details.
        """
        return super().dtype

    @property
    def format(self) -> str:
        r"""Returns a string represents the :ref:`memory format <format-introduction>` of a :class:`~.Tensor`."""
        return super().format()

    @format.setter
    def format(self, format):
        r"""Sets the memory format of a :class:`~.Tensor`."""
        super()._set_format(format)

    @property
    def qparams(self):
        r"""Returns a :class:`~.QParams` object containing quantization params of a :class:`~.Tensor`."""
        from .quantization.utils import create_qparams  # pylint: disable=all

        if self._qparams is None:
            self._qparams = create_qparams()
        return self._qparams

    def numpy(self) -> np.ndarray:
        r"""Returns self :class:`~.Tensor` as a :class:`numpy.ndarray`."""
        return super().numpy()

    def detach(self):
        r"""Returns a new :class:`~.Tensor`, detached from the current graph."""
        return super().detach()

    def _reset(self, other):
        if not isinstance(other, _Tensor):
            other = Tensor(other, dtype=self.dtype, device=self.device)
        super()._reset(other)

    def __repr__(self):
        piece = "{}(".format(self.__class__.__name__)
        with np.printoptions(precision=4, suppress=False):
            piece += "{}".format(str(self.numpy()))
        if self.dtype != np.float32:
            piece += ", dtype={}".format(np.dtype(self.dtype).name)
        piece += ", device={}".format(self.device) + ")"
        return piece

    @property
    def name(self):
        r"""Returns a string represents the name of a :class:`~.Tensor`."""
        return self._custom_name

    @name.setter
    def name(self, name):
        self._custom_name = name
        if name == None:
            name = ""
        self._name = self._prefix + "." + name if self._prefix else name
        self._set_name(self._name)

    @deprecated(
        version="1.0", reason="please use ``tensor_name[...] = value``",
    )
    def set_value(self, value):
        self._reset(value)

    @deprecated(version="1.0", reason="use ``*= 0`` instead")
    def reset_zero(self):
        self *= 0

    def to(self, device, *, _borrow=False):
        r"""Copy self :class:`~.Tensor` to specified device. See :func:`~.copy`"""
        if isinstance(device, str) and not _valid_device(device):
            raise ValueError(
                "invalid device name {}. For the correct format of the device name, please refer to the instruction of megengine.device.set_default_device()".format(
                    device
                )
            )
        cn = as_device(device).to_c()
        op = Borrow(comp_node=cn) if _borrow else Copy(comp_node=cn)
        return apply(op, self)[0]

    @property
    def requires_grad(self):
        r"""Returns a bool indicates whether the :class:`~.Tensor` requires gradient."""
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
        r"""__getnewargs__ will be called for pickle serialization or deep copy"""
        return (self.numpy(), self.dtype, self.device.logical_name)

    def __getstate__(self):
        r"""__getstate__ will be called for pickle serialization or deep copy"""
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


set_py_tensor_type(Tensor)


tensor = Tensor


class Parameter(Tensor):
    r"""A kind of Tensor that is to be considered a module parameter.

    Note:
        Operations happened on Parameter usually return a Tensor instead of Parameter.
        For example, with a Parameter ``x``, ``x.reshape/to/sum/...`` will result into a Tensor.
        Any operations between Parameter and Tensor will have Tensor as outputs.
    """
