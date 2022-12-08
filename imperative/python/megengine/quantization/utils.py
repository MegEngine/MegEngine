import abc
from enum import Enum
from functools import partial, update_wrapper, wraps
from typing import Union

import numpy as np

from .. import functional as F
from ..autodiff import Function
from ..core._imperative_rt.core2 import apply
from ..core.ops import builtin
from ..core.tensor.dtype import (
    QuantDtypeMeta,
    _builtin_quant_dtypes,
    create_quantized_dtype,
)
from ..tensor import Tensor


class Round(Function):
    r"""The functional round have no grad and can not use for quantization-aware-training.
    We use Function and STE(Straight-Through Estimator) to implement backward propagation.
    """

    def forward(self, x):
        return F.round(x)

    def backward(self, output_grads):
        return output_grads


def tqt_forward(qmin, qmax, inp, scale):
    op = builtin.TQT(qmin=qmin, qmax=qmax)
    (output,) = apply(op, inp, scale)
    return output


def lsq_forward(qmin, qmax, inp, step_size, zero_point=None, scale_grad=None):
    if zero_point is None:
        zero_point = Tensor([0.0], dtype=np.float32)
    if scale_grad is None:
        scale_grad = Tensor([1.0], dtype=np.float32)
    op = builtin.LSQ(qmin=qmin, qmax=qmax)
    (output,) = apply(op, inp, step_size, zero_point, scale_grad)
    return output


def register_method_to_class(cls):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        if isinstance(func, partial):
            update_wrapper(func, func.func)
        setattr(cls, func.__name__, wrapper)
        return func

    return decorator


class QuantMode(Enum):
    r"""Quantization mode enumerate class."""

    SYMMERTIC = 1
    ASYMMERTIC = 2


class QParams:
    r"""To standardize FakeQuant, Observer and Tensor's qparams format. If custom
    qparams is needed, inherit this class and add custom ``__slots__``.
    """

    __slots__ = "mode", "dtype_meta", "scale", "zero_point"

    def __init__(
        self,
        mode: QuantMode,
        dtype_meta: QuantDtypeMeta,
        scale: Tensor,
        zero_point: Tensor,
    ):
        self.mode = mode
        self.dtype_meta = dtype_meta
        self.scale = scale
        self.zero_point = zero_point

    def update(self, qparams: "QParams"):
        for key in self.__slots__:
            setattr(self, key, getattr(qparams, key))

    def __eq__(self, other):
        if len(self.__slots__) != len(other.__slots__):
            return False
        for key in self.__slots__:
            if not hasattr(other, key) or getattr(self, key) != getattr(other, key):
                return False
        return True

    def __repr__(self):
        content = ", ".join(
            ["{}={}".format(key, getattr(self, key)) for key in self.__slots__]
        )
        return "QParams({})".format(content)


class LSQParams(QParams):
    r"""LSQ qparams with extra grad_scale slot."""

    __slots__ = "mode", "dtype_meta", "scale", "zero_point", "grad_scale"

    def __init__(
        self,
        mode: QuantMode,
        dtype_meta: QuantDtypeMeta,
        scale: Tensor,
        zero_point: Tensor,
        grad_scale: Tensor,
    ):
        super().__init__(mode, dtype_meta, scale, zero_point)
        self.grad_scale = grad_scale


class QParamsModuleMixin(abc.ABC):
    def get_quantized_dtype(self):
        qparams = self.get_qparams()
        dtype = qparams.dtype_meta
        scale = float(qparams.scale.numpy()) if qparams.scale is not None else None
        zero_point = (
            int(qparams.zero_point.numpy()) if qparams.zero_point is not None else None
        )
        return create_quantized_dtype(dtype, scale, zero_point)

    @abc.abstractmethod
    def get_qparams(self) -> QParams:
        pass


_builtin_qparams = {
    QuantMode.SYMMERTIC: partial(QParams, mode=QuantMode.SYMMERTIC),
    QuantMode.ASYMMERTIC: partial(QParams, mode=QuantMode.ASYMMERTIC),
}


def create_qparams(
    mode: QuantMode = QuantMode.SYMMERTIC,
    dtype_meta: Union[str, QuantDtypeMeta] = None,
    scale: Tensor = None,
    zero_point: Tensor = None,
):
    r"""

    Args:
        mode: QuantMode:
        dtype_meta: Union[str:
        QuantDtypeMeta]:
        scale: Tensor:
        zero_point: Tensor:
    """
    if isinstance(dtype_meta, str):
        dtype_meta = _builtin_quant_dtypes[dtype_meta]
    if mode is None:
        return QParams(mode, dtype_meta, scale, zero_point)
    assert isinstance(mode, QuantMode)
    return _builtin_qparams[mode](
        dtype_meta=dtype_meta, scale=scale, zero_point=zero_point
    )


def fake_quant_tensor(inp: Tensor, qparams: QParams) -> Tensor:
    """Apply fake quantization to the inp tensor.

    Args:
        inp: the input tensor which need to be faked.
        qparams: to get mode, qmin, qmax, scale and zero_point from.
    """
    scale = qparams.scale
    if qparams.mode == QuantMode.ASYMMERTIC:
        zero_point = qparams.zero_point
    else:
        zero_point = Tensor([0.0], dtype=np.float32)
    qmin = qparams.dtype_meta.qmin
    qmax = qparams.dtype_meta.qmax

    op = builtin.FakeQuant(qmin=qmin, qmax=qmax)
    return apply(op, inp, scale, zero_point)[0]


def fake_quant_bias(bias: Tensor, inp: Tensor, w_qat: Tensor) -> Tensor:
    """Apply fake quantization to bias, with the special scale from input tensor
    and weight tensor, the quantized type set to qint32 also.

    Args:
        bias: the bias tensor which need to be faked.
        inp: the input tensor which contain the quantization parameters.
        w_qat: the weight tensor which contain the quantization parameters.

    Warning:
        Only work for symmetric quantization method now.
    """
    b_qat = bias
    if (
        getattr(inp, "qparams", None) is not None
        and getattr(w_qat, "qparams", None) is not None
        and bias is not None
    ):
        inp_params = inp.qparams
        w_params = w_qat.qparams

        if inp_params.scale is None or w_params.scale is None:
            return b_qat

        if inp_params.mode != QuantMode.SYMMERTIC:
            return b_qat

        # TODO: support different mode
        if inp_params.mode != w_params.mode:
            return b_qat

        # TODO: support quint8 dtype.
        if inp_params.dtype_meta.np_dtype_str != "int8":
            return b_qat
        if w_params.dtype_meta.np_dtype_str != "int8":
            return b_qat

        # use the same mode with weight.
        # TODO: avoid hardcode
        b_dtype = _builtin_quant_dtypes["qint32"]
        b_param = create_qparams(
            w_params.mode, b_dtype, scale=inp_params.scale * w_params.scale
        )
        b_qat = fake_quant_tensor(bias, b_param)
        b_qat.qparams.update(b_param)
    return b_qat
