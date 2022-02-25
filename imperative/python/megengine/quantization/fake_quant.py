import math
from typing import Union

from .. import functional as F
from ..core.tensor.dtype import QuantDtypeMeta, _builtin_quant_dtypes
from ..logger import get_logger
from ..module import Module
from ..tensor import Parameter, Tensor
from .utils import (
    LSQParams,
    QParams,
    QParamsModuleMixin,
    QuantMode,
    create_qparams,
    fake_quant_tensor,
    lsq_forward,
    tqt_forward,
)

logger = get_logger(__name__)


class _FakeQuantize(Module):
    def __init__(
        self, dtype: Union[str, QuantDtypeMeta], enable: bool = True, **kwargs
    ):
        super().__init__()
        if isinstance(dtype, str):
            if not dtype in _builtin_quant_dtypes:
                raise ValueError(
                    "unknown dtype: {}, only support {}".format(
                        dtype, _builtin_quant_dtypes.keys()
                    )
                )
            dtype = _builtin_quant_dtypes[dtype]
        if "narrow_range" in kwargs:
            del kwargs["narrow_range"]
            logger.warning(
                "FakeQuantize currently has no narrow_range param "
                "so it is ignored here",
                exc_info=DeprecationWarning,
            )
        self.dtype = dtype
        self.qmin = dtype.qmin
        self.qmax = dtype.qmax
        self.enabled = enable

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def fake_quant_forward(self, inp, qparams: QParams = None):
        raise NotImplementedError

    def normal_forward(self, inp, qparams: QParams = None):
        return inp

    def forward(self, inp, qparams: QParams = None):
        if self.enabled:
            return self.fake_quant_forward(inp, qparams=qparams)
        else:
            return self.normal_forward(inp, qparams=qparams)


class TQT(_FakeQuantize, QParamsModuleMixin):
    r"""TQT: https://arxiv.org/abs/1903.08066 Trained Quantization Thresholds
    for Accurate and Efficient Fixed-Point Inference of Deep Neural Networks.

    Args:
        dtype: a string or :class:`~.QuantDtypeMeta` indicating the target
            quantization dtype of input.
        enable: whether do ``normal_forward`` or ``fake_quant_forward``.
    """

    def __init__(
        self, dtype: Union[str, QuantDtypeMeta], enable: bool = True, **kwargs
    ):
        super().__init__(dtype, enable, **kwargs)
        self.scale = Parameter(0.0, dtype="float32")

    def fake_quant_forward(self, inp, qparams: QParams = None):
        # when enable, TQT will do fakequant forward, finetune the scale
        return tqt_forward(self.qmin, self.qmax, inp, self.scale)

    def set_qparams(self, qparams: QParams):
        assert (
            qparams.mode == QuantMode.SYMMERTIC
        ), "only symmetric quantization is supported by TQT"
        if qparams.scale is None:
            raise AssertionError("Can not get an initialized scale")
        self.scale[...] = F.log(qparams.scale) / math.log(2)

    def get_qparams(self):
        return create_qparams(QuantMode.SYMMERTIC, self.dtype, scale=2 ** self.scale)


class FakeQuantize(_FakeQuantize):
    r"""A module to do quant and dequant according to observer's scale and zero_point.

    Args:
        dtype: a string or :class:`~.QuantDtypeMeta` indicating the target
            quantization dtype of input.
        enable: whether do ``normal_forward`` or ``fake_quant_forward``.
    """

    def fake_quant_forward(self, inp, qparams: QParams = None):
        assert (
            qparams.dtype_meta is self.dtype
        ), "input qparams' dtype is not equal to self.dtype.\nqparams.dtype_meta={}\nself.dtype={}".format(
            qparams.dtype_meta, self.dtype
        )
        return fake_quant_tensor(inp, qparams)


class LSQ(_FakeQuantize, QParamsModuleMixin):
    r"""LSQ: https://arxiv.org/pdf/1902.08153.pdf Estimating and scaling the
    task loss gradient at each weight and activation layer's quantizer step size

    Args:
        dtype: a string or :class:`~.QuantDtypeMeta` indicating the target
            quantization dtype of input.
        enable: whether do ``normal_forward`` or ``fake_quant_forward``.
        eps: a small value to avoid division by zero. Default: 1e-5
    """

    def __init__(
        self,
        dtype: Union[str, QuantDtypeMeta],
        enable: bool = True,
        eps: float = 1e-5,
        **kwargs
    ):
        super().__init__(dtype=dtype, enable=enable, **kwargs)
        self.eps = Tensor(eps, dtype="float32")
        self.step_size = Parameter(1.0, dtype="float32")
        self.mode = None
        self.zero_point = Tensor(0.0, dtype="float32")
        self.grad_scale = Tensor(1.0, dtype="float32")

    def set_qparams(self, qparams: LSQParams):
        self.mode = qparams.mode
        if qparams.mode == QuantMode.ASYMMERTIC:
            self.zero_point = qparams.zero_point
        else:
            self.zero_point = Tensor([0.0], dtype="float32")
        if qparams.scale is None:
            raise AssertionError("Can not get an initialized scale")
        init_step_size = qparams.scale
        if init_step_size < self.eps:
            init_step_size = 0
        else:
            init_step_size = init_step_size - self.eps
        self.step_size = Parameter(init_step_size, dtype="float32")

        self.grad_scale = qparams.grad_scale

    def fake_quant_forward(self, inp, qparams: LSQParams = None):
        step_size = F.abs(self.step_size) + self.eps
        return lsq_forward(
            self.qmin, self.qmax, inp, step_size, self.zero_point, self.grad_scale
        )

    def get_qparams(self):
        return LSQParams(
            mode=self.mode,
            dtype_meta=self.dtype,
            scale=F.abs(self.step_size.detach()) + self.eps,
            zero_point=self.zero_point,
            grad_scale=self.grad_scale,
        )
