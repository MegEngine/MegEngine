# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import math
from typing import Union

from .. import functional as F
from ..core.tensor.dtype import QuantDtypeMeta, _builtin_quant_dtypes
from ..logger import get_logger
from ..module import Module
from ..tensor import Parameter
from .utils import (
    QParams,
    QParamsModuleMixin,
    QuantMode,
    create_qparams,
    fake_quant_tensor,
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

    def normal_foward(self, inp, qparams: QParams = None):
        return inp

    def forward(self, inp, qparams: QParams = None):
        if self.enabled:
            return self.fake_quant_forward(inp, qparams=qparams)
        else:
            return self.normal_foward(inp, qparams=qparams)


class TQT(_FakeQuantize, QParamsModuleMixin):
    r"""
    TQT: https://arxiv.org/abs/1903.08066 Trained Quantization Thresholds
    for Accurate and Efficient Fixed-Point Inference of Deep Neural Networks.

    :param dtype: a string or :class:`~.QuantDtypeMeta` indicating the target
        quantization dtype of input.
    :param enable: whether do ``normal_forward`` or ``fake_quant_forward``.
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
    r"""
    A module to do quant and dequant according to observer's scale and zero_point.

    :param dtype: a string or :class:`~.QuantDtypeMeta` indicating the target
        quantization dtype of input.
    :param enable: whether do ``normal_forward`` or ``fake_quant_forward``.
    """

    def fake_quant_forward(self, inp, qparams: QParams = None):
        assert (
            qparams.dtype_meta is self.dtype
        ), "input qparams' dtype is not equal to self.dtype.\nqparams.dtype_meta={}\nself.dtype={}".format(
            qparams.dtype_meta, self.dtype
        )
        return fake_quant_tensor(inp, qparams)
