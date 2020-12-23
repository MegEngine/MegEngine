# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import math
from typing import Iterable

import numpy as np

from .. import functional as F
from ..core.autodiff.grad import Function
from ..core.tensor.dtype import _metadata_dict, get_quantized_dtype
from ..module import Module
from ..tensor import Parameter, Tensor
from .utils import QuantMode, fake_quant_tensor, get_qparam_dict, tqt_forward


class _FakeQuantize(Module):
    r"""
    A Basic Fake Quant module.

    :param dtype: a string indicating the target quantization type of input.
    :param narrow_range: whether the absolute value of ``qmin`` is the same as ``qmax``,
        instead of 1 greater. Usually True for weight and False for activation.
    :param enable: whether do ``normal_forward`` or ``fake_quant_forward``.
    """

    def __init__(
        self, dtype: str, narrow_range: bool = False, enable: bool = True, **kwargs
    ):
        super().__init__()
        if not dtype in _metadata_dict.keys():
            raise ValueError(
                "unknown dtype: {}, only support {}".format(
                    dtype, _metadata_dict.keys()
                )
            )
        self.dtype = dtype
        self.narrow_range = narrow_range
        self.qmin = (
            -_metadata_dict[dtype].qmax if narrow_range else _metadata_dict[dtype].qmin
        )
        self.qmax = _metadata_dict[dtype].qmax
        self.enabled = enable

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def fake_quant_forward(self, inp, q_dict=None):
        return inp

    def normal_foward(self, inp, q_dict=None):
        return inp

    def forward(self, inp, q_dict=None):
        if self.enabled:
            return self.fake_quant_forward(inp, q_dict=q_dict)
        else:
            return self.normal_foward(inp, q_dict=q_dict)


class TQT(_FakeQuantize):
    r"""
    TQT: https://arxiv.org/abs/1903.08066 Trained Quantization Thresholds
    for Accurate and Efficient Fixed-Point Inference of Deep Neural Networks.
    """

    def __init__(
        self,
        q_dict,
        dtype: str,
        narrow_range: bool = False,
        enable: bool = True,
        **kwargs
    ):
        super().__init__(dtype, narrow_range, enable, **kwargs)
        assert (
            q_dict["mode"] == QuantMode.SYMMERTIC
        ), "only symmetric quantization is supported by TQT"
        if "scale" not in q_dict or q_dict["scale"] is None:
            raise AssertionError("Can not get an initialized scale")
        self.scale = Tensor(F.log(q_dict["scale"]) / math.log(2))

    def fake_quant_forward(self, inp, q_dict=None):
        # when enable, TQT will do fakequant forward, finetune the scale
        return tqt_forward(self.qmin, self.qmax, inp, self.scale)

    def get_qparams(self):
        q_dict = get_qparam_dict(QuantMode.SYMMERTIC)
        q_dict["scale"] = 2 ** self.scale
        return q_dict

    def get_dtype(self):
        q_dict = self.get_qparams()
        scale = None if "scale" not in q_dict else q_dict["scale"].numpy()[0]
        zero_point = (
            None if "zero_point" not in q_dict else q_dict["zero_point"].numpy()[0]
        )
        return get_quantized_dtype(self.dtype, scale, zero_point)


class FakeQuantize(_FakeQuantize):
    r"""
    A module to do quant and dequant according to observer's scale and zero_point.
    """

    def fake_quant_forward(self, inp, q_dict=None):
        return fake_quant_tensor(inp, self.qmin, self.qmax, q_dict)
