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
from ..core.tensor.dtype import _metadata_dict, get_quantized_dtype
from ..core.tensor.function import Function
from ..module import Module
from ..tensor import Parameter, Tensor
from .utils import QuantMode, fake_quant_tensor, get_qparam_dict


class _FakeQuantize(Module):
    r"""
    A Basic Fake Quant module.

    :param dtype: a string indicating the target quantization type of input.
    :param narrow_range: whether the absolute value of ``qmin`` is the same as ``qmax``,
        instead of 1 greater. Usually True for weight and False for activation.
    :param enable: whether do ``normal_forward`` or ``fake_quant_forward``.
    """

    def __init__(self, dtype: str, narrow_range: bool = False, enable: bool = True):
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


class TQT_Function(Function):
    def __init__(self, lowerbound, upperbound):
        super().__init__()
        self.lowerbound = lowerbound
        self.upperbound = upperbound
        self.saved_tensors = ()

    def save_for_backward(self, *tensors: Iterable[Tensor]):
        """
        Saves tensors needed for gradient computation. This method should be called only
        once in :meth:`~.function.Function.forward`, additional calls will replace values saved previously.

        The saved tensors can be accessed through the ``saved_tensors`` attribute.
        """
        self.saved_tensors = tensors

    def forward(self, inp, scale):
        t = 2 ** scale
        # t = F.maximum(t, 1e-4)
        inp_scaled = inp / t
        inp_clipped = F.maximum(F.minimum(inp_scaled, self.upperbound), self.lowerbound)
        inp_rounded = F.round(inp_clipped)
        inp_flq = inp_rounded * t
        self.save_for_backward(inp_scaled, inp_rounded, t)
        return inp_flq

    def backward(self, grad_inp_flq):
        (inp_scaled, inp_rounded, t) = self.saved_tensors
        mask_clip = F.logical_and(
            inp_scaled < -0.5 + self.lowerbound, inp_scaled > self.upperbound + 0.5
        )  # mask for accumulating the gradients of |data_scaled|>L
        mask_quant = F.logical_not(mask_clip)
        grad_quant = (
            grad_inp_flq * mask_quant * (inp_rounded - inp_scaled)
        )  # gradient within |data_scaled|<=L
        grad_clip = (
            grad_inp_flq * mask_clip * inp_rounded
        )  # gradient with   | data_scaled|>L
        grad_s = grad_clip.sum() + grad_quant.sum()
        # dL/ds = dL/dt * t * ln(2)
        grad_s = grad_s * t * math.log(2)
        grad_inp = grad_inp_flq * mask_quant
        return grad_inp, grad_s


class TQT(_FakeQuantize):
    r"""
    TQT: https://arxiv.org/abs/1903.08066 Trained Quantization Thresholds
    for Accurate and Efficient Fixed-Point Inference of Deep Neural Networks.
    """

    def __init__(self, dtype: str, narrow_range: bool = False, enable: bool = True):
        super().__init__(dtype, narrow_range, enable)
        self.scale = Parameter(0.0, dtype=np.float32)

    def fake_quant_forward(self, inp, q_dict=None):
        # when enable, TQT will do fakequant forward, finetune the scale
        return TQT_Function(self.qmin, self.qmax)(inp, self.scale)

    def normal_foward(self, inp, q_dict=None):
        if q_dict["enable_observer"]:
            # when disable, TQT will do normal forward, initialize scale weight
            tmp_scale = F.maximum(F.abs(q_dict["min_val"]), F.abs(q_dict["max_val"]))
            tmp_scale = F.log(tmp_scale / 127) / math.log(2)
            self.scale[...] = tmp_scale
        return inp

    def get_qparams(self):
        q_dict = get_qparam_dict(QuantMode.TQT)
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
