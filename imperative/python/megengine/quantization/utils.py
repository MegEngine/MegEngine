# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from enum import Enum
from functools import partial, update_wrapper, wraps
from typing import Dict

import numpy as np

from .. import functional as F
from ..core.ops import builtin
from ..core.tensor import megbrain_graph
from ..core.tensor.core import apply
from ..core.tensor.dtype import _metadata_dict
from ..core.tensor.function import Function
from ..tensor import Tensor


class Round(Function):
    """
    The functional round have no grad and can not use for quantization-aware-training.
    We use Function and STE(Straight-Through Estimator) to implement backward propagation.
    """

    def forward(self, x):
        return F.round(x)

    def backward(self, output_grads):
        return output_grads


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
    """
    Quantization mode enumerate class.
    """

    SYMMERTIC = 1
    ASYMMERTIC = 2
    TQT = 3


qparam_dict = {
    QuantMode.SYMMERTIC: {"mode": QuantMode.SYMMERTIC, "scale": None,},
    QuantMode.ASYMMERTIC: {
        "mode": QuantMode.ASYMMERTIC,
        "scale": None,
        "zero_point": None,
    },
    QuantMode.TQT: {"mode": QuantMode.TQT, "scale": None,},
}


def get_qparam_dict(mode: QuantMode):
    """
    Return the quantization parameters dictionary according to the mode.
    """
    return qparam_dict.get(mode, None)


def fake_quant_tensor(inp: Tensor, qmin: int, qmax: int, q_dict: Dict) -> Tensor:
    """
    Apply fake quantization to the inp tensor.

    :param inp: the input tensor which need to be faked.
    :param qmin: the minimum value which the integer limit to.
    :param qmax: the maximum value which the integer limit to.
    :param q_dict: the quantization parameter dict.

    """
    scale = q_dict["scale"]
    zero_point = Tensor([0.0], dtype=np.float32)
    if q_dict["mode"] == QuantMode.ASYMMERTIC:
        zero_point = q_dict["zero_point"]

    assert isinstance(inp, (Tensor, megbrain_graph.VarNode)), "inp must be Tensor type"
    assert isinstance(
        scale, (Tensor, megbrain_graph.VarNode)
    ), "scale must be Tensor type"
    assert isinstance(
        zero_point, (Tensor, megbrain_graph.VarNode)
    ), "zero point must be Tensor type"

    op = builtin.FakeQuant(qmin=qmin, qmax=qmax)
    return apply(op, inp, scale, zero_point)[0]


def fake_quant_bias(bias: Tensor, inp: Tensor, w_qat: Tensor) -> Tensor:
    """
    Apply fake quantization to bias, with the special scale from input tensor
    and weight tensor, the quantized type set to qint32 also.

    :param bias: the bias tensor which need to be faked.
    :param inp:  the input tensor which contain the quantization parameters.
    :param qmax: the weight tensor which contain the quantization parameters.

    .. warning::
        Only work for symmetric quantization method now.

    """
    b_qat = bias
    if hasattr(inp, "q_dict") and b_qat is not None:
        if inp.q_dict["scale"] is not None and w_qat.q_dict["scale"] is not None:
            # use the same mode with weight.
            b_dict = get_qparam_dict(w_qat.q_dict["mode"])
            b_dict["scale"] = inp.q_dict["scale"] * w_qat.q_dict["scale"]
            # TODO: add zero_point for ASYMMERTIC mode.
            qmax = _metadata_dict["qint32"].qmax
            qmin = _metadata_dict["qint32"].qmin
            b_qat = fake_quant_tensor(b_qat, qmin, qmax, b_dict)
            b_qat.q_dict.update(b_dict)

    return b_qat
