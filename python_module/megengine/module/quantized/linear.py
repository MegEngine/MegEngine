# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

import megengine._internal as mgb

from ... import functional as F
from ... import module as Float
from ...core import Parameter
from ...quantization.utils import register_method_to_class
from ..module import Module


class Linear(Module):
    r"""Applies a quantized linear transformation to the input. The module
    usually convert from QAT module by to_quantized method.

    :param dtype: output data type.

    """

    def __init__(
        self, dtype: np.dtype = None,
    ):
        super().__init__()
        self.weight = None
        self.bias = None
        self.output_dtype = dtype

    def forward(self, inp):
        if self.training:
            raise ValueError("quantized module only support inference.")
        inp_scale = mgb.dtype.get_scale(inp.dtype)
        w_scale = mgb.dtype.get_scale(self.weight.dtype)
        bias_dtype = mgb.dtype.qint32(inp_scale * w_scale)
        return F.linear(
            inp,
            self.weight,
            None if self.bias is None else self.bias.astype(bias_dtype),
        ).astype(self.output_dtype)


@register_method_to_class(Float.Linear)
def to_quantized(float_module):
    r"""
    Replace :class:`~.module.QATModule`'s ``to_quantized`` method.
    implemented here to avoid circular import.
    """
    output_dtype = float_module.act_observer.get_dtype()
    qmod = Linear(dtype=output_dtype,)
    weight = float_module.weight.astype(float_module.weight_observer.get_dtype())
    qmod.weight = Parameter(weight.numpy())
    if float_module.bias is not None:
        qmod.bias = Parameter(float_module.bias.numpy())
    return qmod
