# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

from ... import functional as F
from ...core.tensor import dtype
from ...tensor import Parameter
from ..qat import linear as QAT
from .module import QuantizedModule


class Linear(QuantizedModule):
    r"""Quantized version of :class:`~.qat.linear.Linear`."""

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
        inp_scale = dtype.get_scale(inp.dtype)
        w_scale = dtype.get_scale(self.weight.dtype)
        bias_dtype = dtype.qint32(inp_scale * w_scale)
        return F.nn.linear(
            inp,
            self.weight,
            None if self.bias is None else self.bias.astype(bias_dtype),
        ).astype(self.output_dtype)

    @classmethod
    def from_qat_module(cls, qat_module: QAT.Linear):
        r"""
        Return a :class:`~.QuantizedModule` instance converted from a
        :class:`~.QATModule` instance.
        """
        output_dtype = qat_module.get_activation_dtype()
        qmod = cls(dtype=output_dtype)
        weight = qat_module.weight.astype(qat_module.get_weight_dtype())
        qmod.weight = Parameter(weight.numpy())
        if qat_module.bias is not None:
            qmod.bias = Parameter(qat_module.bias.numpy())
        return qmod
