# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Tuple, Union

import numpy as np

from ... import module as Float
from ...core.tensor import dtype
from ...functional import expand_dims, squeeze
from ...functional.quantized import batch_conv_bias_activation
from ...tensor import Parameter
from ..qat import batch_matmul_activation as QAT
from .module import QuantizedModule


class BatchMatMulActivation(Float.BatchMatMulActivation, QuantizedModule):
    def __init__(
        self,
        batch: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
        nonlinear_mode="IDENTITY",
        dtype=None,
        **kwargs
    ):
        super().__init__(batch, in_features, out_features, bias, **kwargs)
        self.output_dtype = dtype

    def calc_bmm_quantized(self, inp):
        inp_scale = dtype.get_scale(inp.dtype)
        w_scale = dtype.get_scale(self.weight.dtype)
        bias_scale = inp_scale * w_scale
        inp = expand_dims(inp, [-1])
        res = batch_conv_bias_activation(
            inp,
            self.weight,
            self.bias.astype(dtype.qint32(bias_scale)),
            dtype=self.output_dtype,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            nonlinear_mode=self.nonlinear_mode,
        )
        return squeeze(res, -1)

    @classmethod
    def from_qat_module(cls, qat_module: QAT.BatchMatMulActivation):
        output_dtype = qat_module.get_activation_dtype()
        qbmm = cls(
            qat_module.batch,
            qat_module.in_features,
            qat_module.out_features,
            qat_module.bias is not None,
            dtype=output_dtype,
        )
        weight = qat_module.weight.astype(qat_module.get_weight_dtype())
        weight = expand_dims(weight, [-1, -2])
        qbmm.weight = Parameter(weight.numpy())
        if qat_module.bias is not None:
            bias = qat_module.bias.reshape((1, qbmm.out_features, 1, 1))
            qbmm.bias = Parameter(bias.numpy())
        else:
            qbmm.bias = Parameter(
                np.zeros((1, qbmm.out_features, 1, 1), dtype=np.float32)
            )
        return qbmm

    def forward(self, inp):
        return self.calc_bmm_quantized(inp)
