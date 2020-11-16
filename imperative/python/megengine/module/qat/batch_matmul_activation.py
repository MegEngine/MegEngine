# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from ...quantization.utils import fake_quant_bias
from .. import batch_matmul_activation as Float
from .module import QATModule


class BatchMatMulActivation(Float.BatchMatMulActivation, QATModule):
    def forward(self, inp):
        w_qat = self.apply_quant_weight(self.weight)
        b_qat = fake_quant_bias(self.bias, inp, w_qat)
        return self.apply_quant_activation(self._calc_linear(inp, w_qat, b_qat))

    @classmethod
    def from_float_module(cls, float_module: Float.BatchMatMulActivation):
        qat_module = cls(
            float_module.batch,
            float_module.in_features,
            float_module.out_features,
            float_module.bias is not None,
        )
        qat_module.weight = float_module.weight
        qat_module.bias = float_module.bias
        return qat_module
