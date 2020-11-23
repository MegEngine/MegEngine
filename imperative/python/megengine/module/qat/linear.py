# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ...quantization.utils import fake_quant_bias
from .. import linear as Float
from .module import QATModule


class Linear(Float.Linear, QATModule):
    r"""
    A :class:`~.QATModule` version of :class:`~.module.linear.Linear`.
    Could be applied with :class:`~.Observer` and :class:`~.FakeQuantize`.

    :param in_features: size of each input sample.
    :param out_features: size of each output sample.
    :param bias: If set to ``False``, the layer will not learn an additive bias.
        Default: True

    """

    def forward(self, x):
        w_qat = self.apply_quant_weight(self.weight)
        if self.weight_fake_quant and self.weight_fake_quant.enabled:
            b_qat = fake_quant_bias(self.bias, x, w_qat)
        else:
            b_qat = self.bias
        return self.apply_quant_activation(self._calc_linear(x, w_qat, b_qat))

    @classmethod
    def from_float_module(cls, float_module: Float.Linear):
        r"""
        Return a :class:`~.QATModule` instance converted from
        a float :class:`~.Module` instance.
        """
        qmod = cls(float_module.in_features, float_module.out_features)
        qmod.weight = float_module.weight
        qmod.bias = float_module.bias
        return qmod
