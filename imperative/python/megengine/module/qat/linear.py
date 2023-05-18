from ... import functional as F
from .. import linear as Float
from .module import QATModule


class Linear(Float.Linear, QATModule):
    r"""A :class:`~.QATModule` version of :class:`~.module.Linear`.
    Could be applied with :class:`~.Observer` and :class:`~.quantization.fake_quant.FakeQuantize`.

    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: True
    """

    def calc_linear_qat(self, inp):
        w_qat = self.apply_quant_weight(self.weight)
        b_qat = self.apply_quant_bias(self.bias, inp, w_qat)
        linear = self.calc_linear(inp, w_qat, b_qat)
        return linear

    def forward(self, inp):
        w_qat = self.apply_quant_weight(self.weight)
        b_qat = self.apply_quant_bias(self.bias, inp, w_qat)
        return self.apply_quant_activation(self.calc_linear(inp, w_qat, b_qat))

    @classmethod
    def from_float_module(cls, float_module: Float.Linear):
        r"""
        Return a :class:`~.QATModule` instance converted from
        a float :class:`~.Module` instance.
        """
        qmod = cls(
            float_module.in_features, float_module.out_features, name=float_module.name
        )
        qmod.weight = float_module.weight
        qmod.bias = float_module.bias
        return qmod


class LinearRelu(Linear):
    r"""A :class:`~.QATModule` include :class:`~.module.Linear` and :func:`~.relu` with QAT support.
    Could be applied with :class:`~.Observer` and :class:`~.quantization.fake_quant.FakeQuantize`.
    """

    def forward(self, inp):
        return self.apply_quant_activation(F.relu(self.calc_linear_qat(inp)))
