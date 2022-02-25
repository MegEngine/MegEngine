from .. import batch_matmul_activation as Float
from .module import QATModule


class BatchMatMulActivation(Float.BatchMatMulActivation, QATModule):
    r"""A :class:`~.QATModule` :class:`~.module.BatchMatMulActivation` with QAT support."""

    def forward(self, inp):
        w_qat = self.apply_quant_weight(self.weight)
        b_qat = self.apply_quant_bias(self.bias, inp, w_qat)
        return self.apply_quant_activation(self._calc_linear(inp, w_qat, b_qat))

    @classmethod
    def from_float_module(cls, float_module: Float.BatchMatMulActivation):
        qat_module = cls(
            float_module.batch,
            float_module.in_features,
            float_module.out_features,
            float_module.bias is not None,
            name=float_module.name,
        )
        qat_module.weight = float_module.weight
        qat_module.bias = float_module.bias
        return qat_module
