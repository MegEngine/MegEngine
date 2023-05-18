from ...tensor import Parameter
from ..qat import linear_bn as QAT
from .linear import Linear


class _LinearBnActivation1d(Linear):
    r"""Applies a Linear over a quantized input tensor, used for inference only.
    """

    @classmethod
    def from_qat_module(cls, qat_module: QAT._LinearBnActivation1d):
        r"""
        Return a :class:`~.QuantizedModule` instance converted from a
        :class:`~.QATModule` instance.
        """
        output_dtype = qat_module.get_activation_dtype()
        qlinear = cls(dtype=output_dtype, name=qat_module.name,)
        w_fold, b_fold = qat_module.fold_weight_bias(
            qat_module.bn.running_mean, qat_module.bn.running_var
        )
        weight = w_fold.astype(qat_module.get_weight_dtype())
        qlinear.weight = Parameter(weight.numpy(), name=qat_module.linear.weight.name)
        qlinear.bias = Parameter(b_fold.numpy())
        if qat_module.linear.bias is not None:
            qlinear.bias.name = qat_module.linear.bias.name
        return qlinear


class LinearBn1d(_LinearBnActivation1d):
    r"""Quantized version of :class:`~.qat.LinearBn1d`."""

    def forward(self, inp):
        return self.calc_linear_quantized(inp, nonlinear_mode="identity")


class LinearBnRelu1d(_LinearBnActivation1d):
    r"""Quantized version of :class:`~.qat.LinearBnRelu1d`."""

    def forward(self, inp):
        return self.calc_linear_quantized(inp, nonlinear_mode="relu")
