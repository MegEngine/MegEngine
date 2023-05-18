import numpy as np

from ... import functional as F
from ... import module as Float
from ...core.tensor import dtype
from ...tensor import Parameter
from ..qat import linear as QAT
from .module import QuantizedModule


class Linear(QuantizedModule):
    r"""Quantized version of :class:`~.qat.Linear`."""

    def __init__(self, dtype: np.dtype = None, **kwargs):
        super().__init__(**kwargs)
        self.weight = None
        self.bias = None
        self.output_dtype = dtype

    def calc_linear_quantized(self, inp, nonlinear_mode="identity"):
        if self.training:
            raise ValueError("quantized module only support inference.")

        assert nonlinear_mode in ["identity", "relu"]

        inp_scale = dtype.get_scale(inp.dtype)
        w_scale = dtype.get_scale(self.weight.dtype)
        bias_dtype = dtype.qint32(inp_scale * w_scale)
        ret = F.linear(
            inp,
            self.weight,
            None if self.bias is None else self.bias.astype(bias_dtype),
        )
        ret = ret if self.output_dtype is None else ret.astype(self.output_dtype)

        if nonlinear_mode == "relu":
            ret = F.relu(ret)

        return ret

    def forward(self, inp):
        return self.calc_linear_quantized(inp)

    @classmethod
    def from_qat_module(cls, qat_module: QAT.Linear):
        r"""
        Return a :class:`~.QuantizedModule` instance converted from a
        :class:`~.QATModule` instance.
        """
        output_dtype = qat_module.get_activation_dtype()
        qmod = cls(dtype=output_dtype, name=qat_module.name)
        qmod.name = qat_module.name
        weight = qat_module.weight.astype(qat_module.get_weight_dtype())
        qmod.weight = Parameter(weight.numpy(), name=qat_module.weight.name)
        if qat_module.bias is not None:
            qmod.bias = Parameter(qat_module.bias.numpy(), name=qat_module.bias.name)
        return qmod


class LinearRelu(Linear):
    r"""Quantized version of :class:`~.qat.LinearRelu`."""

    def forward(self, inp):
        return self.calc_linear_quantized(inp, nonlinear_mode="relu")
