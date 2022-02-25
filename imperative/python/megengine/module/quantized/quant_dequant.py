from ..qat import quant_dequant as QAT
from .module import QuantizedModule


class QuantStub(QuantizedModule):
    r"""Quantized version of :class:`~.qat.QuantStub`,
    will convert input to quantized dtype.
    """

    def __init__(self, dtype=None, **kwargs):
        super().__init__(**kwargs)
        self.output_dtype = dtype

    def forward(self, inp):
        return inp.astype(self.output_dtype)

    @classmethod
    def from_qat_module(cls, qat_module: QAT.QuantStub):
        return cls(qat_module.get_activation_dtype(), name=qat_module.name)


class DequantStub(QuantizedModule):
    r"""Quantized version of :class:`~.qat.DequantStub`,
    will restore quantized input to float32 dtype.
    """

    def forward(self, inp):
        return inp.astype("float32")

    @classmethod
    def from_qat_module(cls, qat_module: QAT.DequantStub):
        return cls(name=qat_module.name)
