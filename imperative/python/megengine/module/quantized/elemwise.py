from ...functional.elemwise import _elemwise_multi_type
from ...tensor import Tensor
from ..qat import elemwise as QAT
from .module import QuantizedModule


class Elemwise(QuantizedModule):
    r"""Quantized version of :class:`~.qat.Elemwise`."""

    def __init__(self, method, dtype=None, **kwargs):
        super().__init__(**kwargs)
        self.method = "q" + method
        self.output_dtype = dtype

    def forward(self, *inps):
        if self.training:
            raise ValueError("quantized module only support inference.")
        return _elemwise_multi_type(*inps, mode=self.method, dtype=self.output_dtype)

    @classmethod
    def from_qat_module(cls, qat_module: QAT.Elemwise):
        r"""
        Return a :class:`~.QuantizedModule` instance converted from a
        :class:`~.QATModule` instance.
        """
        return cls(
            qat_module.method, qat_module.get_activation_dtype(), name=qat_module.name
        )
