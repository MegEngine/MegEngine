from typing import Iterable

from ... import functional as F
from ...tensor import Tensor
from ..qat import concat as QAT
from .module import QuantizedModule


class Concat(QuantizedModule):
    r"""A :class:`~.QuantizedModule` to do quantized :func:`~.concat`, used for inference only."""

    def __init__(self, dtype=None, **kwargs):
        super().__init__(**kwargs)
        self.output_dtype = dtype

    def forward(self, inps: Iterable[Tensor], axis: int = 0):
        new_inps = tuple(x.astype(self.output_dtype) for x in inps)
        return F.concat(new_inps, axis)

    @classmethod
    def from_qat_module(cls, qat_module: QAT.Concat):
        r"""
        Return a :class:`~.QuantizedModule` instance converted from a
        :class:`~.QATModule` instance.
        """
        return cls(qat_module.get_activation_dtype(), name=qat_module.name)
