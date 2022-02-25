from typing import Iterable

from ...tensor import Tensor
from .. import concat as Float
from .module import QATModule


class Concat(Float.Concat, QATModule):
    r"""A :class:`~.QATModule` to do functional :func:`~.concat` with QAT support.
    Could be applied with :class:`~.Observer` and :class:`~.quantization.fake_quant.FakeQuantize`.
    """

    def forward(self, inps: Iterable[Tensor], axis: int = 0):
        return self.apply_quant_activation(super().forward(inps, axis))

    @classmethod
    def from_float_module(cls, float_module):
        return cls(name=float_module.name)
