from .. import elemwise as Float
from .module import QATModule


class Elemwise(Float.Elemwise, QATModule):
    r"""A :class:`~.QATModule` to do :mod:`~.functional.elemwise` operator with QAT support.
    Could be applied with :class:`~.Observer` and :class:`~.quantization.fake_quant.FakeQuantize`.
    """

    with_weight = False

    def forward(self, *inps):
        return self.apply_quant_activation(super().forward(*inps))

    @classmethod
    def from_float_module(cls, float_module: Float.Elemwise):
        r"""
        Return a :class:`~.QATModule` instance converted from
        a float :class:`~.Module` instance.
        """
        return cls(float_module.method, name=float_module.name)
