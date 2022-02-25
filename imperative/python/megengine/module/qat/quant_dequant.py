from .. import quant_dequant as Float
from .module import QATModule


class QuantStub(Float.QuantStub, QATModule):
    r"""A helper :class:`~.QATModule` simply return input, but will quantize
    input after converted to :class:`~.QuantizedModule`.
    """

    with_weight = False

    def forward(self, inp):
        return self.apply_quant_activation(inp)

    @classmethod
    def from_float_module(cls, float_module: Float.QuantStub):
        r"""
        Return a :class:`~.QATModule` instance converted from
        a float :class:`~.Module` instance.
        """
        return cls(name=float_module.name)


class DequantStub(Float.DequantStub, QATModule):
    r"""A helper :class:`~.QATModule` simply return input, but will de-quantize
    input after converted to :class:`~.QuantizedModule`.
    """

    with_weight = False
    with_act = False

    def forward(self, inp):
        return inp

    @classmethod
    def from_float_module(cls, float_module: Float.DequantStub):
        r"""
        Return a :class:`~.QATModule` instance converted from
        a float :class:`~.Module` instance.
        """
        return cls(name=float_module.name)
