from .module import Module


class QuantStub(Module):
    r"""A helper :class:`~.Module` simply returning input. Could be replaced with :class:`~.QATModule`
    version :class:`~.qat.QuantStub` using :func:`~.quantize.quantize_qat`.
    """

    def forward(self, inp):
        return inp


class DequantStub(Module):
    r"""A helper :class:`~.Module` simply returning input. Could be replaced with :class:`~.QATModule`
    version :class:`~.qat.DequantStub` using :func:`~.quantize.quantize_qat`.
    """

    def forward(self, inp):
        return inp
