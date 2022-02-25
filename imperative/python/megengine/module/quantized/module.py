from abc import abstractmethod

from ..module import Module
from ..qat import QATModule


class QuantizedModule(Module):
    r"""Base class of quantized :class:`~.Module`,
    which should be converted from :class:`~.QATModule` and not support traning.
    """

    def __call__(self, *inputs, **kwargs):
        if self.training:
            raise ValueError("quantized module only support inference.")
        return super().__call__(*inputs, **kwargs)

    def __repr__(self):
        return "Quantized." + super().__repr__()

    @classmethod
    @abstractmethod
    def from_qat_module(cls, qat_module: QATModule):
        r"""
        Return a :class:`~.QATModule` instance converted from
        a float :class:`~.Module` instance.
        """
