from ..functional.elemwise import _elwise
from ..tensor import Tensor
from .module import Module


class Elemwise(Module):
    r"""A :class:`~.Module` to do :mod:`~.functional.elemwise` operator. Could be replaced with :class:`~.QATModule`
    version :class:`~.qat.Elemwise` using :func:`~.quantize.quantize_qat`.

    Args:
        method: the elemwise method, support the following string.
                It will do the normal elemwise operator for float.
    """

    def __init__(self, method, **kwargs):
        super().__init__(**kwargs)
        self.method = method

    def forward(self, *inps):
        return _elwise(*inps, mode=self.method)
