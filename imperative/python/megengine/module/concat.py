from typing import Iterable

from ..functional import concat
from ..tensor import Tensor
from .module import Module


class Concat(Module):
    r"""A :class:`~.Module` to do functional :func:`~.concat`. Could be replaced with :class:`~.QATModule`
    version :class:`~.qat.Concat` using :func:`~.quantize.quantize_qat`.
    """

    def forward(self, inps: Iterable[Tensor], axis: int = 0):
        return concat(inps, axis)
