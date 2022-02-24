from typing import Tuple

from ..functional import nn
from .module import Module


class Pad(Module):
    r"""Pads the input tensor.

    Args:
        pad_width: A tuple. Each element in the tuple is the tuple of 2-elements,
            the 2 elements represent the padding size on both sides of the current dimension, ``(front_offset, back_offset)``
        mode: One of the following string values. Default: ``'constant'``

            * ``'constant'``: Pads with a constant value.
            * ``'reflect'``: Pads with the edge values of tensor.
            * ``'replicate'``: Pads with the reflection of the tensor mirrored on the first and last values of the tensor along each axis.
        constant_val: Fill value for ``'constant'`` padding. Default: 0

    Examples:

        >>> import numpy as np
        >>> inp = Tensor([[1., 2., 3.],[4., 5., 6.]])
        >>> inp
        Tensor([[1. 2. 3.]
         [4. 5. 6.]], device=xpux:0)
        >>> m = M.Pad(pad_width=((1, 1),), mode="constant")
        >>> m(inp)
        Tensor([[0. 0. 0.]
         [1. 2. 3.]
         [4. 5. 6.]
         [0. 0. 0.]], device=xpux:0)
        >>> m = M.Pad(pad_width=((1, 1),), mode="constant", constant_val=9)
        >>> m(inp)
        Tensor([[9. 9. 9.]
         [1. 2. 3.]
         [4. 5. 6.]
         [9. 9. 9.]], device=xpux:0)
        >>> m = M.Pad(pad_width=((1, 1), (1, 2)), mode="reflect")
        >>> m(inp)
        Tensor([[5. 4. 5. 6. 5. 4.]
         [2. 1. 2. 3. 2. 1.]
         [5. 4. 5. 6. 5. 4.]
         [2. 1. 2. 3. 2. 1.]], device=xpux:0)
        >>> m = M.Pad(pad_width=((1, 1), (1, 2)), mode="replicate")
        >>> m(inp)
        Tensor([[1. 1. 2. 3. 3. 3.]
         [1. 1. 2. 3. 3. 3.]
         [4. 4. 5. 6. 6. 6.]
         [4. 4. 5. 6. 6. 6.]], device=xpux:0)

    """

    def __init__(
        self,
        pad_width: Tuple[Tuple[int, int], ...],
        mode: str = "constant",
        constant_val: float = 0.0,
    ):
        super().__init__()
        self.pad_width = pad_width
        self.mode = mode
        self.pad_val = constant_val

    def forward(self, src):
        return nn.pad(
            src, pad_width=self.pad_width, mode=self.mode, constant_value=self.pad_val
        )
