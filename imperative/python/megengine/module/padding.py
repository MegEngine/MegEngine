from typing import Tuple

from ..functional import nn
from .module import Module


class Pad(Module):
    """
    Pad is python warpper for padding opr in megbrain, can padding in random one of the max 7 dimensions.
    Supported constant, edge(replicate) and reflect mode, constatnt is the default mode.
    """

    def __init__(
        self,
        pad_witdth: Tuple[Tuple[int, int], ...],
        mode: str = "constant",
        constant_val: float = 0.0,
    ):
        super().__init__()
        self.pad_width = pad_witdth
        self.mode = mode
        self.pad_val = constant_val

    def forward(self, src):
        return nn.pad(
            src, pad_witdth=self.pad_width, mode=self.mode, constant_value=self.pad_val
        )
