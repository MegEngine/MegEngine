from typing import Tuple

from ..functional import nn
from .module import Module


class Pad(Module):
    def __init__(
        self,
        pad_witdth: Tuple[Tuple[int, int], ...],
        mode: str = "CONSTANT",
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
