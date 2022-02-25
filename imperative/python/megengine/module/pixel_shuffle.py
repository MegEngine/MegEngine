from ..functional.nn import pixel_shuffle
from .module import Module


class PixelShuffle(Module):
    r"""
    Rearranges elements in a tensor of shape (*, C x r^2, H, W) to a tensor of
    shape (*, C, H x r, W x r), where r is an upscale factor, where * is zero
    or more batch dimensions.
    """

    def __init__(self, upscale_factor: int, **kwargs):
        super().__init__(**kwargs)
        self.upscale_factor = upscale_factor

    def forward(self, x):
        return pixel_shuffle(x, self.upscale_factor)
