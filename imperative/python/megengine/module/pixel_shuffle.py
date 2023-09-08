from ..functional.nn import pixel_shuffle
from .module import Module


class PixelShuffle(Module):
    r"""
    Rearranges elements in a tensor of shape (*, C x r^2, H, W) to a tensor of
    shape (*, C, H x r, W x r), where r is an upscale factor, where * is zero
    or more batch dimensions.

    See the paper: `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network<https://arxiv.org/abs/1609.05158>` for more details.

    Args:
        upscale_factor(:class:`int`): factor to increase spatial resolution by.

    Shape:
        - input: :math:`(*, C_{in}, H_{in}, W_{in})`, where * is zero or more batch dimensions.
        - output: :math:`(*, C_{out}, H_{out}, W_{out})`, where:

            .. math::
            \begin{aligned}
                C_{out} = C_{in} \div \text{upscale\_factor}^2 \\
                H_{out} = H_{in} \times \text{upscale\_factor} \\
                W_{out} = W_{in} \times \text{upscale\_factor} \\
            \end{aligned}
        
    Examples:
        >>> import numpy as np
        >>> pixel_shuffle = M.PixelShuffle(3)
        >>> input = mge.tensor(np.random.randn(1, 9, 4, 4))
        >>> output = pixel_shuffle(input)
        >>> output.numpy().shape
        (1, 1, 12, 12)
    """

    def __init__(self, upscale_factor: int, **kwargs):
        super().__init__(**kwargs)
        self.upscale_factor = upscale_factor

    def forward(self, x):
        return pixel_shuffle(x, self.upscale_factor)
