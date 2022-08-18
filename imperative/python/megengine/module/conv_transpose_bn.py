from typing import Tuple, Union

from ..functional import relu
from .batchnorm import BatchNorm2d
from .conv import ConvTranspose2d
from .module import Module


class _ConvTransposeBnActivation2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        conv_mode: str = "cross_correlation",
        compute_mode: str = "default",
        eps=1e-5,
        momentum=0.9,
        affine=True,
        track_running_stats=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.conv_transpose2d = ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            bias,
            conv_mode,
            compute_mode,
            **kwargs,
        )
        self.bn = BatchNorm2d(out_channels, eps, momentum, affine, track_running_stats)


class ConvTransposeBn2d(_ConvTransposeBnActivation2d):
    r"""A fused :class:`~.Module` including :class:`~.module.ConvTranspose2d` and :class:`~.module.BatchNorm2d`.
    Could be replaced with :class:`~.QATModule` version :class:`~.qat.ConvTransposeBn2d` using:func:`~.quantize.quantize_qat`.
    """

    def forward(self, inp):
        return self.bn(self.conv_transpose2d(inp))


class ConvTransposeBnRelu2d(_ConvTransposeBnActivation2d):
    r"""A fused :class:`~.Module` including :class:`~.module.ConvTranspose2d`, :class:`~.module.BatchNorm2d` and :func:`~.relu`.
    Could be replaced with :class:`~.QATModule` version :class:`~.qat.ConvTransposeBnRelu2d` using :func:`~.quantize.quantize_qat`.
    """

    def forward(self, inp):
        return relu(self.bn(self.conv_transpose2d(inp)))
