from typing import Tuple, Union

import numpy as np

from ... import module as Float
from ...core.tensor import dtype
from ...functional.nn import conv_bias_activation, pad
from ...functional.quantized import conv_transpose2d
from ...tensor import Parameter
from ..qat import conv as QAT
from .module import QuantizedModule


class Conv2d(Float.Conv2d, QuantizedModule):
    r"""Quantized version of :class:`~.qat.Conv2d`.

    Applies a 2D convolution over a quantized input tensor, used for inference only.

    The parameter is same with :class:`~.module.Conv2d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        conv_mode: str = "cross_correlation",
        compute_mode: str = "default",
        dtype=None,
        padding_mode: str = "zeros",
        **kwargs
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            True,
            conv_mode,
            compute_mode,
            padding_mode,
        )
        self.output_dtype = dtype

    def calc_conv_quantized(self, inp, nonlinear_mode="identity"):
        assert self.padding_mode in [
            "zeros",
            "reflect",
            "replicate",
        ]
        inp_scale = dtype.get_scale(inp.dtype)
        w_scale = dtype.get_scale(self.weight.dtype)
        bias_scale = inp_scale * w_scale
        if self.padding_mode != "zeros":
            return conv_bias_activation(
                pad(inp, self.get_pad_witdth(), self.padding_mode),
                self.weight,
                self.bias.astype(dtype.qint32(bias_scale)),
                self.output_dtype,
                self.stride,
                0,
                self.dilation,
                self.groups,
                conv_mode=self.conv_mode,
                compute_mode=self.compute_mode,
                nonlinear_mode=nonlinear_mode,
            )
        return conv_bias_activation(
            inp,
            self.weight,
            self.bias.astype(dtype.qint32(bias_scale)),
            self.output_dtype,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            conv_mode=self.conv_mode,
            compute_mode=self.compute_mode,
            nonlinear_mode=nonlinear_mode,
        )

    @classmethod
    def from_qat_module(cls, qat_module: QAT.Conv2d):
        r"""
        Return a :class:`~.QuantizedModule` instance converted from a
        :class:`~.QATModule` instance.
        """
        output_dtype = qat_module.get_activation_dtype()
        qconv = cls(
            qat_module.in_channels,
            qat_module.out_channels,
            qat_module.kernel_size,
            qat_module.stride,
            qat_module.padding,
            qat_module.dilation,
            qat_module.groups,
            dtype=output_dtype,
            padding_mode=qat_module.padding_mode,
            name=qat_module.name,
        )
        weight = qat_module.weight.astype(qat_module.get_weight_dtype())
        qconv.weight = Parameter(weight.numpy(), name=qat_module.weight.name)
        if qat_module.bias is not None:
            qconv.bias = Parameter(qat_module.bias.numpy(), name=qat_module.bias.name)
        else:
            qconv.bias = Parameter(
                np.zeros(qat_module._infer_bias_shape(), dtype=np.float32)
            )
        return qconv

    def forward(self, inp):
        return self.calc_conv_quantized(inp, nonlinear_mode="identity")


class ConvRelu2d(Conv2d):
    r"""Quantized version of :class:`~.qat.ConvRelu2d`."""

    def forward(self, inp):
        return self.calc_conv_quantized(inp, nonlinear_mode="relu")


class ConvTranspose2d(Float.ConvTranspose2d, QuantizedModule):
    r"""Quantized version of :class:`~.qat.ConvTranspose2d`.

    Applies a 2D transposed convolution over a quantized input tensor, used
    for inference only.

    The parameter is same with :class:`~.module.ConvTranspose2d` but dtype.

    Args:
        dtype: data type of the output, should be qint8.
    """

    output_padding = 0

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
        dtype=None,
        **kwargs
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            conv_mode=conv_mode,
            compute_mode=compute_mode,
        )
        self.output_dtype = dtype

    @classmethod
    def from_qat_module(cls, qat_module: QAT.ConvTranspose2d):
        r"""
        return a :class:`~.QuantizedModule` instance converted from a
        :class:`~.QATModule` instance.
        """
        output_dtype = qat_module.get_activation_dtype()
        qconv_transpose2d = cls(
            qat_module.in_channels,
            qat_module.out_channels,
            qat_module.kernel_size,
            qat_module.stride,
            qat_module.padding,
            qat_module.output_padding,
            qat_module.dilation,
            qat_module.groups,
            qat_module.bias is not None,
            qat_module.conv_mode,
            qat_module.compute_mode,
            dtype=output_dtype,
            name=qat_module.name,
        )
        weight = qat_module.weight.astype(qat_module.get_weight_dtype())
        qconv_transpose2d.weight = Parameter(
            weight.numpy(), name=qat_module.weight.name
        )
        qconv_transpose2d.bias = (
            Parameter(qat_module.bias.numpy(), name=qat_module.bias.name)
            if qat_module.bias is not None
            else None
        )
        return qconv_transpose2d

    def calc_conv_transpose2d_quantized(self, inp, nonlinear_mode):
        assert nonlinear_mode == "identity", "nonlinear_mode shoule be 'identity'"

        if self.bias is not None:
            inp_scale = dtype.get_scale(inp.dtype)
            w_scale = dtype.get_scale(self.weight.dtype)
            bias_scale = inp_scale * w_scale

        return conv_transpose2d(
            inp=inp,
            weight=self.weight,
            bias=self.bias.astype(dtype.qint32(bias_scale))
            if self.bias is not None
            else None,
            dtype=self.output_dtype,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
            conv_mode=self.conv_mode,
            compute_mode=self.compute_mode,
        )

    def forward(self, inp):
        return self.calc_conv_transpose2d_quantized(inp, nonlinear_mode="identity")


class ConvTransposeRelu2d(ConvTranspose2d):
    r"""Quantized version of :class:`~.qat.ConvTransposeRelu2d`."""

    def forward(self, inp):
        return self.calc_conv_transpose2d_quantized(inp, nonlinear_mode="relu")
