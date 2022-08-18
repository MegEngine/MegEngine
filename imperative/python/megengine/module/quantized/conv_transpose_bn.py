from ...tensor import Parameter
from ..qat import conv_transpose_bn as QAT
from .conv import ConvTranspose2d


class _ConvTransposeBnActivation2d(ConvTranspose2d):
    r"""Applies a 2D deconvolution over a quantized input tensor, used for inference only.
    """

    @classmethod
    def from_qat_module(cls, qat_module: QAT._ConvTransposeBnActivation2d):
        r"""
        Return a :class:`~.QuantizedModule` instance converted from a
        :class:`~.QATModule` instance.
        """
        output_dtype = qat_module.get_activation_dtype()
        qconv_transpose2d = cls(
            qat_module.conv_transpose2d.in_channels,
            qat_module.conv_transpose2d.out_channels,
            qat_module.conv_transpose2d.kernel_size,
            qat_module.conv_transpose2d.stride,
            qat_module.conv_transpose2d.padding,
            qat_module.conv_transpose2d.output_padding,
            qat_module.conv_transpose2d.dilation,
            qat_module.conv_transpose2d.groups,
            dtype=output_dtype,
            name=qat_module.name,
        )
        w_fold, b_fold = qat_module.fold_weight_bias(
            qat_module.bn.running_mean, qat_module.bn.running_var
        )
        weight = w_fold.astype(qat_module.get_weight_dtype())
        qconv_transpose2d.weight = Parameter(
            weight.numpy(), name=qat_module.conv_transpose2d.weight.name
        )
        qconv_transpose2d.bias = Parameter(b_fold.numpy())
        if qat_module.conv_transpose2d.bias is not None:
            qconv_transpose2d.bias.name = qat_module.conv_transpose2d.bias.name
        return qconv_transpose2d


class ConvTransposeBn2d(_ConvTransposeBnActivation2d):
    r"""Quantized version of :class:`~.qat.ConvTransposeBn2d`."""

    def forward(self, inp):
        return self.calc_conv_transpose2d_quantized(inp, nonlinear_mode="identity")


class ConvTransposeBnRelu2d(_ConvTransposeBnActivation2d):
    r"""Quantized version of :class:`~.qat.ConvTransposeBnRelu2d`."""

    def forward(self, inp):
        return self.calc_conv_transpose2d_quantized(inp, nonlinear_mode="relu")
