from ... import functional as F
from .. import conv as Float
from .module import QATModule


class Conv2d(Float.Conv2d, QATModule):
    r"""A :class:`~.QATModule` :class:`~.module.Conv2d` with QAT support.
    Could be applied with :class:`~.Observer` and :class:`~.quantization.fake_quant.FakeQuantize`.
    """

    def calc_conv_qat(self, inp):
        w_qat = self.apply_quant_weight(self.weight)
        b_qat = self.apply_quant_bias(self.bias, inp, w_qat)
        conv = self.calc_conv(inp, w_qat, b_qat)
        return conv

    @classmethod
    def from_float_module(cls, float_module: Float.Conv2d):
        r"""
        Return a :class:`~.QATModule` instance converted from
        a float :class:`~.Module` instance.
        """
        qat_module = cls(
            float_module.in_channels,
            float_module.out_channels,
            float_module.kernel_size,
            float_module.stride,
            float_module.padding,
            float_module.dilation,
            float_module.groups,
            float_module.bias is not None,
            float_module.conv_mode,
            float_module.compute_mode,
            float_module.padding_mode,
            name=float_module.name,
        )
        qat_module.weight = float_module.weight
        qat_module.bias = float_module.bias
        return qat_module

    def forward(self, inp):
        return self.apply_quant_activation(self.calc_conv_qat(inp))


class ConvRelu2d(Conv2d):
    r"""A :class:`~.QATModule` include :class:`~.module.Conv2d` and :func:`~.relu` with QAT support.
    Could be applied with :class:`~.Observer` and :class:`~.quantization.fake_quant.FakeQuantize`.
    """

    def forward(self, inp):
        return self.apply_quant_activation(F.relu(self.calc_conv_qat(inp)))


class ConvTranspose2d(Float.ConvTranspose2d, QATModule):
    r"""A :class:`~.QATModule` :class:`~.module.ConvTranspose2d` with QAT support.
    Could be applied with :class:`~.Observer` and :class:`~.quantization.fake_quant.FakeQuantize`.
    """

    def calc_conv_transpose2d_qat(self, inp):
        w_qat = self.apply_quant_weight(self.weight)
        b_qat = self.apply_quant_bias(self.bias, inp, w_qat)
        conv_transpose2d = self.calc_conv_transpose2d(inp, w_qat, b_qat)
        return conv_transpose2d

    @classmethod
    def from_float_module(cls, float_module: Float.ConvTranspose2d):
        r"""
        Return a :class:`~.QATModule` instance converted from
        a float :class:`~.Module` instance.
        """
        qat_module = cls(
            float_module.in_channels,
            float_module.out_channels,
            float_module.kernel_size,
            float_module.stride,
            float_module.padding,
            float_module.output_padding,
            float_module.dilation,
            float_module.groups,
            float_module.bias is not None,
            float_module.conv_mode,
            float_module.compute_mode,
            name=float_module.name,
        )
        qat_module.weight = float_module.weight
        qat_module.bias = float_module.bias
        return qat_module

    def forward(self, inp):
        return self.apply_quant_activation(self.calc_conv_transpose2d_qat(inp))


class ConvTransposeRelu2d(ConvTranspose2d):
    r"""A :class:`~.QATModule` include :class:`~.module.ConvTranspose2d` and :func:`~.relu` with QAT support.
    Could be applied with :class:`~.Observer` and :class:`~.quantization.fake_quant.FakeQuantize`.
    """

    def forward(self, inp):
        return self.apply_quant_activation(F.relu(self.calc_conv_transpose2d_qat(inp)))
