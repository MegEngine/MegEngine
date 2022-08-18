from abc import abstractmethod
from typing import Tuple, Union

import numpy as np

from ..functional import (
    conv1d,
    conv2d,
    conv3d,
    conv_transpose2d,
    conv_transpose3d,
    deformable_conv2d,
    local_conv2d,
    pad,
    region_restricted_conv,
    relu,
)
from ..tensor import Parameter
from ..utils.tuple_function import _pair, _pair_nonzero, _triple, _triple_nonzero
from . import init
from .module import Module


class _ConvNd(Module):
    """base class for convolution modules, including transposed conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        output_padding: Union[int, Tuple[int, int]],
        dilation: Union[int, Tuple[int, int]],
        groups: int,
        bias: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

        self.weight = Parameter(np.zeros(self._infer_weight_shape(), dtype=np.float32))
        self.bias = None
        if bias:
            self.bias = Parameter(np.zeros(self._infer_bias_shape(), dtype=np.float32))
        self.reset_parameters()

    @abstractmethod
    def _get_fanin(self):
        pass

    def reset_parameters(self) -> None:
        fanin = self._get_fanin()
        std = np.sqrt(1 / fanin)
        init.normal_(self.weight, 0.0, std)
        if self.bias is not None:
            init.zeros_(self.bias)

    @abstractmethod
    def _infer_weight_shape(self):
        pass

    @abstractmethod
    def _infer_bias_shape(self):
        pass

    def _module_info_string(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}"

        if self.stride != (1,) * len(self.stride):
            s += ", stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)


class Conv1d(_ConvNd):

    r"""Applies a 1D convolution over an input tensor.

    For instance, given an input of the size :math:`(N, C_{\text{in}}, H)`,
    this layer generates an output of the size
    :math:`(N, C_{\text{out}}, H_{\text{out}})` through the
    process described as below:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)

    where :math:`\star` is the valid 1D cross-correlation operator,
    :math:`N` is batch size, :math:`C` denotes number of channels, and
    :math:`H` is length of 1D data element.

    When `groups == in_channels` and `out_channels == K * in_channels`,
    where K is a positive integer, this operation is also known as depthwise
    convolution.

    In other words, for an input of size :math:`(N, C_{in}, H_{in})`,
    a depthwise convolution with a depthwise multiplier `K`, can be constructed
    by arguments :math:`(in\_channels=C_{in}, out\_channels=C_{in} \times K, ..., groups=C_{in})`.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: size of weight on spatial dimensions.
        stride: stride of the 1D convolution operation.
        padding: size of the paddings added to the input on both sides of its
            spatial dimensions. Default: 0
        dilation: dilation of the 1D convolution operation. Default: 1
        groups: number of groups to divide input and output channels into,
            so as to perform a "grouped convolution". When ``groups`` is not 1,
            ``in_channels`` and ``out_channels`` must be divisible by ``groups``,
            and the shape of weight should be ``(groups, out_channel // groups,
            in_channels // groups, kernel_size)``. Default: 1
        bias: whether to add a bias onto the result of convolution. Default: True
        conv_mode: Supports `cross_correlation`. Default: `cross_correlation`
        compute_mode: When set to "default", no special requirements will be
            placed on the precision of intermediate results. When set to "float32",
            "float32" would be used for accumulator and intermediate result, but only
            effective when input and output are of float16 dtype.
        padding_mode: "zeros", "reflect" or "replicate". Default: "zeros".
            Refer to :class:`~.module.padding.Pad` for more information.

    Note:
        * ``weight`` usually has shape ``(out_channels, in_channels, kernel_size)`` ,
          if groups is not 1, shape will be ``(groups, out_channels // groups, in_channels // groups, kernel_size)``
        * ``bias`` usually has shape ``(1, out_channels, 1)``

    Examples:
        >>> import numpy as np
        >>> m = M.Conv1d(in_channels=3, out_channels=1, kernel_size=3)
        >>> inp = mge.tensor(np.arange(0, 24).astype("float32").reshape(2, 3, 4))
        >>> oup = m(inp)
        >>> oup.numpy().shape
        (2, 1, 2)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        conv_mode: str = "cross_correlation",
        compute_mode: str = "default",
        padding_mode: str = "zeros",
        **kwargs
    ):
        kernel_size = kernel_size
        stride = stride
        padding = padding
        dilation = dilation
        self.conv_mode = conv_mode
        self.compute_mode = compute_mode
        self.padding_mode = padding_mode
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            0,
            dilation,
            groups,
            bias,
            **kwargs,
        )

    def _get_fanin(self):
        kh = self.kernel_size
        ic = self.in_channels
        return kh * ic

    def _infer_weight_shape(self):
        group = self.groups
        ichl = self.in_channels
        ochl = self.out_channels
        kh = self.kernel_size
        if group == 1:
            # Assume format is NCH(W=1)
            return (ochl, ichl, kh)

        assert (
            ichl % group == 0 and ochl % group == 0
        ), "invalid config: in_channels={} out_channels={} group={}".format(
            ichl, ochl, group
        )
        # Assume format is NCH(W=1)
        return (group, ochl // group, ichl // group, kh)

    def _infer_bias_shape(self):
        # Assume format is NCH(W=1)
        return (1, self.out_channels, 1)

    def get_pad_witdth(self):
        return ((0, 0), (0, 0), (self.padding, self.padding))

    def calc_conv(self, inp, weight, bias):
        assert self.padding_mode in [
            "zeros",
            "reflect",
            "replicate",
        ]
        if self.padding_mode != "zeros":
            return conv1d(
                pad(inp, self.get_pad_witdth(), self.padding_mode),
                weight,
                bias,
                self.stride,
                0,
                self.dilation,
                self.groups,
                self.conv_mode,
                self.compute_mode,
            )
        return conv1d(
            inp,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.conv_mode,
            self.compute_mode,
        )

    def forward(self, inp):
        return self.calc_conv(inp, self.weight, self.bias)


class Conv2d(_ConvNd):
    r"""Applies a 2D convolution over an input tensor.

    For instance, given an input of the size :math:`(N, C_{\text{in}}, H, W)`,
    this layer generates an output of the size
    :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})` through the
    process described as below:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)

    where :math:`\star` is the valid 2D cross-correlation operator,
    :math:`N` is batch size, :math:`C` denotes number of channels,
    :math:`H` is height of input planes in pixels, and :math:`W` is
    width in pixels.

    In general, output feature maps' shapes can be inferred as follows:

    input: :math:`(N, C_{\text{in}}, H_{\text{in}}, W_{\text{in}})`

    output: :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})` where

    .. math::
        \text{H}_{out} = \lfloor \frac{\text{H}_{in} + 2 * \text{padding[0]} -
        \text{dilation[0]} * (\text{kernel_size[0]} - 1) - 1}{\text{stride[0]}} + 1 \rfloor

    .. math::
        \text{W}_{out} = \lfloor \frac{\text{W}_{in} + 2 * \text{padding[1]} -
        \text{dilation[1]} * (\text{kernel_size[1]} - 1) - 1}{\text{stride[1]}} + 1 \rfloor

    When `groups == in_channels` and `out_channels == K * in_channels`,
    where K is a positive integer, this operation is also known as depthwise
    convolution.

    In other words, for an input of size :math:`(N, C_{in}, H_{in}, W_{in})`,
    a depthwise convolution with a depthwise multiplier `K`, can be constructed
    by arguments :math:`(in\_channels=C_{in}, out\_channels=C_{in} \times K, ..., groups=C_{in})`.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: size of weight on spatial dimensions. If kernel_size is
            an :class:`int`, the actual kernel size would be
            ``(kernel_size, kernel_size)``.
        stride: stride of the 2D convolution operation. Default: 1
        padding: size of the paddings added to the input on both sides of its
            spatial dimensions. Default: 0
        dilation: dilation of the 2D convolution operation. Default: 1
        groups: number of groups into which the input and output channels are divided,
            so as to perform a ``grouped convolution``. When ``groups`` is not 1,
            ``in_channels`` and ``out_channels`` must be divisible by ``groups``,
            and the shape of weight should be ``(groups, out_channel // groups,
            in_channels // groups, height, width)``. Default: 1
        bias: whether to add a bias onto the result of convolution. Default: True
        conv_mode: Supports `cross_correlation`. Default: `cross_correlation`
        compute_mode: When set to "default", no special requirements will be
            placed on the precision of intermediate results. When set to "float32",
            "float32" would be used for accumulator and intermediate result, but only
            effective when input and output are of float16 dtype.
        padding_mode: "zeros", "reflect" or "replicate". Default: "zeros".
            Refer to :class:`~.module.padding.Pad` for more information.

    Note:
        * ``weight`` usually has shape ``(out_channels, in_channels, height, width)`` ,
            if groups is not 1, shape will be ``(groups, out_channels // groups, in_channels // groups, height, width)``
        * ``bias`` usually has shape ``(1, out_channels, *1)``

    Examples:
        >>> import numpy as np
        >>> m = M.Conv2d(in_channels=3, out_channels=1, kernel_size=3)
        >>> inp = mge.tensor(np.arange(0, 96).astype("float32").reshape(2, 3, 4, 4))
        >>> oup = m(inp)
        >>> oup.numpy().shape
        (2, 1, 2, 2)
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
        bias: bool = True,
        conv_mode: str = "cross_correlation",
        compute_mode: str = "default",
        padding_mode: str = "zeros",
        **kwargs
    ):
        kernel_size = _pair_nonzero(kernel_size)
        stride = _pair_nonzero(stride)
        padding = _pair(padding)
        dilation = _pair_nonzero(dilation)
        self.conv_mode = conv_mode
        self.compute_mode = compute_mode
        self.padding_mode = padding_mode
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            0,
            dilation,
            groups,
            bias,
            **kwargs,
        )

    def _get_fanin(self):
        kh, kw = self.kernel_size
        ic = self.in_channels
        return kh * kw * ic

    def _infer_weight_shape(self):
        group = self.groups
        ichl = self.in_channels
        ochl = self.out_channels
        kh, kw = self.kernel_size
        if group == 1:
            # Assume format is NCHW
            return (ochl, ichl, kh, kw)

        assert (
            ichl % group == 0 and ochl % group == 0
        ), "invalid config: in_channels={} out_channels={} group={}".format(
            ichl, ochl, group
        )
        # Assume format is NCHW
        return (group, ochl // group, ichl // group, kh, kw)

    def _infer_bias_shape(self):
        # Assume format is NCHW
        return (1, self.out_channels, 1, 1)

    def get_pad_witdth(self):
        return (
            (0, 0),
            (0, 0),
            (self.padding[0], self.padding[0]),
            (self.padding[1], self.padding[1]),
        )

    def calc_conv(self, inp, weight, bias):
        assert self.padding_mode in [
            "zeros",
            "reflect",
            "replicate",
        ]
        if self.padding_mode != "zeros":
            return conv2d(
                pad(inp, self.get_pad_witdth(), self.padding_mode),
                weight,
                bias,
                self.stride,
                0,
                self.dilation,
                self.groups,
                self.conv_mode,
                self.compute_mode,
            )
        return conv2d(
            inp,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.conv_mode,
            self.compute_mode,
        )

    def forward(self, inp):
        return self.calc_conv(inp, self.weight, self.bias)


class Conv3d(_ConvNd):

    r"""Applies a 3D convolution over an input tensor.

    For instance, given an input of the size :math:`(N, C_{\text{in}}, T, H, W)`,
    this layer generates an output of the size
    :math:`(N, C_{\text{out}}, T_{\text{out}}, H_{\text{out}}, W_{\text{out}})` through the
    process described as below:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)

    where :math:`\star` is the valid 3D cross-correlation operator,
    :math:`N` is batch size, :math:`C` denotes number of channels.

    When `groups == in_channels` and `out_channels == K * in_channels`,
    where K is a positive integer, this operation is also known as depthwise
    convolution.

    In other words, for an input of size :math:`(N, C_{in}, T_{int}, H_{in}, W_{in})`,
    a depthwise convolution with a depthwise multiplier `K`, can be constructed
    by arguments :math:`(in\_channels=C_{in}, out\_channels=C_{in} \times K, ..., groups=C_{in})`.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: size of weight on spatial dimensions. If kernel_size is
            an :class:`int`, the actual kernel size would be
            `(kernel_size, kernel_size, kernel_size)`.
        stride: stride of the 3D convolution operation. Default: 1
        padding: size of the paddings added to the input on both sides of its
            spatial dimensions. Only zero-padding is supported. Default: 0
        dilation: dilation of the 3D convolution operation. Default: 1
        groups: number of groups into which the input and output channels are divided,
            so as to perform a ``grouped convolution``. When ``groups`` is not 1,
            ``in_channels`` and ``out_channels`` must be divisible by ``groups``,
            and the shape of weight should be ``(groups, out_channel // groups,
            in_channels // groups, depth, height, width)``. Default: 1
        bias: whether to add a bias onto the result of convolution. Default: True
        conv_mode: Supports `cross_correlation`. Default: `cross_correlation`

    Note:
        * ``weight`` usually has shape ``(out_channels, in_channels, depth, height, width)`` ,
          if groups is not 1, shape will be ``(groups, out_channels // groups, in_channels // groups, depth, height, width)``
        * ``bias`` usually has shape ``(1, out_channels, *1)``

    Examples:
        >>> import numpy as np
        >>> m = M.Conv3d(in_channels=3, out_channels=1, kernel_size=3)
        >>> inp = mge.tensor(np.arange(0, 384).astype("float32").reshape(2, 3, 4, 4, 4))
        >>> oup = m(inp)
        >>> oup.numpy().shape
        (2, 1, 2, 2, 2)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        conv_mode: str = "cross_correlation",
    ):
        kernel_size = _triple_nonzero(kernel_size)
        stride = _triple_nonzero(stride)
        padding = _triple(padding)
        dilation = _triple_nonzero(dilation)
        self.conv_mode = conv_mode
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            0,
            dilation,
            groups,
            bias,
        )

    def _get_fanin(self):
        kt, kh, kw = self.kernel_size
        ic = self.in_channels
        return kt * kh * kw * ic

    def _infer_weight_shape(self):
        group = self.groups
        ichl = self.in_channels
        ochl = self.out_channels
        kt, kh, kw = self.kernel_size
        if group == 1:
            # Assume format is NCTHW
            return (ochl, ichl, kt, kh, kw)

        assert (
            ichl % group == 0 and ochl % group == 0
        ), "invalid config: in_channels={} out_channels={} group={}".format(
            ichl, ochl, group
        )
        # Assume format is NCTHW
        return (group, ochl // group, ichl // group, kt, kh, kw)

    def _infer_bias_shape(self):
        # Assume format is NCTHW
        return (1, self.out_channels, 1, 1, 1)

    def calc_conv(self, inp, weight, bias):
        return conv3d(
            inp,
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.conv_mode,
        )

    def forward(self, inp):
        return self.calc_conv(inp, self.weight, self.bias)


class ConvTranspose2d(_ConvNd):
    r"""Applies a 2D transposed convolution over an input tensor.

    This module is also known as a deconvolution or a fractionally-strided convolution.
    :class:`ConvTranspose2d` can be seen as the gradient of :class:`Conv2d` operation
    with respect to its input.

    Convolution usually reduces the size of input, while transposed convolution works
    the opposite way, transforming a smaller input to a larger output while preserving the
    connectivity pattern.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: size of weight on spatial dimensions. If ``kernel_size`` is
            an :class:`int`, the actual kernel size would be
            ``(kernel_size, kernel_size)``.
        stride: stride of the 2D convolution operation. Default: 1
        padding: size of the paddings added to the input on both sides of its
            spatial dimensions. Only zero-padding is supported. Default: 0
        output_padding: size of paddings appended to output. Default: 0
        dilation: dilation of the 2D convolution operation. Default: 1
        groups: number of groups into which the input and output channels are divided,
            so as to perform a ``grouped convolution``. When ``groups`` is not 1,
            ``in_channels`` and ``out_channels`` must be divisible by groups,
            and the shape of weight should be ``(groups, in_channels // groups,
            out_channels // groups, height, width)``. Default: 1
        bias: wether to add a bias onto the result of convolution. Default: True
            conv_mode: Supports `cross_correlation`. Default: `cross_correlation`
        compute_mode: When set to "default", no special requirements will be
            placed on the precision of intermediate results. When set to "float32",
            "float32" would be used for accumulator and intermediate result, but only
            effective when input and output are of float16 dtype.

    Note:
        * ``weight`` usually has shape ``(in_channels, out_channels, height, width)`` ,
          if groups is not 1, shape will be ``(groups, in_channels // groups, out_channels // groups, height, width)``
        * ``bias`` usually has shape ``(1, out_channels, *1)``
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
        **kwargs
    ):
        kernel_size = _pair_nonzero(kernel_size)
        stride = _pair_nonzero(stride)
        padding = _pair(padding)
        output_padding = _pair(output_padding)
        dilation = _pair_nonzero(dilation)
        self.conv_mode = conv_mode
        self.compute_mode = compute_mode
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            bias,
            **kwargs,
        )

    def _get_fanin(self):
        kh, kw = self.kernel_size
        oc = self.out_channels
        return kh * kw * oc

    def _infer_weight_shape(self):
        group = self.groups
        ichl = self.in_channels
        ochl = self.out_channels
        kh, kw = self.kernel_size
        if group == 1:
            # Assume format is NCHW
            return (ichl, ochl, kh, kw)

        assert (
            ichl % group == 0 and ochl % group == 0
        ), "invalid config: in_channels={} out_channels={} group={}".format(
            ichl, ochl, group
        )
        # Assume format is NCHW
        return (group, ichl // group, ochl // group, kh, kw)

    def _infer_bias_shape(self):
        # Assume format is NCHW
        return (1, self.out_channels, 1, 1)

    def calc_conv_transpose2d(self, inp, weight, bias):
        return conv_transpose2d(
            inp,
            weight,
            bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
            self.conv_mode,
            self.compute_mode,
        )

    def forward(self, inp):
        return self.calc_conv_transpose2d(inp, self.weight, self.bias)


class LocalConv2d(Conv2d):
    r"""Applies a spatial convolution with untied kernels over an groupped channeled input 4D tensor.
    It is also known as the locally connected layer.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        input_height: the height of the input images.
        input_width: the width of the input images.
        kernel_size: size of weight on spatial dimensions. If kernel_size is
            an :class:`int`, the actual kernel size would be
            ``(kernel_size, kernel_size)``.
        stride: stride of the 2D convolution operation. Default: 1
        padding: size of the paddings added to the input on both sides of its
            spatial dimensions. Only zero-padding is supported. Default: 0
        dilation: dilation of the 2D convolution operation. Default: 1
        groups: number of groups into which the input and output channels are divided,
            so as to perform a "grouped convolution". When ``groups`` is not 1,
            ``in_channels`` and ``out_channels`` must be divisible by ``groups``. Default: 1

    Note:
        * ``weight`` usually has shape ``(out_height, out_width, in_channels, height, width, in_channels)`` ,
          if groups is not 1, shape will be ``(groups, out_height, out_width, in_channels // groups, height, width, out_channels // groups)``
        * ``bias`` usually has shape ``(1, out_channels, *1)``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_height: int,
        input_width: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        conv_mode: str = "cross_correlation",
        **kwargs
    ):
        self.input_height = input_height
        self.input_width = input_width
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=False,
            **kwargs,
        )

    def _infer_weight_shape(self):
        group = self.groups
        out_height = (
            self.input_height + self.padding[0] * 2 - self.kernel_size[0]
        ) // self.stride[0] + 1
        out_width = (
            self.input_width + self.padding[1] * 2 - self.kernel_size[1]
        ) // self.stride[1] + 1
        # Assume format is NCHW
        return (
            group,
            out_height,
            out_width,
            self.in_channels // group,
            self.kernel_size[0],
            self.kernel_size[1],
            self.out_channels // group,
        )

    def forward(self, inp):
        return local_conv2d(
            inp,
            self.weight,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.conv_mode,
        )


class ConvRelu2d(Conv2d):
    r"""A fused :class:`~.Module` including :class:`~.module.Conv2d` and :func:`~.relu`.
    Could be replaced with :class:`~.QATModule` version :class:`~.qat.ConvRelu2d` using :func:`~.quantize.quantize_qat`.
    """

    def forward(self, inp):
        return relu(self.calc_conv(inp, self.weight, self.bias))


class ConvTransposeRelu2d(ConvTranspose2d):
    r"""A fused :class:`~.Module` including :class:`~.module.ConvTranspose2d` and :func:`~.relu`.
    Could be replaced with :class:`~.QATModule` version :class:`~.qat.ConvTransposeRelu2d` using :func:`~.quantize.quantize_qat`.
    """

    def forward(self, inp):
        return relu(self.calc_conv_transpose2d(inp, self.weight, self.bias))


class DeformableConv2d(_ConvNd):
    r"""Deformable Convolution.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: size of weight on spatial dimensions. If kernel_size is
            an :class:`int`, the actual kernel size would be
            ``(kernel_size, kernel_size)``.
        stride: stride of the 2D convolution operation. Default: 1
        padding: size of the paddings added to the input on both sides of its
            spatial dimensions. Only zero-padding is supported. Default: 0
        dilation: dilation of the 2D convolution operation. Default: 1
        groups: number of groups into which the input and output channels are divided,
            so as to perform a ``grouped convolution``. When ``groups`` is not 1,
            ``in_channels`` and ``out_channels`` must be divisible by groups,
            and the shape of weight should be ``(groups, out_channel // groups,
            in_channels // groups, height, width)``. Default: 1
        bias: whether to add a bias onto the result of convolution. Default: True
        conv_mode: Supports `cross_correlation`. Default: `cross_correlation`
        compute_mode: When set to "default", no special requirements will be
            placed on the precision of intermediate results. When set to "float32",
            "float32" would be used for accumulator and intermediate result, but only
            effective when input and output are of float16 dtype.

    Note:
        * ``weight`` usually has shape ``(out_channels, in_channels, height, width)`` ,
          if groups is not 1, shape will be ``(groups, out_channels // groups, in_channels // groups, height, width)``
        * ``bias`` usually has shape ``(1, out_channels, *1)``
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
        bias: bool = True,
        conv_mode: str = "cross_correlation",
        compute_mode: str = "default",
        **kwargs
    ):
        kernel_size = _pair_nonzero(kernel_size)
        stride = _pair_nonzero(stride)
        padding = _pair(padding)
        dilation = _pair_nonzero(dilation)
        self.conv_mode = conv_mode
        self.compute_mode = compute_mode
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            0,
            dilation,
            groups,
            bias,
            **kwargs,
        )

    def _get_fanin(self):
        kh, kw = self.kernel_size
        ic = self.in_channels
        return kh * kw * ic

    def _infer_weight_shape(self):
        group = self.groups
        ichl = self.in_channels
        ochl = self.out_channels
        kh, kw = self.kernel_size
        if group == 1:
            # Assume format is NCHW
            return (ochl, ichl, kh, kw)

        assert (
            ichl % group == 0 and ochl % group == 0
        ), "invalid config: in_channels={} out_channels={} group={}".format(
            ichl, ochl, group
        )
        # Assume format is NCHW
        return (group, ochl // group, ichl // group, kh, kw)

    def _infer_bias_shape(self):
        # Assume format is NCHW
        return (1, self.out_channels, 1, 1)

    def calc_conv(self, inp, weight, offset, mask, bias):
        return deformable_conv2d(
            inp,
            weight,
            offset,
            mask,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.conv_mode,
            self.compute_mode,
        )

    def forward(self, inp, offset, mask):
        return self.calc_conv(inp, self.weight, offset, mask, self.bias)


class ConvTranspose3d(_ConvNd):
    r"""Applies a 3D transposed convolution over an input tensor.

    Only support the case that groups = 1 and conv_mode = "cross_correlation".

    :class:`ConvTranspose3d` can be seen as the gradient of :class:`Conv3d` operation
    with respect to its input.

    Convolution3D usually reduces the size of input, while transposed convolution3d
    works the opposite way, transforming a smaller input to a larger output while
    preserving the connectivity pattern.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: size of weight on spatial dimensions. If ``kernel_size`` is
            an :class:`int`, the actual kernel size would be
            ``(kernel_size, kernel_size, kernel_size)``.
        stride: stride of the 3D convolution operation. Default: 1
        padding: size of the paddings added to the input on all sides of its
            spatial dimensions. Only zero-padding is supported. Default: 0
        output_padding: size of paddings appended to output. Default: 0
        dilation: dilation of the 3D convolution operation. Default: 1
        groups: number of groups into which the input and output channels are divided,
            so as to perform a ``grouped convolution``. When ``groups`` is not 1,
            ``in_channels`` and ``out_channels`` must be divisible by groups,
            and the shape of weight should be ``(groups, in_channels // groups,
            out_channels // groups, depth, height, width)``. Default: 1
        bias: wether to add a bias onto the result of convolution. Default: True

    Note:
        * ``weight`` usually has shape ``(in_channels, out_channels, depth, height, width)`` .
        * ``bias`` usually has shape ``(1, out_channels, *1)``
    """

    output_padding = 0

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        output_padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        kernel_size = _triple_nonzero(kernel_size)
        stride = _triple_nonzero(stride)
        padding = _triple(padding)
        dilation = _triple_nonzero(dilation)
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
        )

    def _get_fanin(self):
        kt, kh, kw = self.kernel_size
        ic = self.in_channels
        return kt * kh * kw * ic

    def _infer_weight_shape(self):
        group = self.groups
        ichl = self.in_channels
        ochl = self.out_channels
        kt, kh, kw = self.kernel_size
        if group == 1:
            # Assume format is NCHW
            return (ichl, ochl, kt, kh, kw)

        assert (
            ichl % group == 0 and ochl % group == 0
        ), "invalid config: in_channels={} out_channels={} group={}".format(
            ichl, ochl, group
        )
        # Assume format is NCHW
        return (group, ichl // group, ochl // group, kt, kh, kw)

    def _infer_bias_shape(self):
        # Assume format is NCTHW
        return (1, self.out_channels, 1, 1, 1)

    def forward(self, inp):
        return conv_transpose3d(
            inp,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
        )


class RegionRestrictedConv(_ConvNd):

    r"""Applies a 2D RegionRestricted Convolution over an input tensor.

    For instance, given an input of the size :math:`(N, C_{\text{in}}, H, W)`,
    this layer generates an output of the size
    :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})` through the
    process described as below:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) =
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)

    where :math:`\star` is the valid 2D cross-correlation operator,
    :math:`N` is batch size, :math:`C` denotes number of channels,
    :math:`H` is height of input planes in pixels, and :math:`W` is
    width in pixels.

    In general, output feature maps' shapes can be inferred as follows:

    input: :math:`(N, C_{\text{in}}, H_{\text{in}}, W_{\text{in}})`

    output: :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})` where

    .. math::
        \text{H}_{out} = \lfloor \frac{\text{H}_{in} + 2 * \text{padding[0]} -
        \text{dilation[0]} * (\text{kernel_size[0]} - 1) - 1}{\text{stride[0]}} + 1 \rfloor

    .. math::
        \text{W}_{out} = \lfloor \frac{\text{W}_{in} + 2 * \text{padding[1]} -
        \text{dilation[1]} * (\text{kernel_size[1]} - 1) - 1}{\text{stride[1]}} + 1 \rfloor

    When `groups == in_channels` and `out_channels == K * in_channels`,
    where K is a positive integer, this operation is also known as depthwise
    convolution.

    In other words, for an input of size :math:`(N, C_{in}, H_{in}, W_{in})`,
    a depthwise convolution with a depthwise multiplier `K`, can be constructed
    by arguments :math:`(in\_channels=C_{in}, out\_channels=C_{in} \times K, ..., groups=C_{in})`.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: size of weight on spatial dimensions. If kernel_size is
            an :class:`int`, the actual kernel size would be
            ``(kernel_size, kernel_size)``.
        stride: stride of the 2D convolution operation. Default: 1
        padding: size of the paddings added to the input on both sides of its
            spatial dimensions. Default: 0
        dilation: dilation of the 2D convolution operation. Default: 1
        groups: number of groups into which the input and output channels are divided,
            so as to perform a ``grouped convolution``. When ``groups`` is not 1,
            ``in_channels`` and ``out_channels`` must be divisible by ``groups``,
            and the shape of weight should be ``(groups, out_channel // groups,
            in_channels // groups, height, width)``. Default: 1
        bias: whether to add a bias onto the result of convolution. Default: True
        conv_mode: Supports `cross_correlation`. Default: `cross_correlation`
        compute_mode: When set to "default", no special requirements will be
            placed on the precision of intermediate results. When set to "float32",
            "float32" would be used for accumulator and intermediate result, but only
            effective when input and output are of float16 dtype.
        padding_mode: "zeros", "reflect" or "replicate". Default: "zeros".
            Refer to :class:`~.module.padding.Pad` for more information.

    Note:
        * weight shape will be ``(groups, out_channels // groups, in_channels // groups, height, width)``,
            becasue RegionRestrictedConv support grouped conv only.

    Examples:
        >>> import numpy as np
        >>> import megengine as mge
        >>> import megengine.module as M
        >>> rrconv = M.RegionRestrictedConv(in_channels=2, out_channels=2, kernel_size=2, groups=2)
        >>> inp = mge.tensor(np.random.randn(1, 2, 2, 2).astype(np.float32))
        >>> rin = mge.tensor(np.random.randn(1, 2, 2).astype(np.int32))
        >>> rout = mge.tensor(np.random.randn(1, 1, 1).astype(np.int32))
        >>> oup = rrconv(inp, rin, rout)
        >>> oup.numpy().shape
        (1, 2, 1, 1)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        groups: int = 1,
        bias: bool = True,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        conv_mode: str = "cross_correlation",
        compute_mode: str = "default",
        padding_mode: str = "zeros",
        **kwargs
    ):
        kernel_size = _pair_nonzero(kernel_size)
        stride = _pair_nonzero(stride)
        padding = _pair(padding)
        dilation = _pair_nonzero(dilation)
        self.conv_mode = conv_mode
        self.compute_mode = compute_mode
        self.padding_mode = padding_mode
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            0,
            dilation,
            groups,
            bias,
            **kwargs,
        )

    def _get_fanin(self):
        kh, kw = self.kernel_size
        ic = self.in_channels
        return kh * kw * ic

    def _infer_weight_shape(self):
        group = self.groups
        ichl = self.in_channels
        ochl = self.out_channels
        kh, kw = self.kernel_size

        assert (
            ichl % group == 0 and ochl % group == 0
        ), "invalid config: in_channels={} out_channels={} group={}".format(
            ichl, ochl, group
        )
        # Assume format is NCHW
        return (group, ochl // group, ichl // group, kh, kw)

    def _infer_bias_shape(self):
        # Assume format is NCHW
        return (1, self.out_channels, 1, 1)

    def get_pad_width(self):
        return (
            (0, 0),
            (0, 0),
            (self.padding[0], self.padding[0]),
            (self.padding[1], self.padding[1]),
        )

    def calc_conv(self, inp, weight, rin, rout, bias):
        assert self.padding_mode in [
            "zeros",
            "reflect",
            "replicate",
        ]
        return region_restricted_conv(
            inp,
            weight,
            rin,
            rout,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.conv_mode,
            self.compute_mode,
        )

    def forward(self, inp, rin, rout):
        return self.calc_conv(inp, self.weight, rin, rout, self.bias)
