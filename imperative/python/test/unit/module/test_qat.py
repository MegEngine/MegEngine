import io
from itertools import product

import numpy as np
import pytest

import megengine.utils.comp_graph_tools as cgtools
from megengine import jit
from megengine import module as M
from megengine import tensor
from megengine.device import get_device_count
from megengine.functional import expand_dims
from megengine.module import (
    BatchMatMulActivation,
    Conv2d,
    ConvBn2d,
    ConvRelu2d,
    ConvTranspose2d,
    ConvTransposeBn2d,
    ConvTransposeRelu2d,
    DequantStub,
    Linear,
    LinearBn1d,
    LinearBnRelu1d,
    LinearRelu,
    Module,
    QuantStub,
)
from megengine.quantization.quantize import (
    disable_fake_quant,
    enable_fake_quant,
    quantize,
    quantize_qat,
)


def test_qat_convbn2d():
    in_channels = 32
    out_channels = 64
    kernel_size = 3

    class TestNet(Module):
        def __init__(self, groups, bias):
            super().__init__()
            self.quant = QuantStub()
            self.dequant = DequantStub()
            self.conv_bn = ConvBn2d(
                in_channels, out_channels, kernel_size, groups=groups, bias=bias,
            )

        def forward(self, inp):
            out = self.quant(inp)
            out = self.conv_bn(out)
            out = self.dequant(out)
            return out

    inputs = tensor(np.random.randn(4, in_channels, 32, 32).astype(np.float32))
    for groups, bias in product([1, 4], [True, False]):
        net = TestNet(groups, bias)
        net.train()
        qat_net = quantize_qat(net, inplace=False)
        disable_fake_quant(qat_net)
        normal_outputs = net(inputs)
        qat_outputs = qat_net(inputs)
        np.testing.assert_allclose(
            normal_outputs.numpy(), qat_outputs.numpy(), atol=1e-4,
        )
        np.testing.assert_allclose(
            net.conv_bn.bn.running_mean.numpy(),
            qat_net.conv_bn.bn.running_mean.numpy(),
            atol=5e-8,
        )
        np.testing.assert_allclose(
            net.conv_bn.bn.running_var.numpy(),
            qat_net.conv_bn.bn.running_var.numpy(),
            atol=5e-7,
        )
        net.eval()
        normal_outputs = net(inputs)
        qat_net.eval()
        qat_outputs = qat_net(inputs)
        np.testing.assert_allclose(
            normal_outputs.numpy(), qat_outputs.numpy(), atol=1e-4,
        )


def test_qat_convtransposebn2d():
    in_channels = 32
    out_channels = 64
    kernel_size = 3

    class TestNet(Module):
        def __init__(self, groups, bias):
            super().__init__()
            self.quant = QuantStub()
            self.dequant = DequantStub()
            self.conv_transpose_bn = ConvTransposeBn2d(
                in_channels, out_channels, kernel_size, groups=groups, bias=bias,
            )

        def forward(self, inp):
            out = self.quant(inp)
            out = self.conv_transpose_bn(out)
            out = self.dequant(out)
            return out

    for groups, bias in product([1, 4], [True, False]):
        net = TestNet(groups, bias)
        net.train()
        qat_net = quantize_qat(net, inplace=False)
        disable_fake_quant(qat_net)
        inputs = tensor(np.random.randn(4, in_channels, 32, 32).astype(np.float32))
        normal_outputs = net(inputs)
        qat_outputs = qat_net(inputs)
        np.testing.assert_allclose(
            normal_outputs.numpy(), qat_outputs.numpy(), atol=1e-5,
        )
        np.testing.assert_allclose(
            net.conv_transpose_bn.bn.running_var.numpy(),
            qat_net.conv_transpose_bn.bn.running_var.numpy(),
            atol=5e-7,
        )
        net.eval()
        normal_outputs = net(inputs)
        qat_net.eval()
        qat_outputs = qat_net(inputs)
        np.testing.assert_allclose(
            normal_outputs.numpy(), qat_outputs.numpy(), atol=1e-5,
        )


@pytest.mark.parametrize(
    "padding, padding_mode",
    [
        (0, "zeros"),
        ((1, 2), "zeros"),
        (3, "reflect"),
        ((1, 2), "reflect"),
        (4, "replicate"),
        ((1, 2), "replicate"),
    ],
)
def test_qat_conv(padding, padding_mode):

    in_channels = 32
    out_channels = 64
    kernel_size = 3

    class TestNet(Module):
        def __init__(self, groups, bias):
            super().__init__()
            self.quant = QuantStub()
            self.dequant = DequantStub()
            self.conv = Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                groups=groups,
                bias=bias,
                padding=padding,
                padding_mode=padding_mode,
            )
            self.conv_relu = ConvRelu2d(
                out_channels, in_channels, kernel_size, groups=groups, bias=bias
            )

        def forward(self, inp):
            out = self.quant(inp)
            out = self.conv(out)
            out = self.conv_relu(out)
            out = self.dequant(out)
            return out

    inputs = tensor(np.random.randn(4, in_channels, 32, 32).astype(np.float32))
    for groups, bias in product([1, 4], [True, False]):
        net = TestNet(groups, bias)
        net.train()
        qat_net = quantize_qat(net, inplace=False)
        disable_fake_quant(qat_net)
        normal_outputs = net(inputs)
        qat_outputs = qat_net(inputs)
        np.testing.assert_allclose(normal_outputs.numpy(), qat_outputs.numpy())

        net.eval()
        normal_outputs = net(inputs)
        qat_net.eval()
        qat_outputs = qat_net(inputs)
        np.testing.assert_allclose(normal_outputs.numpy(), qat_outputs.numpy())


@pytest.mark.skipif(get_device_count("gpu") > 0, reason="no int8 algorithm on cuda")
def test_qat_batchmatmul_activation():
    batch = 4
    in_features = 8
    out_features = 4

    class TestNet(Module):
        def __init__(self, bias):
            super().__init__()
            self.quant = QuantStub()
            self.dequant = DequantStub()
            self.batch_mm = BatchMatMulActivation(
                batch, in_features, out_features, bias=bias
            )

        def forward(self, inp):
            out = self.quant(inp)
            out = self.batch_mm(out)
            out = self.dequant(out)
            return out

    inputs = tensor(
        np.random.randn(batch, in_features, out_features).astype(np.float32)
    )
    for bias in (True, False):
        net = TestNet(bias)
        net.train()
        qat_net = quantize_qat(net, inplace=False)
        disable_fake_quant(qat_net)
        normal_outputs = net(inputs)
        qat_outputs = qat_net(inputs)
        np.testing.assert_allclose(normal_outputs.numpy(), qat_outputs.numpy())

        net.eval()
        normal_outputs = net(inputs)
        qat_net.eval()
        qat_outputs = qat_net(inputs)
        np.testing.assert_allclose(normal_outputs.numpy(), qat_outputs.numpy())


@pytest.mark.skip(reason="FIXME: abnormal exit")
def test_quantize_batchmatmul_activation():
    batch = 4
    in_features = 8
    out_features = 4

    class TestNet(Module):
        def __init__(self, bias):
            super().__init__()
            self.quant = QuantStub()
            self.dequant = DequantStub()
            self.batch_mm = BatchMatMulActivation(
                batch, in_features, out_features, bias=bias
            )

        def forward(self, inp):
            out = self.quant(inp)
            out = self.batch_mm(out)
            out = expand_dims(out, -1)
            out = self.dequant(out)
            return out

    inputs = tensor(
        np.random.randn(batch, in_features, out_features).astype(np.float32)
    )
    for bias in (True, False):
        net = TestNet(bias)
        net.train()
        qat_net = quantize_qat(net, inplace=False)
        disable_fake_quant(qat_net)
        normal_outputs = net(inputs)
        qat_outputs = qat_net(inputs)
        np.testing.assert_allclose(normal_outputs.numpy(), qat_outputs.numpy())

        net.eval()
        normal_outputs = net(inputs)
        qat_net.eval()
        qat_outputs = qat_net(inputs)
        np.testing.assert_allclose(normal_outputs.numpy(), qat_outputs.numpy())

        enable_fake_quant(qat_net)
        qat_outputs = qat_net(inputs)
        qnet = quantize(qat_net, inplace=False)
        qnet.eval()
        quantize_outputs = qnet(inputs)
        np.testing.assert_allclose(
            qat_outputs.numpy(), quantize_outputs.numpy(), atol=1e-6
        )

        @jit.trace(capture_as_const=True)
        def f(x):
            qnet.eval()
            return qnet(x)

        f(inputs)
        file = io.BytesIO()
        f.dump(file, enable_nchw4=True)
        file.seek(0)
        infer_cg = cgtools.GraphInference(file)[0]
        dumped_outputs = list(infer_cg.run(inputs.numpy()).values())[0]
        np.testing.assert_allclose(quantize_outputs.numpy(), dumped_outputs, atol=1e-6)


def test_qat_conv_transpose2d():
    in_channels = 32
    out_channels = 64
    kernel_size = 3

    class TestNet(Module):
        def __init__(self, bias):
            super().__init__()
            self.quant = QuantStub()
            self.dequant = DequantStub()
            self.conv = ConvTranspose2d(
                in_channels, out_channels, kernel_size, bias=bias
            )
            self.conv_transpose2d_relu = ConvTransposeRelu2d(
                out_channels, in_channels, kernel_size, bias=bias
            )

        def forward(self, inp):
            out = self.quant(inp)
            out = self.conv(out)
            out = self.conv_transpose2d_relu(out)
            out = self.dequant(out)
            return out

    inputs = tensor(np.random.randn(4, in_channels, 32, 32).astype(np.float32))
    for bias in [True, False]:
        net = TestNet(bias)
        net.train()
        qat_net = quantize_qat(net, inplace=False)
        disable_fake_quant(qat_net)
        normal_outputs = net(inputs)
        qat_outputs = qat_net(inputs)
        np.testing.assert_allclose(
            normal_outputs.numpy(), qat_outputs.numpy(), atol=1e-6
        )

        net.eval()
        normal_outputs = net(inputs)
        qat_net.eval()
        qat_outputs = qat_net(inputs)
        np.testing.assert_allclose(
            normal_outputs.numpy(), qat_outputs.numpy(), atol=1e-6
        )


def test_qat_linearbn1d():
    in_features = 10
    out_features = 5

    class TestNet(Module):
        def __init__(self, bias):
            super().__init__()
            self.quant = QuantStub()
            self.dequant = DequantStub()
            self.linear_bn = LinearBn1d(in_features, out_features, bias=bias,)

        def forward(self, inp):
            out = self.quant(inp)
            out = self.linear_bn(out)
            out = self.dequant(out)
            return out

    inputs = tensor(np.random.randn(4, in_features).astype(np.float32))
    for bias in [True, False]:
        net = TestNet(bias)
        net.train()
        qat_net = quantize_qat(net, inplace=False)
        disable_fake_quant(qat_net)
        normal_outputs = net(inputs)
        qat_outputs = qat_net(inputs)
        np.testing.assert_allclose(
            normal_outputs.numpy(), qat_outputs.numpy(), atol=1e-6,
        )
        np.testing.assert_allclose(
            net.linear_bn.bn.running_mean.numpy(),
            qat_net.linear_bn.bn.running_mean.numpy(),
            atol=5e-8,
        )
        np.testing.assert_allclose(
            net.linear_bn.bn.running_var.numpy(),
            qat_net.linear_bn.bn.running_var.numpy(),
            atol=5e-7,
        )

        net.eval()
        normal_outputs = net(inputs)
        qat_net.eval()
        qat_outputs = qat_net(inputs)
        np.testing.assert_allclose(
            normal_outputs.numpy(), qat_outputs.numpy(), atol=1e-6,
        )


def test_qat_linear_relu():
    in_features = 10
    out_features = 5

    class TestNet(Module):
        def __init__(self, bias):
            super().__init__()
            self.quant = QuantStub()
            self.dequant = DequantStub()
            self.linear_relu = LinearRelu(in_features, out_features, bias=bias,)

        def forward(self, inp):
            out = self.quant(inp)
            out = self.linear_relu(out)
            out = self.dequant(out)
            return out

    inputs = tensor(np.random.randn(4, in_features).astype(np.float32))
    for bias in [True, False]:
        net = TestNet(bias)
        net.train()
        qat_net = quantize_qat(net, inplace=False)
        disable_fake_quant(qat_net)
        normal_outputs = net(inputs)
        qat_outputs = qat_net(inputs)
        np.testing.assert_allclose(
            normal_outputs.numpy(), qat_outputs.numpy(), atol=1e-6,
        )

        net.eval()
        normal_outputs = net(inputs)
        qat_net.eval()
        qat_outputs = qat_net(inputs)
        np.testing.assert_allclose(
            normal_outputs.numpy(), qat_outputs.numpy(), atol=1e-6,
        )


def test_qat_linear_bn_relu():
    in_features = 10
    out_features = 5

    class TestNet(Module):
        def __init__(self, bias):
            super().__init__()
            self.quant = QuantStub()
            self.dequant = DequantStub()
            self.linear_bn_relu = LinearBnRelu1d(in_features, out_features, bias=bias,)

        def forward(self, inp):
            out = self.quant(inp)
            out = self.linear_bn_relu(out)
            out = self.dequant(out)
            return out

    inputs = tensor(np.random.randn(4, in_features).astype(np.float32))
    for bias in [True, False]:
        net = TestNet(bias)
        net.train()
        qat_net = quantize_qat(net, inplace=False)
        disable_fake_quant(qat_net)
        normal_outputs = net(inputs)
        qat_outputs = qat_net(inputs)
        np.testing.assert_allclose(
            normal_outputs.numpy(), qat_outputs.numpy(), atol=1e-6,
        )

        net.eval()
        normal_outputs = net(inputs)
        qat_net.eval()
        qat_outputs = qat_net(inputs)
        np.testing.assert_allclose(
            normal_outputs.numpy(), qat_outputs.numpy(), atol=1e-6,
        )
