from itertools import product

import numpy as np

from megengine import tensor
from megengine.module import (
    Conv2d,
    ConvBn2d,
    ConvRelu2d,
    DequantStub,
    Module,
    QuantStub,
)
from megengine.quantization.quantize import disable_fake_quant, quantize_qat


def test_qat_convbn2d():
    in_channels = 32
    out_channels = 64
    kernel_size = 3
    for groups, bias in product([1, 4], [True, False]):
        module = ConvBn2d(
            in_channels, out_channels, kernel_size, groups=groups, bias=bias
        )
        module.train()
        qat_module = quantize_qat(module, inplace=False)
        disable_fake_quant(qat_module)
        inputs = tensor(np.random.randn(4, in_channels, 32, 32).astype(np.float32))
        normal_outputs = module(inputs)
        # import pdb
        # pdb.set_trace()
        qat_outputs = qat_module(inputs)
        np.testing.assert_allclose(
            normal_outputs.numpy(), qat_outputs.numpy(), atol=5e-6
        )
        np.testing.assert_allclose(
            module.bn.running_mean.numpy(),
            qat_module.bn.running_mean.numpy(),
            atol=5e-8,
        )
        np.testing.assert_allclose(
            module.bn.running_var.numpy(), qat_module.bn.running_var.numpy(), atol=5e-7,
        )
        module.eval()
        normal_outputs = module(inputs)
        qat_module.eval()
        qat_outputs = qat_module(inputs)
        np.testing.assert_allclose(
            normal_outputs.numpy(), qat_outputs.numpy(), atol=5e-6
        )


def test_qat_conv():

    in_channels = 32
    out_channels = 64
    kernel_size = 3

    class TestNet(Module):
        def __init__(self, groups, bias):
            super().__init__()
            self.quant = QuantStub()
            self.dequant = DequantStub()
            self.conv = Conv2d(
                in_channels, out_channels, kernel_size, groups=groups, bias=bias
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
