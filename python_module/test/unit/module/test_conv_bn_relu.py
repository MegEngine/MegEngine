import copy
from itertools import product

import numpy as np

from megengine import tensor
from megengine.module import ConvBn2d
from megengine.quantization.quantize import disable_fake_quant, quantize_qat
from megengine.test import assertTensorClose


def test_convbn2d():
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
        qat_outputs = qat_module(inputs)
        assertTensorClose(normal_outputs, qat_outputs, max_err=5e-6)
        assertTensorClose(
            module.bn.running_mean, qat_module.bn.running_mean, max_err=5e-8
        )
        assertTensorClose(
            module.bn.running_var, qat_module.bn.running_var, max_err=5e-7
        )
        module.eval()
        normal_outputs = module(inputs)
        qat_module.eval()
        qat_outputs = qat_module(inputs)
        assertTensorClose(normal_outputs, qat_outputs, max_err=5e-6)
