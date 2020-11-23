# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np
import pytest

from megengine import module as Float
from megengine import tensor
from megengine.module import qat as QAT
from megengine.quantization import min_max_fakequant_qconfig
from megengine.quantization.quantize import (
    _get_quantable_module_names,
    disable_fake_quant,
    quantize_qat,
)


def test_get_quantable_module_names():
    # need to make sure names from Quantized and QAT are the same
    def _get_qat_module_names():
        def is_qat(key: str):
            value = getattr(QAT, key)
            return (
                isinstance(value, type)
                and issubclass(value, QAT.QATModule)
                and value != QAT.QATModule
            )

        # source should have all quantable modules' names
        quantable_module_names = [key for key in dir(QAT) if is_qat(key)]
        return quantable_module_names

    qat_module_names = _get_qat_module_names()
    quantized_module_names = _get_quantable_module_names()
    assert set(qat_module_names) == set(quantized_module_names)

    for key in qat_module_names:
        value = getattr(Float, key)
        assert (
            isinstance(value, type)
            and issubclass(value, Float.Module)
            and value != Float.Module
        )


def test_disable_quantize():
    class Net(Float.Module):
        def __init__(self):
            super().__init__()
            self.conv = Float.ConvBnRelu2d(3, 3, 3)
            self.conv.disable_quantize()

        def forward(self, x):
            return self.conv(x)

    net = Net()
    qat_net = quantize_qat(net, inplace=False)
    assert isinstance(qat_net.conv, Float.ConvBnRelu2d)
    assert isinstance(qat_net.conv.conv, Float.Conv2d)


def test_convert_with_custom_mapping():
    class FloatExample(Float.Module):
        def forward(self, x):
            return x

    class QATExample(QAT.QATModule):
        def forward(self, x):
            return x

        @classmethod
        def from_float_module(cls, float_module):
            return cls()

    class Net(Float.Module):
        def __init__(self):
            super().__init__()
            self.example = FloatExample()

        def forward(self, x):
            return self.example(x)

    net = Net()
    qat_net = quantize_qat(net, inplace=False, mapping={FloatExample: QATExample})
    assert isinstance(qat_net.example, QATExample)


def test_disable_fake_quant():
    class Net(Float.Module):
        def __init__(self):
            super().__init__()
            self.quant = Float.QuantStub()
            self.linear = Float.Linear(3, 3)
            self.dequant = Float.DequantStub()
            self.linear.bias.set_value(np.random.rand(3))

        def forward(self, x):
            x = self.quant(x)
            x = self.linear(x)
            x = self.dequant(x)
            return x

    x = tensor(np.random.randint(1, 10, size=(3, 3)).astype(np.float32))
    net = Net()
    y1 = net(x).numpy()
    net = quantize_qat(net, min_max_fakequant_qconfig)
    y2 = net(x).numpy()
    disable_fake_quant(net)
    y3 = net(x).numpy()
    np.testing.assert_allclose(y1, y3)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(y2, y3)
