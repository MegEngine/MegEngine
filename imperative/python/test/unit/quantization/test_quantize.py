# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np
import pytest

from megengine import functional
from megengine import module as Float
from megengine import tensor
from megengine.module import qat as QAT
from megengine.module import quantized as Q
from megengine.quantization import (
    min_max_fakequant_qconfig,
    passive_qconfig,
    tqt_qconfig,
)
from megengine.quantization.fake_quant import TQT, FakeQuantize
from megengine.quantization.observer import MinMaxObserver, PassiveObserver
from megengine.quantization.quantize import (
    _get_quantable_module_names,
    apply_easy_quant,
    disable_fake_quant,
    disable_observer,
    enable_fake_quant,
    enable_observer,
    propagate_qconfig,
    quantize,
    quantize_qat,
    reset_qconfig,
)


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


class QATNet(Float.Module):
    def __init__(self):
        super().__init__()
        self.quant = QAT.QuantStub()
        self.linear = QAT.Linear(3, 3)
        self.dequant = QAT.DequantStub()
        self.linear.bias.set_value(np.random.rand(3))

    def forward(self, x):
        x = self.quant(x)
        x = self.linear(x)
        x = self.dequant(x)
        return x


def test_propagate_qconfig():
    net = QATNet()
    propagate_qconfig(net, min_max_fakequant_qconfig)
    assert all(
        [
            net.quant.weight_observer is None,
            net.quant.weight_fake_quant is None,
            isinstance(net.quant.act_observer, MinMaxObserver),
            isinstance(net.quant.act_fake_quant, FakeQuantize),
            isinstance(net.linear.weight_observer, MinMaxObserver),
            isinstance(net.linear.weight_fake_quant, FakeQuantize),
            isinstance(net.linear.act_observer, MinMaxObserver),
            isinstance(net.linear.act_fake_quant, FakeQuantize),
            net.dequant.weight_observer is None,
            net.dequant.weight_fake_quant is None,
            net.dequant.act_observer is None,
            net.dequant.act_observer is None,
        ]
    )


def init_qat_net():
    net = QATNet()
    propagate_qconfig(net, min_max_fakequant_qconfig)
    min_val = np.random.randint(-127, 0, size=(2,))
    max_val = np.random.randint(1, 127, size=(2,))
    net.linear.weight_observer.min_val.set_value(min_val[0])
    net.linear.weight_observer.max_val.set_value(max_val[0])
    net.linear.act_observer.min_val.set_value(min_val[1])
    net.linear.act_observer.max_val.set_value(max_val[1])
    return net


def test_reset_qconfig():
    qat_net = init_qat_net()
    new_qat_net = reset_qconfig(qat_net, passive_qconfig)
    assert (
        new_qat_net.linear.get_weight_qparams() == qat_net.linear.get_weight_qparams()
    )
    assert (
        new_qat_net.linear.get_activation_qparams()
        == qat_net.linear.get_activation_qparams()
    )


def test_enable_and_disable_observer():
    net = init_qat_net()
    enable_observer(net)
    assert net.quant.act_observer.enabled == True
    assert net.linear.weight_observer.enabled == True
    assert net.linear.act_observer.enabled == True
    disable_observer(net)
    assert net.quant.act_observer.enabled == False
    assert net.linear.weight_observer.enabled == False
    assert net.linear.act_observer.enabled == False


def test_enable_and_disable_fake_quant():
    net = init_qat_net()
    disable_fake_quant(net)
    assert net.quant.act_fake_quant.enabled == False
    assert net.linear.weight_fake_quant.enabled == False
    assert net.linear.act_fake_quant.enabled == False
    enable_fake_quant(net)
    assert net.quant.act_fake_quant.enabled == True
    assert net.linear.weight_fake_quant.enabled == True
    assert net.linear.act_fake_quant.enabled == True


def init_observer(module, data):
    enable_observer(module)
    disable_fake_quant(module)
    module(data)
    disable_observer(module)
    enable_fake_quant(module)


def test_enable_and_disable_all():
    x = tensor(np.random.randint(1, 10, size=(3, 3)).astype(np.float32))
    net = Net()
    y1 = net(x).numpy()
    net = quantize_qat(net, min_max_fakequant_qconfig)

    init_observer(net, x)

    y2 = net(x).numpy()
    disable_fake_quant(net)
    y3 = net(x).numpy()
    enable_fake_quant(net)
    y4 = net(x).numpy()
    np.testing.assert_allclose(y1, y3)
    np.testing.assert_allclose(y2, y4)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(y2, y3)


def test_quantize_qat():
    net = Net()
    qat_net = quantize_qat(net, inplace=False, qconfig=min_max_fakequant_qconfig)
    assert isinstance(qat_net.quant, QAT.QuantStub)
    assert isinstance(qat_net.linear, QAT.Linear)
    assert isinstance(qat_net.dequant, QAT.DequantStub)


def test_quantize():
    qat_net = init_qat_net()
    q_net = quantize(qat_net, inplace=False)
    assert isinstance(q_net.quant, Q.QuantStub)
    assert isinstance(q_net.linear, Q.Linear)
    assert isinstance(q_net.dequant, Q.DequantStub)


def test_apply_easy_quant():
    qat_net = init_qat_net()
    data = tensor(np.random.rand(2, 3, 3, 3), dtype=np.float32)
    eq_net = reset_qconfig(qat_net, passive_qconfig, inplace=False)
    apply_easy_quant(eq_net, data, 0.9, 1.1, 10)
    assert isinstance(eq_net.quant.act_observer, PassiveObserver)
    assert isinstance(eq_net.linear.weight_observer, PassiveObserver)
    assert isinstance(eq_net.linear.act_observer, PassiveObserver)
    assert eq_net.dequant.act_observer is None


def test_apply_tqt():
    qat_net = init_qat_net()
    tqt_net = reset_qconfig(qat_net, tqt_qconfig, inplace=False)
    assert isinstance(tqt_net.quant.act_fake_quant, TQT)
    assert isinstance(tqt_net.linear.weight_fake_quant, TQT)
    assert isinstance(tqt_net.linear.act_fake_quant, TQT)
    assert tqt_net.dequant.act_fake_quant is None


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
