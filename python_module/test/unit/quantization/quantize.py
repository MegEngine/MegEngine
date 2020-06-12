# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from megengine import module as Float
from megengine.module import qat as QAT
from megengine.quantization.quantize import _get_quantable_module_names, quantize_qat


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
