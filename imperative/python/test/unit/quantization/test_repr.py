# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import megengine.module as M
from megengine.quantization import quantize, quantize_qat


def test_repr():
    class Net(M.Module):
        def __init__(self):
            super().__init__()
            self.conv_bn = M.ConvBnRelu2d(3, 3, 3)
            self.linear = M.Linear(3, 3)

        def forward(self, x):
            return x

    net = Net()
    ground_truth = (
        "Net(\n"
        "  (conv_bn): ConvBnRelu2d(\n"
        "    (conv): Conv2d(3, 3, kernel_size=(3, 3))\n"
        "    (bn): BatchNorm2d(3, eps=1e-05, momentum=0.9, affine=True, track_running_stats=True)\n"
        "  )\n"
        "  (linear): Linear(in_features=3, out_features=3, bias=True)\n"
        ")"
    )
    assert net.__repr__() == ground_truth
    quantize_qat(net)
    ground_truth = (
        "Net(\n"
        "  (conv_bn): QAT.ConvBnRelu2d(\n"
        "    (conv): Conv2d(3, 3, kernel_size=(3, 3))\n"
        "    (bn): BatchNorm2d(3, eps=1e-05, momentum=0.9, affine=True, track_running_stats=True)\n"
        "    (act_observer): ExponentialMovingAverageObserver()\n"
        "    (act_fake_quant): FakeQuantize()\n"
        "    (weight_observer): MinMaxObserver()\n"
        "    (weight_fake_quant): FakeQuantize()\n"
        "  )\n"
        "  (linear): QAT.Linear(\n"
        "    in_features=3, out_features=3, bias=True\n"
        "    (act_observer): ExponentialMovingAverageObserver()\n"
        "    (act_fake_quant): FakeQuantize()\n"
        "    (weight_observer): MinMaxObserver()\n"
        "    (weight_fake_quant): FakeQuantize()\n"
        "  )\n"
        ")"
    )
    assert net.__repr__() == ground_truth
    quantize(net)
    ground_truth = (
        "Net(\n"
        "  (conv_bn): Quantized.ConvBnRelu2d(3, 3, kernel_size=(3, 3))\n"
        "  (linear): Quantized.Linear()\n"
        ")"
    )
    assert net.__repr__() == ground_truth
