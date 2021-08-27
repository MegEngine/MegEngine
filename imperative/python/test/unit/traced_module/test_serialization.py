# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import pickle

import numpy as np

import megengine.functional as F
import megengine.module as M
from megengine import Tensor
from megengine.module import Module
from megengine.traced_module import trace_module


class MyBlock(Module):
    def __init__(self, in_channels, channels):
        super(MyBlock, self).__init__()
        self.conv1 = M.Conv2d(in_channels, channels, 3, 1, padding=1, bias=False)
        self.bn1 = M.BatchNorm2d(channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x) + 1
        return x


class MyModule(Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.block0 = MyBlock(8, 4)
        self.block1 = MyBlock(4, 2)

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        return x


def test_dump_and_load():
    module = MyModule()
    x = Tensor(np.ones((1, 8, 14, 14)))
    expect = module(x)
    traced_module = trace_module(module, x)
    np.testing.assert_array_equal(expect, traced_module(x))
    obj = pickle.dumps(traced_module)
    pickle.loads(obj)
    np.testing.assert_array_equal(expect, traced_module(x))
