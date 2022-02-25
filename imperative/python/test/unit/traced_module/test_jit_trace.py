import io

import numpy as np

import megengine.functional as F
import megengine.module as M
import megengine.utils.comp_graph_tools as cgtools
from megengine.jit import trace
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


def test_jit_trace():
    module = MyModule()
    module.eval()
    x = F.ones((1, 8, 14, 14))
    expect = module(x)
    traced_module = trace_module(module, x)
    func = trace(traced_module, capture_as_const=True)
    np.testing.assert_array_equal(func(x), expect)
    model = io.BytesIO()
    func.dump(model)
    model.seek(0)
    infer_cg = cgtools.GraphInference(model)
    np.testing.assert_allclose(
        list(infer_cg.run(x.numpy()).values())[0], expect, atol=1e-6
    )
