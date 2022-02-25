import types

import numpy as np
import pytest

import megengine as mge
import megengine.functional as F
import megengine.module as M
import megengine.traced_module as tm


class myconv(M.Conv2d):
    pass


class mybn(M.BatchNorm2d):
    pass


class MyBlock(M.Module):
    def __init__(self, conv_cls, bn_cls):
        super().__init__()
        self.conv = conv_cls(3, 3, 1, 1, 0)
        self.bn = bn_cls(3)
        self.conv2 = conv_cls(3, 3, 1, 1, 0)
        self.bn2 = bn_cls(3)
        self.scale = mge.Tensor([3, 4])

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.bn(x1)
        x1 = F.relu(x1)
        x1 = x1 * self.scale[0]
        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = x2 * self.scale[1]
        y = x1 + x2
        y = y + 4
        y = self.scale[0] + y
        y = F.relu(y) * 3
        return y


class MyModule(M.Module):
    def __init__(self, conv_cls, bn_cls):
        super().__init__()
        self.block_0 = MyBlock(conv_cls, bn_cls)
        self.block_1 = MyBlock(conv_cls, bn_cls)

    def forward(self, x):
        x1 = self.block_0(x)
        x2 = self.block_1(x)
        y = x1 + x2
        y = F.reshape(y, (-1))
        y = y * 3
        return y


@pytest.mark.parametrize("conv_cls", [M.Conv2d, myconv])
@pytest.mark.parametrize("bn_cls", [M.BatchNorm2d, mybn])
def test_backward_fold_scale(conv_cls, bn_cls):
    module = MyModule(conv_cls, bn_cls)
    module.eval()
    inp = mge.Tensor(np.random.random((1, 3, 32, 32)))
    desired = module(inp)
    traced_net = tm.trace_module(module, inp)

    traced_net = traced_net.flatten()
    optimized_net = tm.optimize(traced_net, "BackwardFoldScale")

    actual = optimized_net(inp)
    np.testing.assert_allclose(desired=desired, actual=actual, atol=1e-4)
    # fuse all mul to conv
    mul_list = optimized_net.graph.get_method_by_type("__mul__").as_list()
    assert len(mul_list) == 0


@pytest.mark.parametrize("conv_cls", [M.Conv2d, myconv])
@pytest.mark.parametrize("bn_cls", [M.BatchNorm2d, mybn])
def test_fuse_bn(conv_cls, bn_cls):
    module = MyModule(conv_cls, bn_cls)
    module.eval()
    inp = mge.Tensor(np.random.random((1, 3, 32, 32)))
    desired = module(inp)
    traced_net = tm.trace_module(module, inp)

    traced_net = traced_net.flatten()
    optimized_net = tm.optimize(traced_net, "FuseConvBn")

    actual = optimized_net(inp)
    np.testing.assert_allclose(desired=desired, actual=actual, atol=1e-4)
    # fuse all mul to conv
    bn_list = optimized_net.graph.get_function_by_type(F.batch_norm).as_list()
    assert len(bn_list) == 0

    bn_list = optimized_net.graph.get_module_by_type(M.BatchNorm2d).as_list()
    assert len(bn_list) == 0
