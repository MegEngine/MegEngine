# -*- coding: utf-8 -*-
import platform

import numpy as np
import pytest

import megengine
import megengine.autodiff as ad
import megengine.distributed as dist
import megengine.optimizer as optimizer
from megengine import Parameter, tensor
from megengine.module import Module
from megengine.optimizer import SGD


class Simple(Module):
    def __init__(self, param_shape):
        super().__init__()
        self.params = [
            Parameter(np.ones(param_shape), dtype=np.float32) for i in range(10)
        ]

    def forward(self, x):
        for p in self.params:
            x = x * p
        return x


@pytest.mark.require_ngpu(2)
@pytest.mark.isolated_distributed
@pytest.mark.parametrize(
    "threshold", [0, 128, None], ids=["no_pack", "small_pack", "large_pack"]
)
@pytest.mark.parametrize("param_shape", [(16,), (128, 256), (2, 1024, 1024)])
def test_param_pack(param_shape, threshold, n_iters=100):
    data = np.ones(param_shape, dtype="float32")

    @dist.launcher(n_gpus=2)
    def worker():
        net = Simple(param_shape)
        opt = SGD(net.parameters(), lr=0.1)

        allreduce_cb = dist.make_allreduce_cb("MEAN", dist.WORLD)
        if threshold is not None:
            allreduce_cb._param_pack_thd = threshold
        gm = ad.GradManager().attach(net.parameters(), callbacks=[allreduce_cb])

        def run():
            opt.clear_grad()
            with gm:
                x = tensor(data)
                loss = net(x)
                loss = loss.sum()
                gm.backward(loss)

        for i in range(n_iters):
            run()

        for p in net.params:
            np.testing.assert_equal(p.grad.numpy(), np.ones_like(p.grad.numpy()))

    worker()
