# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import os

import numpy as np

import megengine as mge
import megengine.autodiff as ad
import megengine.module as M
import megengine.optimizer as optimizer
from megengine import Parameter, tensor
from megengine.module import Module


class Simple(Module):
    def __init__(self):
        super().__init__()
        self.a = Parameter([1.23], dtype=np.float32)

    def forward(self, x):
        x = x * self.a
        return x


class Net(Module):
    def __init__(self):
        super().__init__()
        self.fc = M.Linear(1, 1)

    def forward(self, images):
        x = self.fc(images)
        loss = x.mean() * 10000
        return loss


def test_load_state_dict_no_cache(monkeypatch):
    with monkeypatch.context() as mk:
        mk.setenv("MEGENGINE_INPLACE_UPDATE", "1")
        net = Net()

        optim = optimizer.SGD(net.parameters(), lr=0.1)
        gm = ad.GradManager().attach(net.parameters())
        state = {
            "fc.weight": np.array([[0]], dtype=np.float32),
            "fc.bias": np.array([0.0], dtype=np.float32),
        }
        net.load_state_dict(state)
        images = mge.tensor([[0]], dtype=np.float32)
        with gm:
            loss = net(images)
            gm.backward(loss)
            optim.step()
            optim.clear_grad()


def test_save_load():
    net = Simple()

    optim = optimizer.SGD(net.parameters(), lr=1.0, momentum=0.9)
    optim.clear_grad()
    gm = ad.GradManager().attach(net.parameters())

    data = tensor([2.34])

    with gm:
        loss = net(data)
        gm.backward(loss)

    optim.step()

    model_name = "simple.pkl"

    mge.save(
        {
            "name": "simple",
            "state_dict": net.state_dict(),
            "opt_state": optim.state_dict(),
        },
        model_name,
    )

    # Load param to cpu
    checkpoint = mge.load(model_name, map_location="cpu0")
    device_save = mge.get_default_device()
    mge.set_default_device("cpu0")
    net = Simple()
    net.load_state_dict(checkpoint["state_dict"])
    optim = optimizer.SGD(net.parameters(), lr=1.0, momentum=0.9)
    optim.load_state_dict(checkpoint["opt_state"])
    os.remove("simple.pkl")

    with gm:
        loss = net([1.23])
        gm.backward(loss)

    optim.step()
    # Restore device
    mge.set_default_device(device_save)
