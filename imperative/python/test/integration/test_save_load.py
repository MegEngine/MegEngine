import numpy as np

import megengine as mge
import megengine.autodiff as ad
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
    print("save to {}".format(model_name))

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
    print("load done")

    with gm:
        loss = net([1.23])
        gm.backward(loss)

    optim.step()
    # Restore device
    mge.set_default_device(device_save)
