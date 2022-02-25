import numpy as np

import megengine as mge
from megengine.amp import GradScaler
from megengine.autodiff import GradManager
from megengine.jit import trace


def test_grad_scaler():
    def f():
        gm = GradManager()
        scaler = GradScaler()

        x = mge.tensor(1.0)
        for _ in range(3):
            with gm:
                y = x + 1
                gm.attach(y)
                loss = y + 1
                scaler.backward(gm, loss, unscale_grad=False)
            np.testing.assert_equal(y.grad.numpy(), scaler.scale_factor)
            scaler.unscale(gm.attached_tensors())
            np.testing.assert_equal(y.grad.numpy(), 1)
        # test handle None elements
        scaler.unscale(gm.attached_tensors())

    f()
    trace(f)()
