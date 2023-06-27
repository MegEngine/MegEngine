import numpy as np

import megengine as mge
import megengine.functional as F
import megengine.module as M
from megengine.autodiff import GradManager
from megengine.jit import trace
from megengine.optimizer import Adam


def test_xla_conv_module():
    m = M.Conv2d(3, 3, 3)

    @trace(without_host=True, use_xla=True)
    def step(m, inp):
        return m(inp)

    inp = mge.tensor(np.random.random((3, 3, 32, 32)))
    step(m, inp)

    xla_rst = step(m, inp)
    mge_rst = step.__wrapped__(m, inp)
    np.testing.assert_allclose(mge_rst, xla_rst)


def test_train():
    def run(use_trace):
        np.random.seed(1024)
        mge.random.seed(233)
        m = M.Conv2d(3, 3, 3, padding=1)
        inp = mge.tensor(np.random.random((3, 3, 32, 32)))
        gm = GradManager()
        opt = Adam(m.parameters(), lr=0.1)
        gm.attach(m.parameters())

        def train_step(model, opt, inp):
            with gm:
                out = model(inp) + 1
                loss = F.loss.square_loss(out, F.sin(inp))
                gm.backward(loss)
            opt.step().clear_grad()
            return loss

        if use_trace:
            train_step = trace(train_step, without_host=True)

        for i in range(100):
            loss = train_step(m, opt, inp)
        return m.weight, m.bias, opt.state_dict()["state"][0]["exp_avg"]

    w0, b0, s0 = run(False)
    w1, b1, s1 = run(True)
    np.testing.assert_allclose(w0, w1, rtol=1e-3)
    np.testing.assert_allclose(b0, b1, rtol=1e-3)
    np.testing.assert_allclose(s0, s1, rtol=1e-3)


if __name__ == "__main__":
    test_train()
    test_xla_conv_module()
