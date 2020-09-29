import numpy as np

import megengine as mge
from megengine import autodiff as ad


def test_attach_in_with_block():
    a = mge.Parameter([1.0])
    g = ad.GradManager()
    with g:
        b = a * 3
        g.attach(b)
        c = b + 1
        g.backward(c)
    assert int(b.grad.numpy()) == 1
