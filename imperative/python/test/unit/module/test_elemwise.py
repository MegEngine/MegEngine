# -*- coding: utf-8 -*-
import numpy as np

import megengine.functional as F
from megengine import tensor
from megengine.module import Elemwise


def test_module_elemwise():
    def test_func(method, *inps):
        elemwise = Elemwise(method)
        outputs = elemwise(*inps)
        return outputs.numpy()

    x = np.random.rand(100).astype("float32")
    y = np.random.rand(100).astype("float32")
    x, y = tensor(x), tensor(y)
    np.testing.assert_almost_equal(
        test_func("h_swish", x), F.hswish(x).numpy(), decimal=6
    )
    np.testing.assert_almost_equal(
        test_func("add", x, y), F.add(x, y).numpy(), decimal=6
    )
