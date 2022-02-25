# -*- coding: utf-8 -*-
import itertools

import numpy as np
import pytest

import megengine as mge
import megengine.module as M
from megengine import tensor


# NOTE: test in module for convenience. should really test in functional
@pytest.mark.parametrize(
    "name",
    ["Conv1d", "Conv2d", "Conv3d", "ConvTranspose2d", "ConvTranspose3d", "LocalConv2d"],
)
def test_conv_dtype_promotion(name):
    old = mge.config.deterministic_kernel
    mge.config.deterministic_kernel = True
    N, Ci, Co, K = 2, 16, 32, 3
    S = (7,) * int(name[-2])
    if "Local" in name:
        m = getattr(M, name)(Ci, Co, *S, K)
    else:
        m = getattr(M, name)(Ci, Co, K)
    x = tensor(np.random.random(size=(N, Ci) + S).astype("float16"))
    np.testing.assert_equal(m(x).numpy(), m(x.astype("float32")).numpy())
    mge.config.deterministic_kernel = old
