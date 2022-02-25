# -*- coding: utf-8 -*-
import numpy as np
import pytest

import megengine as mge
from megengine.jit.tracing import set_symbolic_shape
from megengine.module import LeakyReLU, PReLU


def test_leaky_relu():
    data = np.array([-8, -12, 6, 10]).astype(np.float32)
    negative_slope = 0.1

    leaky_relu = LeakyReLU(negative_slope)
    output = leaky_relu(mge.tensor(data))

    np_output = np.maximum(0, data) + negative_slope * np.minimum(0, data)
    np.testing.assert_equal(output.numpy(), np_output)


@pytest.mark.parametrize("shape", [(1, 64, 15, 15), (64,)])
@pytest.mark.parametrize("use_symbolic", [False, True])
def test_prelu(shape, use_symbolic):
    old_flag = set_symbolic_shape(use_symbolic)
    data = np.random.random(size=shape)

    num_channel = 1 if len(shape) == 1 else shape[1]
    prelu = PReLU(num_parameters=num_channel, init=0.25)
    output = prelu(mge.Tensor(data))

    np_output = np.maximum(data, 0) + prelu.weight.numpy() * np.minimum(data, 0)
    set_symbolic_shape(old_flag)

    np.testing.assert_allclose(output.numpy(), np_output, atol=1e-5)
