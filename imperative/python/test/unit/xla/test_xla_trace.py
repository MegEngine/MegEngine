import platform

import numpy as np
import pytest

import megengine
from megengine import is_cuda_available, tensor
from megengine.jit import partial_trace


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_partial_trace_inplace():
    def func(x, y):
        x += 1
        y += 1

    xla_func = partial_trace(func, backend="xla")
    xla_func(tensor(1), tensor(2))

    a1 = megengine.tensor(1)
    a2 = megengine.tensor(2)
    xla_func(a1, a2)
    np.testing.assert_allclose(a1, 2)
    np.testing.assert_allclose(a2, 3)


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_xla_trace_shape_change():
    def func(x, y):
        return x + y

    xla_func = partial_trace(func, backend="xla")
    a = np.random.randn(1, 3, 3, 3)
    b = np.random.randn(1, 3, 3, 3)
    rst0 = xla_func(tensor(a), tensor(b))
    rst1 = xla_func(tensor(1.0), tensor(2.0))  # fallback to python function
    rst2 = xla_func(tensor(a), tensor(b))  # exec in xla

    assert not rst1._is_external_value()
    assert rst2._is_external_value()
