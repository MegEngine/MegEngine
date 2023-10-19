import platform

import numpy as np
import pytest

import megengine
import megengine.functional as F
from megengine import is_cuda_available, tensor
from megengine.jit import partial_trace, xla_trace


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


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_xla_trace_random_seed_update():
    def tester(inp, drop_prob):
        @xla_trace(without_host=True)
        def func(x):
            a = F.dropout(x, drop_prob, True)
            b = F.dropout(x, drop_prob, True)
            return a, b

        _ = func(inp)
        a0, b0 = func(inp)
        seed_0 = func.random_seed
        a1, b1 = func(inp)
        seed_1 = func.random_seed

        assert not np.all(a0.numpy() == b0.numpy())
        assert not np.all(a1.numpy() == b1.numpy())
        assert not np.all(a0.numpy() == a1.numpy())
        assert not np.all(seed_0.numpy() == seed_1.numpy())
        return a0, b0, seed_0, seed_1

    inp = megengine.tensor(np.random.randn(4, 8), dtype="float32")
    megengine.random.seed(123)
    _1st_rsts = tester(inp, 0.5)
    _2nd_rsts = tester(inp, 0.5)
    megengine.random.seed(456)
    _3rd_rsts = tester(inp, 0.5)

    for _1st_rst, _2nd_rst, _3rd_rst in zip(_1st_rsts, _2nd_rsts, _3rd_rsts):
        assert np.all(_1st_rst.numpy() == _2nd_rst.numpy())
        assert not np.all(_1st_rst.numpy() == _3rd_rst.numpy())
