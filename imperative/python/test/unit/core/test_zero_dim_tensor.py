import numpy as np

import megengine.functional as F
from megengine import Tensor
from megengine.core._trace_option import use_symbolic_shape


def test_zero_dim():
    a = Tensor(1)
    a_np = np.array(1, dtype=np.int32)
    np.testing.assert_equal(a, a_np)
    if use_symbolic_shape():
        np.testing.assert_equal(a.shape, np.array(a_np.shape))
    else:
        np.testing.assert_equal(a.shape, a_np.shape)


def test_sum():
    a = Tensor([1, 2])
    a = a.reshape((1, 2))
    assert a.sum().ndim == 0
    assert a.sum(axis=1).ndim == 1


def test_max():
    a = Tensor([1, 2])
    a = a.reshape((1, 2))
    assert a.max().ndim == 0
    assert a.max(axis=1).ndim == 1


def test_reshape():
    a = Tensor(1)
    a = a.reshape((1, 1))


def test_squeeze():
    a = Tensor(1)
    a = a.reshape((1, 1))
    assert F.squeeze(a).ndim == 0


def test_elemementwise():
    a = Tensor(1.0)
    assert F.exp(a).ndim == 0
    assert (a + a).ndim == 0
    assert (a + 1).ndim == 0


def test_astype():
    a = Tensor(1.0)
    assert a.astype("int32").ndim == 0


def test_tranpose():
    a = Tensor(1.0)
    assert a.transpose().ndim == 0
