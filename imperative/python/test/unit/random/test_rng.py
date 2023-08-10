# -*- coding: utf-8 -*-
import numpy as np
import pytest

import megengine as mge
import megengine.functional as F
from megengine import Tensor, jit, random
from megengine.core._imperative_rt import CompNode
from megengine.core._imperative_rt.core2 import apply
from megengine.core._imperative_rt.ops import (
    delete_rng_handle,
    get_global_rng_seed,
    new_rng_handle,
)
from megengine.core.autodiff.grad import Grad
from megengine.core.ops.builtin import (
    BetaRNG,
    GammaRNG,
    GaussianRNG,
    PermutationRNG,
    PoissonRNG,
    UniformRNG,
)
from megengine.device import get_device_count
from megengine.jit import trace
from megengine.random import RNG
from megengine.random import seed as set_global_seed
from megengine.random import uniform


@pytest.mark.skipif(
    get_device_count("xpu") <= 2, reason="xpu counts need > 2",
)
def test_gaussian_op():
    set_global_seed(1024)
    shape = (
        8,
        9,
        11,
        12,
    )
    shape = Tensor(shape, dtype="int32")
    op = GaussianRNG(seed=get_global_rng_seed(), mean=1.0, std=3.0, dtype="float32")
    (output,) = apply(op, shape)
    assert np.fabs(output.numpy().mean() - 1.0) < 1e-1
    assert np.fabs(np.sqrt(output.numpy().var()) - 3.0) < 1e-1
    assert str(output.device) == str(CompNode("xpux"))
    assert output.dtype == np.float32

    cn = CompNode("xpu2")
    seed = 233333
    h = new_rng_handle(cn, seed)
    op = GaussianRNG(seed=seed, mean=3.0, std=1.0, dtype="float32", handle=h)
    (output,) = apply(op, shape)
    delete_rng_handle(h)
    assert np.fabs(output.numpy().mean() - 3.0) < 1e-1
    assert np.fabs(np.sqrt(output.numpy().var()) - 1.0) < 1e-1
    assert str(output.device) == str(cn)
    assert output.dtype == np.float32


@pytest.mark.skipif(
    get_device_count("xpu") <= 2, reason="xpu counts need > 2",
)
def test_uniform_op():
    set_global_seed(1024)
    shape = (
        8,
        9,
        11,
        12,
    )
    shape = Tensor(shape, dtype="int32")
    op = UniformRNG(seed=get_global_rng_seed(), dtype="float32")
    (output,) = apply(op, shape)
    assert np.fabs(output.numpy().mean() - 0.5) < 1e-1
    assert str(output.device) == str(CompNode("xpux"))
    assert output.dtype == np.float32

    cn = CompNode("xpu2")
    seed = 233333
    h = new_rng_handle(cn, seed)
    op = UniformRNG(seed=seed, dtype="float32", handle=h)
    (output,) = apply(op, shape)
    delete_rng_handle(h)
    assert np.fabs(output.numpy().mean() - 0.5) < 1e-1
    assert str(output.device) == str(cn)
    assert output.dtype == np.float32


@pytest.mark.skipif(
    get_device_count("xpu") <= 2, reason="xpu counts need > 2",
)
def test_gamma_op():
    set_global_seed(1024)
    _shape, _scale = 2, 0.8
    _expected_mean, _expected_std = _shape * _scale, np.sqrt(_shape) * _scale

    shape = F.full([8, 9, 11, 12], value=_shape, dtype="float32")
    scale = F.full([8, 9, 11, 12], value=_scale, dtype="float32")
    op = GammaRNG(seed=get_global_rng_seed(), handle=0)
    (output,) = apply(op, shape, scale)
    assert np.fabs(output.numpy().mean() - _expected_mean) < 1e-1
    assert np.fabs(np.sqrt(output.numpy().var()) - _expected_std) < 1e-1
    assert str(output.device) == str(CompNode("xpux"))

    cn = CompNode("xpu2")
    seed = 233333
    h = new_rng_handle(cn, seed)
    shape = F.full([8, 9, 11, 12], value=_shape, dtype="float32", device="xpu2")
    scale = F.full([8, 9, 11, 12], value=_scale, dtype="float32", device="xpu2")
    op = GammaRNG(seed=seed, handle=h)
    (output,) = apply(op, shape, scale)
    delete_rng_handle(h)
    assert np.fabs(output.numpy().mean() - _expected_mean) < 1e-1
    assert np.fabs(np.sqrt(output.numpy().var()) - _expected_std) < 1e-1
    assert str(output.device) == str(cn)


@pytest.mark.skipif(
    get_device_count("xpu") <= 2, reason="xpu counts need > 2",
)
def test_beta_op():
    set_global_seed(1024)
    _alpha, _beta = 2, 0.8
    _expected_mean = _alpha / (_alpha + _beta)
    _expected_std = np.sqrt(
        _alpha * _beta / ((_alpha + _beta) ** 2 * (_alpha + _beta + 1))
    )

    alpha = F.full([8, 9, 11, 12], value=_alpha, dtype="float32")
    beta = F.full([8, 9, 11, 12], value=_beta, dtype="float32")
    op = BetaRNG(seed=get_global_rng_seed())
    (output,) = apply(op, alpha, beta)
    assert np.fabs(output.numpy().mean() - _expected_mean) < 1e-1
    assert np.fabs(np.sqrt(output.numpy().var()) - _expected_std) < 1e-1
    assert str(output.device) == str(CompNode("xpux"))

    cn = CompNode("xpu2")
    seed = 233333
    h = new_rng_handle(cn, seed)
    alpha = F.full([8, 9, 11, 12], value=_alpha, dtype="float32", device=cn)
    beta = F.full([8, 9, 11, 12], value=_beta, dtype="float32", device=cn)
    op = BetaRNG(seed=seed, handle=h)
    (output,) = apply(op, alpha, beta)
    delete_rng_handle(h)
    assert np.fabs(output.numpy().mean() - _expected_mean) < 1e-1
    assert np.fabs(np.sqrt(output.numpy().var()) - _expected_std) < 1e-1
    assert str(output.device) == str(cn)


@pytest.mark.skipif(
    get_device_count("xpu") <= 2, reason="xpu counts need > 2",
)
def test_poisson_op():
    set_global_seed(1024)
    lam = F.full([8, 9, 11, 12], value=2, dtype="float32")
    op = PoissonRNG(seed=get_global_rng_seed())
    (output,) = apply(op, lam)
    assert np.fabs(output.numpy().mean() - 2.0) < 1e-1
    assert np.fabs(np.sqrt(output.numpy().var()) - np.sqrt(2.0)) < 1e-1
    assert str(output.device) == str(CompNode("xpux"))

    cn = CompNode("xpu2")
    seed = 233333
    h = new_rng_handle(cn, seed)
    lam = F.full([8, 9, 11, 12], value=2, dtype="float32", device=cn)
    op = PoissonRNG(seed=seed, handle=h)
    (output,) = apply(op, lam)
    delete_rng_handle(h)
    assert np.fabs(output.numpy().mean() - 2.0) < 1e-1
    assert np.fabs(np.sqrt(output.numpy().var()) - np.sqrt(2.0)) < 1e-1
    assert str(output.device) == str(cn)


@pytest.mark.skipif(
    get_device_count("xpu") <= 2, reason="xpu counts need > 2",
)
def test_permutation_op():
    set_global_seed(1024)
    n = 1000

    def test_permutation_op_dtype(dtype):
        def sum_result(res, fun):
            return sum([1 if i == v else 0 for i, v in enumerate(fun(res.numpy()))])

        shape = Tensor((n,), dtype="int32")
        op = PermutationRNG(seed=get_global_rng_seed(), dtype=dtype)
        (output,) = apply(op, shape)
        assert sum_result(output, lambda x: x) < 500
        assert sum_result(output, np.sort) == n
        assert str(output.device) == str(CompNode("xpux"))
        assert output.dtype == dtype

        cn = CompNode("xpu2")
        seed = 233333
        h = new_rng_handle(cn, seed)
        op = PermutationRNG(seed=seed, handle=h, dtype=dtype)
        (output,) = apply(op, shape)
        delete_rng_handle(h)
        assert sum_result(output, lambda x: x) < 500
        assert sum_result(output, np.sort) == n
        assert str(output.device) == str(cn)
        assert output.dtype == dtype

    test_permutation_op_dtype(np.float32)
    test_permutation_op_dtype(np.int32)
    test_permutation_op_dtype(np.int16)


@pytest.mark.skipif(
    get_device_count("xpu") <= 1, reason="xpu counts need > 1",
)
def test_UniformRNG():
    m1 = RNG(seed=111, device="xpu0")
    m2 = RNG(seed=111, device="xpu1")
    m3 = RNG(seed=222, device="xpu0")
    out1 = m1.uniform(size=(100,))
    out1_ = m1.uniform(size=(100,))
    out2 = m2.uniform(size=(100,))
    out3 = m3.uniform(size=(100,))

    np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-6)
    assert out1.device == "xpu0" and out2.device == "xpu1"
    assert not (out1.numpy() == out3.numpy()).all()
    assert not (out1.numpy() == out1_.numpy()).all()

    low = -234
    high = 123
    out = m1.uniform(low=low, high=high, size=(20, 30, 40))
    out_shp = out.shape
    if isinstance(out_shp, tuple):
        assert out_shp == (20, 30, 40)
    else:
        assert all(out.shape.numpy() == np.array([20, 30, 40]))
    assert np.abs(out.mean().numpy() - ((low + high) / 2)) / (high - low) < 0.1


@pytest.mark.skipif(
    get_device_count("xpu") <= 1, reason="xpu counts need > 1",
)
def test_NormalRNG():
    m1 = RNG(seed=111, device="xpu0")
    m2 = RNG(seed=111, device="xpu1")
    m3 = RNG(seed=222, device="xpu0")
    out1 = m1.normal(size=(100,))
    out1_ = m1.uniform(size=(100,))
    out2 = m2.normal(size=(100,))
    out3 = m3.normal(size=(100,))

    np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-6)
    assert out1.device == "xpu0" and out2.device == "xpu1"
    assert not (out1.numpy() == out3.numpy()).all()
    assert not (out1.numpy() == out1_.numpy()).all()

    mean = -1
    std = 2
    out = m1.normal(mean=mean, std=std, size=(20, 30, 40))
    out_shp = out.shape
    if isinstance(out_shp, tuple):
        assert out_shp == (20, 30, 40)
    else:
        assert all(out.shape.numpy() == np.array([20, 30, 40]))
    assert np.abs(out.mean().numpy() - mean) / std < 0.1
    assert np.abs(np.std(out.numpy()) - std) < 0.1


@pytest.mark.skipif(
    get_device_count("xpu") <= 1, reason="xpu counts need > 1",
)
def test_GammaRNG():
    m1 = RNG(seed=111, device="xpu0")
    m2 = RNG(seed=111, device="xpu1")
    m3 = RNG(seed=222, device="xpu0")
    out1 = m1.gamma(2, size=(100,))
    out1_ = m1.uniform(size=(100,))
    out2 = m2.gamma(2, size=(100,))
    out3 = m3.gamma(2, size=(100,))

    np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-6)
    assert out1.device == "xpu0" and out2.device == "xpu1"
    assert not (out1.numpy() == out3.numpy()).all()
    assert not (out1.numpy() == out1_.numpy()).all()

    shape = Tensor([[2, 3, 4], [9, 10, 11]], dtype=np.float32, device="xpu0")
    scale = Tensor([0.5, 1, 1.5], dtype=np.float32, device="xpu0")
    expected_mean = (shape * scale).numpy()
    expected_std = (F.sqrt(shape) * scale).numpy()
    out = m1.gamma(shape=shape, scale=scale, size=(20, 30, 40))
    out_shp = out.shape
    if isinstance(out_shp, tuple):
        assert out_shp == (20, 30, 40, 2, 3)
    else:
        assert all(out.shape.numpy() == np.array([20, 30, 40, 2, 3]))
    assert (
        np.abs(out.mean(axis=(0, 1)).numpy() - expected_mean) / expected_std
    ).mean() < 0.1
    assert (np.abs(np.std(out.numpy(), axis=(0, 1)) - expected_std)).mean() < 0.1


@pytest.mark.skipif(
    get_device_count("xpu") <= 1, reason="xpu counts need > 1",
)
def test_BetaRNG():
    m1 = RNG(seed=111, device="xpu0")
    m2 = RNG(seed=111, device="xpu1")
    m3 = RNG(seed=222, device="xpu0")
    out1 = m1.beta(2, 1, size=(100,))
    out1_ = m1.uniform(size=(100,))
    out2 = m2.beta(2, 1, size=(100,))
    out3 = m3.beta(2, 1, size=(100,))

    np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-6)
    assert out1.device == "xpu0" and out2.device == "xpu1"
    assert not (out1.numpy() == out3.numpy()).all()
    assert not (out1.numpy() == out1_.numpy()).all()

    alpha = Tensor([[2, 3, 4], [9, 10, 11]], dtype=np.float32, device="xpu0")
    beta = Tensor([0.5, 1, 1.5], dtype=np.float32, device="xpu0")
    expected_mean = (alpha / (alpha + beta)).numpy()
    expected_std = (
        F.sqrt(alpha * beta / (F.pow(alpha + beta, 2) * (alpha + beta + 1)))
    ).numpy()
    out = m1.beta(alpha=alpha, beta=beta, size=(20, 30))
    out_shp = out.shape
    if isinstance(out_shp, tuple):
        assert out_shp == (20, 30, 2, 3)
    else:
        assert all(out.shape.numpy() == np.array([20, 30, 2, 3]))
    assert (
        np.abs(out.mean(axis=(0, 1)).numpy() - expected_mean) / expected_std
    ).mean() < 0.1
    assert (np.abs(np.std(out.numpy(), axis=(0, 1)) - expected_std)).mean() < 0.1


@pytest.mark.skipif(
    get_device_count("xpu") <= 1, reason="xpu counts need > 1",
)
def test_PoissonRNG():
    m1 = RNG(seed=111, device="xpu0")
    m2 = RNG(seed=111, device="xpu1")
    m3 = RNG(seed=222, device="xpu0")
    lam = Tensor([[2, 3, 4], [9, 10, 11]], dtype=np.float32)
    out1 = m1.poisson(lam.to("xpu0"), size=(100,))
    out2 = m2.poisson(lam.to("xpu1"), size=(100,))
    out3 = m3.poisson(lam.to("xpu0"), size=(100,))

    np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-6)
    assert out1.device == "xpu0" and out2.device == "xpu1"
    assert not (out1.numpy() == out3.numpy()).all()

    out = m1.poisson(lam.to("xpu0"), size=(20, 30))
    out_shp = out.shape
    expected_shape = (20, 30) + lam._tuple_shape
    if isinstance(out_shp, tuple):
        assert out_shp == expected_shape
    else:
        assert all(out.shape.numpy() == np.array(expected_shape))
    lam = lam.numpy()

    assert (np.abs(out.mean(axis=(0, 1)).numpy() - lam) / np.sqrt(lam)).mean() < 0.1
    assert np.abs(np.std(out.numpy(), axis=(0, 1)) - np.sqrt(lam)).mean() < 0.1


@pytest.mark.skipif(
    get_device_count("xpu") <= 1, reason="xpu counts need > 1",
)
@pytest.mark.parametrize("symbolic", [True, False])
def test_PermutationRNG(symbolic):
    m1 = RNG(seed=111, device="xpu0")
    m2 = RNG(seed=111, device="xpu1")
    m3 = RNG(seed=222, device="xpu0")
    out1 = m1.permutation(1000)
    out1_ = m1.uniform(size=(1000,))
    out2 = m2.permutation(1000)
    out3 = m3.permutation(1000)

    np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-6)
    assert out1.device == "xpu0" and out2.device == "xpu1"
    assert not (out1.numpy() == out3.numpy()).all()
    assert not (out1.numpy() == out1_.numpy()).all()

    out = m1.permutation(1000)
    out_shp = out.shape
    if isinstance(out_shp, tuple):
        assert out_shp == (1000,)
    else:
        assert all(out.shape.numpy() == np.array([1000]))

    def sum_result(res, fun):
        return sum([1 if i == v else 0 for i, v in enumerate(fun(res.numpy()))])

    assert sum_result(out, lambda x: x) < 500
    assert sum_result(out, np.sort) == 1000

    def func():
        out = m1.permutation(Tensor(7))
        out_shp = out.shape
        if isinstance(out_shp, tuple):
            assert out_shp == (1,)
        else:
            assert all(out.shape.numpy() == np.array([1]))
        n, m = 6, 3
        out = m1.permutation(Tensor(np.arange(n * m), dtype="float32").reshape(n, m))
        out_shp = out.shape
        if isinstance(out_shp, tuple):
            assert out_shp == (n, m)
        else:
            assert all(out.shape.numpy() == np.array([n, m]))

    func = trace(symbolic=symbolic)(func)
    func()


@pytest.mark.skipif(
    get_device_count("xpu") <= 1, reason="xpu counts need > 1",
)
def test_ShuffleRNG():
    g = []

    def cb(grad):
        g.append(grad)

    n, m = 6, 3
    arr = np.arange(n * m)
    out0 = Tensor(arr, dtype="float32")
    with Grad() as grad:
        grad.wrt(out0, callback=cb)
        random.shuffle(out0)
        grad(out0, F.ones_like(out0))
    m1 = RNG(seed=111, device="xpu0")
    m2 = RNG(seed=111, device="xpu1")
    m3 = RNG(seed=222, device="xpu0")
    out1 = Tensor(arr, dtype="float32", device="xpu0")
    out2 = Tensor(arr, dtype="float32", device="xpu1")
    out3 = Tensor(arr, dtype="float32", device="xpu0")
    m1.shuffle(out1)
    m2.shuffle(out2)
    m3.shuffle(out3)

    np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-6)
    assert out1.device == "xpu0" and out2.device == "xpu1"
    assert not (out1.numpy() == out3.numpy()).all()

    out = Tensor(arr, dtype="float32").reshape(n, m)
    m1.shuffle(out)

    out_shp = out.shape
    if isinstance(out_shp, tuple):
        assert out_shp == (n, m)
    else:
        assert all(out.shape.numpy() == np.array([n, m]))

@pytest.mark.skipif(
    get_device_count("xpu") <= 1,
    reason="xpu counts need > 1",
)
def test_ExponentialRNG():
    m1 = RNG(seed=111, device="xpu0")
    m2 = RNG(seed=111, device="xpu1")
    m3 = RNG(seed=222, device="xpu0")
    rate = Tensor([[2, 3, 4], [9, 10, 11]], dtype=np.float32)
    out1 = m1.exponential(rate.to("xpu0"), size=(100,))
    out2 = m2.exponential(rate.to("xpu1"), size=(100,))
    out3 = m3.exponential(rate.to("xpu0"), size=(100,))

    np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-6)
    assert out1.device == "xpu0" and out2.device == "xpu1"
    assert not (out1.numpy() == out3.numpy()).all()

    out = m1.exponential(rate.to("xpu0"), size=(20, 30))
    out_shp = out.shape
    expected_shape = (20, 30) + rate._tuple_shape
    if isinstance(out_shp, tuple):
        assert out_shp == expected_shape
    else:
        assert all(out.shape.numpy() == np.array(expected_shape))
    rate = rate.numpy()

    expected_mean = 1.0 / rate
    expected_std = np.sqrt(1.0 / (rate * rate))
    assert (
        np.abs(out.mean(axis=(0, 1)).numpy() - expected_mean) / expected_std
    ).mean() < 0.1
    assert np.abs(np.std(out.numpy(), axis=(0, 1)) - expected_std).mean() < 0.1

def test_seed():
    set_global_seed(10)
    out1 = uniform(size=[10, 10])
    out2 = uniform(size=[10, 10])
    assert not (out1.numpy() == out2.numpy()).all()

    set_global_seed(10)
    out3 = uniform(size=[10, 10])
    np.testing.assert_allclose(out1.numpy(), out3.numpy(), atol=1e-6)

    set_global_seed(11)
    out4 = uniform(size=[10, 10])
    assert not (out1.numpy() == out4.numpy()).all()


@pytest.mark.parametrize("is_symbolic", [None, False, True])
def test_rng_empty_tensor(is_symbolic):
    set_global_seed(1024)
    shapes = [
        (0,),
        (0, 0, 0),
        (10, 0, 10),
    ]

    def fn(shape):
        o1 = random.uniform(0, 1, shape)
        o2 = random.normal(0, 1, shape)
        o3 = random.gamma(2, 1, shape)
        o4 = random.beta(2, 1, shape)
        o5 = random.poisson(2, shape)
        return o1, o2, o3, o4, o5

    for shape in shapes:
        if is_symbolic is not None:
            fn_ = jit.trace(symbolic=is_symbolic)(fn)
        else:
            fn_ = fn
        for _ in range(3):
            outs = fn_(shape)
            for out in outs:
                np.testing.assert_equal(out.numpy().shape, shape)
            if is_symbolic is None:
                break

    def fn2(n):
        return random.permutation(n=n)

    if is_symbolic is not None:
        fn2 = jit.trace(symbolic=is_symbolic)(fn2)

    for _ in range(3):
        out = fn2(0)
        np.testing.assert_equal(out.numpy().shape, (0,))
        if is_symbolic is None:
            break
