# -*- coding: utf-8 -*-
from functools import partial

import numpy as np
import pytest
from utils import opr_test

import megengine.functional as F
from megengine import Tensor, jit, tensor
from megengine.core._imperative_rt.core2 import apply
from megengine.core.ops import builtin


def common_test_reduce(opr, ref_opr):
    data1_shape = (5, 6, 7)
    data2_shape = (2, 9, 12)
    data1 = np.random.random(data1_shape).astype(np.float32)
    data2 = np.random.random(data2_shape).astype(np.float32)
    cases = [
        {"input": data1},
        {"input": data2},
        {"input": np.array([[[1, 2, np.nan, 4], [8, 6, 5, 2], [2, 3, 4, 5]]])},
    ]

    if opr not in (F.argmin, F.argmax):
        # test default axis
        opr_test(cases, opr, ref_fn=ref_opr)
        # test all axises in range of input shape
        for axis in range(-3, 3):
            # test keepdims False
            opr_test(cases, opr, ref_fn=lambda x: ref_opr(x, axis=axis), axis=axis)
            # test keepdims True
            opr_test(
                cases,
                opr,
                ref_fn=lambda x: ref_opr(x, axis=axis, keepdims=True),
                axis=axis,
                keepdims=True,
            )
    else:
        # test defaut axis
        opr_test(cases, opr, ref_fn=lambda x: ref_opr(x).astype(np.int32))
        # test all axises in range of input shape
        for axis in range(0, 3):
            opr_test(
                cases,
                opr,
                ref_fn=lambda x: ref_opr(x, axis=axis).astype(np.int32),
                axis=axis,
            )
            # test negative axis
            axis = axis - len(data1_shape)
            opr_test(
                cases,
                opr,
                ref_fn=lambda x: ref_opr(x, axis=axis).astype(np.int32),
                axis=axis,
            )


def test_sum():
    common_test_reduce(opr=F.sum, ref_opr=np.sum)

    x = Tensor(np.arange(1, 7, dtype=np.int32).reshape(2, 3))
    y = F.sum(x, axis=-1)
    np.testing.assert_equal(y.numpy(), np.array([6, 15]).astype(np.int32))


def test_prod():
    common_test_reduce(opr=F.prod, ref_opr=np.prod)

    x = Tensor(np.arange(1, 7, dtype=np.int32).reshape(2, 3))
    y = F.prod(x, axis=-2)
    np.testing.assert_equal(y.numpy(), np.array([4, 10, 18]).astype(np.int32))


def test_mean():
    common_test_reduce(opr=F.mean, ref_opr=np.mean)

    x = Tensor(np.arange(1, 7, dtype=np.int32).reshape(2, 3))
    y = F.mean(x, axis=-2)
    np.testing.assert_equal(y.numpy(), np.array([2.5, 3.5, 4.5]).astype(np.float32))


def test_var():
    common_test_reduce(opr=F.var, ref_opr=np.var)

    x = Tensor(np.arange(1, 7, dtype=np.int32).reshape(2, 3))
    y = F.var(x, axis=-2)
    np.testing.assert_equal(y.numpy(), np.array([2.25, 2.25, 2.25]).astype(np.float32))


def test_std():
    common_test_reduce(opr=F.std, ref_opr=np.std)

    x = Tensor(np.arange(1, 7, dtype=np.int32).reshape(2, 3))
    y = F.std(x, axis=-2)
    np.testing.assert_equal(y.numpy(), np.array([1.5, 1.5, 1.5]).astype(np.float32))

    x = Tensor(np.arange(1, 7, dtype=np.int32).reshape(2, 3))
    y = F.std(x, axis=-2)
    np.testing.assert_equal(y.numpy(), np.array([1.5, 1.5, 1.5]).astype(np.float32))


def test_min():
    common_test_reduce(opr=F.min, ref_opr=np.min)

    x = Tensor(np.arange(1, 7, dtype=np.int32).reshape(2, 3))
    y = F.min(x, axis=-1)
    np.testing.assert_equal(y.numpy(), np.array([1, 4]).astype(np.int32))


def test_max():
    common_test_reduce(opr=F.max, ref_opr=np.max)

    x = Tensor(np.arange(1, 7, dtype=np.int32).reshape(2, 3))
    y = F.max(x, axis=-1)
    np.testing.assert_equal(y.numpy(), np.array([3, 6]).astype(np.int32))


def test_argmin():
    common_test_reduce(opr=F.argmin, ref_opr=np.argmin)

    x = Tensor(np.arange(1, 7, dtype=np.int32).reshape(2, 3))
    y = F.argmin(x, axis=-1)
    np.testing.assert_equal(y.numpy(), np.array([0, 0]).astype(np.int32))


def test_argmax():
    common_test_reduce(opr=F.argmax, ref_opr=np.argmax)
    x = Tensor(np.arange(1, 7, dtype=np.int32).reshape(2, 3))
    y = F.argmax(x, axis=-2)
    np.testing.assert_equal(y.numpy(), np.array([1, 1, 1]).astype(np.int32))


def test_norm():
    x = Tensor(np.arange(1, 7, dtype=np.int32).reshape(2, 3))
    y = F.norm(x, axis=-1)
    np.testing.assert_equal(
        y.numpy().round(decimals=3), np.array([3.742, 8.775]).astype(np.float32)
    )


def test_sqrt():
    d1_shape = (15,)
    d2_shape = (25,)
    d1 = np.random.random(d1_shape).astype(np.float32)
    d2 = np.random.random(d2_shape).astype(np.float32)

    cases = [{"input": d1}, {"input": d2}]
    opr_test(cases, F.sqrt, ref_fn=np.sqrt)


def test_sort():
    for _ in range(10):
        dim = np.random.randint(1, 5)
        shape = np.random.randint(1, 20, size=(dim,))
        data = np.random.randint(-100, 101, size=shape)
        by_dim = np.random.randint(-dim, dim)
        input = (data, False, by_dim)
        tns_ref = np.sort(data, by_dim)
        ind_ref = np.argsort(data, by_dim, kind="stable")
        opr_test([{"input": input, "output": [tns_ref, ind_ref]}], F.sort)


@pytest.mark.parametrize("is_symbolic", [None, False, True])
def test_sort_empty(is_symbolic):
    data_shapes = [
        (0,),
        (10, 0),
    ]

    def fn(x):
        return F.sort(x)

    for shape in data_shapes:
        if is_symbolic is not None:
            fn_ = jit.trace(symbolic=is_symbolic)(fn)
        else:
            fn_ = fn
        data = np.random.random(shape).astype(np.float32)
        for _ in range(3):
            outs = fn_(Tensor(data))
            ref_outs = (np.sort(data), np.argsort(data))
            assert len(ref_outs) == len(outs)
            for i in range(len(outs)):
                np.testing.assert_equal(outs[i].numpy(), ref_outs[i])
        if is_symbolic is None:
            break


def test_normalize():
    x = Tensor(np.arange(1, 7, dtype=np.int32).reshape(2, 3))
    y = F.normalize(x, axis=-1)
    np.testing.assert_equal(
        y.numpy().round(decimals=1),
        np.array([[0.3, 0.5, 0.8], [0.5, 0.6, 0.7]]).astype(np.float32),
    )

    cases = [
        {"input": np.random.random((2, 3, 12, 12)).astype(np.float32)} for i in range(2)
    ]

    def np_normalize(x, p=2, axis=None, eps=1e-12):
        if axis is None:
            norm = np.sum(x ** p) ** (1.0 / p)
        else:
            norm = np.sum(x ** p, axis=axis, keepdims=True) ** (1.0 / p)
        return x / np.clip(norm, a_min=eps, a_max=np.inf)

    # # Test L-2 norm along all dimensions
    # opr_test(cases, F.normalize, ref_fn=np_normalize)

    # # Test L-1 norm along all dimensions
    # opr_test(cases, partial(F.normalize, p=1), ref_fn=partial(np_normalize, p=1))

    # Test L-2 norm along the second dimension
    opr_test(cases, partial(F.normalize, axis=1), ref_fn=partial(np_normalize, axis=1))

    # Test some norm == 0
    cases[0]["input"][0, 0, 0, :] = 0
    cases[1]["input"][0, 0, 0, :] = 0
    opr_test(cases, partial(F.normalize, axis=3), ref_fn=partial(np_normalize, axis=3))


def test_sum_neg_axis():
    shape = (2, 3)
    data = np.random.random(shape).astype(np.float32)
    for axis in (-1, -2, (-2, 1), (-1, 0)):
        get = F.sum(Tensor(data), axis=axis)
        ref = np.sum(data, axis=axis)
        np.testing.assert_allclose(get.numpy(), ref, rtol=1e-6)
    with pytest.raises(AssertionError):
        F.sum(Tensor(data), axis=(-1, 1))


def test_builtin_reduce():
    shape = (2, 3, 3, 2)
    data = np.random.random(shape).astype(np.float32)
    for axis in (-1, -2, 0, 1):
        for keepdims in (True, False):
            op = builtin.Reduce(mode="sum", axis=axis, keepdim=keepdims)
            get = apply(op, tensor(data))[0]
            def_op = builtin.Reduce(mode="sum", axis=axis)
            def_get = apply(def_op, tensor(data))[0]
            ref = np.sum(data, axis=axis, keepdims=keepdims)
            np.testing.assert_allclose(get.numpy(), ref, rtol=1e-6)
            if keepdims == True:
                np.testing.assert_allclose(def_get.numpy(), ref, rtol=1e-6)


def test_non_finite():
    shape = (32, 3, 32, 32)
    data = []
    for i in range(2):
        data.append(np.random.random(shape).astype(np.float32))
    tensorList = [Tensor(x) for x in data]
    rst = F.math._check_non_finite(tensorList, 0.7)
    np.testing.assert_equal(rst.numpy(), [0])
    for i in range(len(tensorList)):
        np.testing.assert_allclose(tensorList[i].numpy() / 0.7, data[i], rtol=1e-6)

    data[1][0][0][0][0] = float("inf")
    rst = F.math._check_non_finite([Tensor(x) for x in data], 0.7)
    np.testing.assert_equal(rst.numpy(), [1])

    data[1][0][0][0][0] = float("nan")
    rst = F.math._check_non_finite([Tensor(x) for x in data], 0.7)
    np.testing.assert_equal(rst.numpy(), [1])


@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize("sorted", [True, False])
@pytest.mark.parametrize("inp1d", [True, False])
@pytest.mark.parametrize("kth_only", [True, False])
def test_topk(descending, sorted, inp1d, kth_only):
    k = 3
    if inp1d:
        data = np.random.permutation(7)
    else:
        data = np.random.permutation(5 * 7).reshape(5, 7)
    data = data.astype(np.int32)

    def np_sort(x):
        if descending:
            return np.sort(x)[..., ::-1]
        return np.sort(x)

    res = F.topk(
        Tensor(data), k, descending=descending, no_sort=(not sorted), kth_only=kth_only
    )

    values, indices = res
    values = values.numpy()
    indices = indices.numpy()
    if kth_only:
        np.testing.assert_equal(
            values, np.take_along_axis(data, indices[..., None], -1).squeeze(-1)
        )
        np.testing.assert_equal(values, np_sort(data)[..., k - 1])
    else:
        np.testing.assert_equal(values, np.take_along_axis(data, indices, -1))
        if not sorted:
            values = np_sort(values)
        np.testing.assert_equal(values, np_sort(data)[..., :k])


@pytest.mark.parametrize("is_trace", [True, False])
def test_reduce_on_empty_tensor(is_trace):
    dtypes = [np.float32, np.int32, np.bool]
    inputs = [
        (np.random.random((0,)), None),
        (np.random.random((3, 0, 2)), 1),
        (np.random.random((10, 10, 0, 10)), 0),
    ]

    def run_test(fn, ref_fn, input, dtype, axis=None, symbolic=False):
        if is_trace:
            fn = jit.trace(symbolic=symbolic)(fn)
        for i in range(3):
            out = fn(Tensor(input, dtype=dtype), axis=axis).numpy()
            out_ref = ref_fn(input.astype(dtype), axis=axis)
            np.testing.assert_equal(out, out_ref)

    for dtype in dtypes:
        for inp, axis in inputs:
            run_test(F.sum, np.sum, inp, dtype, axis, True)
            run_test(F.sum, np.sum, inp, dtype, axis, False)
            run_test(F.prod, np.prod, inp, dtype, axis, True)
            run_test(F.prod, np.prod, inp, dtype, axis, False)
