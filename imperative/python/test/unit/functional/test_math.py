# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from functools import partial

import numpy as np
import pytest
from utils import opr_test

import megengine.functional as F
from megengine import jit, tensor


def common_test_reduce(opr, ref_opr):
    data1_shape = (5, 6, 7)
    data2_shape = (2, 9, 12)
    data1 = np.random.random(data1_shape).astype(np.float32)
    data2 = np.random.random(data2_shape).astype(np.float32)
    cases = [{"input": data1}, {"input": data2}]

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


def test_prod():
    common_test_reduce(opr=F.prod, ref_opr=np.prod)


def test_mean():
    common_test_reduce(opr=F.mean, ref_opr=np.mean)


def test_var():
    common_test_reduce(opr=F.var, ref_opr=np.var)


def test_std():
    common_test_reduce(opr=F.std, ref_opr=np.std)


def test_min():
    common_test_reduce(opr=F.min, ref_opr=np.min)


def test_max():
    common_test_reduce(opr=F.max, ref_opr=np.max)


def test_argmin():
    common_test_reduce(opr=F.argmin, ref_opr=np.argmin)


def test_argmax():
    common_test_reduce(opr=F.argmax, ref_opr=np.argmax)


def test_sqrt():
    d1_shape = (15,)
    d2_shape = (25,)
    d1 = np.random.random(d1_shape).astype(np.float32)
    d2 = np.random.random(d2_shape).astype(np.float32)

    cases = [{"input": d1}, {"input": d2}]
    opr_test(cases, F.sqrt, ref_fn=np.sqrt)


def test_sort():
    data1_shape = (10, 3)
    data2_shape = (12, 2)
    data1 = np.random.random(data1_shape).astype(np.float32)
    data2 = np.random.random(data2_shape).astype(np.float32)
    output1 = [np.sort(data1), np.argsort(data1).astype(np.int32)]
    output2 = [np.sort(data2), np.argsort(data2).astype(np.int32)]

    cases = [
        {"input": data1, "output": output1},
        {"input": data2, "output": output2},
    ]
    opr_test(cases, F.sort)


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
            outs = fn_(tensor(data))
            ref_outs = (np.sort(data), np.argsort(data))
            assert len(ref_outs) == len(outs)
            for i in range(len(outs)):
                np.testing.assert_equal(outs[i].numpy(), ref_outs[i])
        if is_symbolic is None:
            break


def test_normalize():

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
        get = F.sum(tensor(data), axis=axis)
        ref = np.sum(data, axis=axis)
        np.testing.assert_allclose(get.numpy(), ref, rtol=1e-6)
    with pytest.raises(AssertionError):
        F.sum(tensor(data), axis=(-1, 1))


def test_has_inf():
    shape = (32, 3, 32, 32)
    data = np.random.random(shape).astype(np.float32)
    rst = F.math._has_inf(tensor(data))
    np.testing.assert_equal(rst.numpy(), [0])

    data[0][0][0][0] = float("inf")
    rst = F.math._has_inf(tensor(data))
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
        tensor(data), k, descending=descending, no_sort=(not sorted), kth_only=kth_only
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
            out = fn(tensor(input, dtype=dtype), axis=axis).numpy()
            out_ref = ref_fn(input.astype(dtype), axis=axis)
            np.testing.assert_equal(out, out_ref)

    for dtype in dtypes:
        for inp, axis in inputs:
            run_test(F.sum, np.sum, inp, dtype, axis, True)
            run_test(F.sum, np.sum, inp, dtype, axis, False)
            run_test(F.prod, np.prod, inp, dtype, axis, True)
            run_test(F.prod, np.prod, inp, dtype, axis, False)
