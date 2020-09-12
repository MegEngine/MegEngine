# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from functools import partial

import numpy as np

import megengine.functional as F
from megengine import tensor
from megengine.test import assertTensorClose


def _default_compare_fn(x, y):
    assertTensorClose(x.numpy(), y)


def opr_test(cases, func, compare_fn=_default_compare_fn, ref_fn=None, **kwargs):
    """
    func: the function to run opr.
    compare_fn: the function to compare the result and expected, use assertTensorClose if None.
    ref_fn: the function to generate expected data, should assign output if None.
    cases: the list which have dict element, the list length should be 2 for dynamic shape test.
           and the dict should have input,
           and should have output if ref_fn is None.
           should use list for multiple inputs and outputs for each case.
    kwargs: The additional kwargs for opr func.

    simple examples:

        dtype = np.float32
        cases = [{"input": [10, 20]}, {"input": [20, 30]}]
        opr_test(cases,
                 F.eye,
                 ref_fn=lambda n, m: np.eye(n, m).astype(dtype),
                 dtype=dtype)

    """

    def check_results(results, expected):
        if not isinstance(results, tuple):
            results = (results,)
        for r, e in zip(results, expected):
            compare_fn(r, e)

    def get_param(cases, idx):
        case = cases[idx]
        inp = case.get("input", None)
        outp = case.get("output", None)
        if inp is None:
            raise ValueError("the test case should have input")
        if not isinstance(inp, list):
            inp = (inp,)
        else:
            inp = tuple(inp)
        if ref_fn is not None and callable(ref_fn):
            outp = ref_fn(*inp)
        if outp is None:
            raise ValueError("the test case should have output or reference function")
        if not isinstance(outp, list):
            outp = (outp,)
        else:
            outp = tuple(outp)

        return inp, outp

    if len(cases) == 0:
        raise ValueError("should give one case at least")

    if not callable(func):
        raise ValueError("the input func should be callable")

    inp, outp = get_param(cases, 0)
    inp_tensor = [tensor(inpi) for inpi in inp]

    results = func(*inp_tensor, **kwargs)
    check_results(results, outp)


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
    output0 = [np.sort(data1), np.argsort(data1).astype(np.int32)]
    output1 = [np.sort(data2), np.argsort(data2).astype(np.int32)]

    cases = [
        {"input": data1, "output": output0},
        {"input": data2, "output": output1},
    ]
    opr_test(cases, F.sort)


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

    # Test L-2 norm along all dimensions
    opr_test(cases, F.normalize, ref_fn=np_normalize)

    # Test L-1 norm along all dimensions
    opr_test(cases, partial(F.normalize, p=1), ref_fn=partial(np_normalize, p=1))

    # Test L-2 norm along the second dimension
    opr_test(cases, partial(F.normalize, axis=1), ref_fn=partial(np_normalize, axis=1))

    # Test some norm == 0
    cases[0]["input"][0, 0, 0, :] = 0
    cases[1]["input"][0, 0, 0, :] = 0
    opr_test(cases, partial(F.normalize, axis=3), ref_fn=partial(np_normalize, axis=3))


def test_matmul():
    shape1 = 3
    shape2 = 3
    shape3 = (3, 5)
    shape4 = (5, 6)
    data1 = np.random.random(shape1).astype("float32")
    data2 = np.random.random(shape2).astype("float32")
    data3 = np.random.random(shape3).astype("float32")
    data4 = np.random.random(shape4).astype("float32")

    cases = [
        {"input": [data1, data2]},
        {"input": [data2, data3]},
        {"input": [data3, data4]},
    ]
    opr_test(cases, F.matmul, ref_fn=np.matmul)

    batch_size = 10
    shape1 = (batch_size, 2, 3)
    shape2 = (batch_size, 3, 4)
    shape3 = (batch_size, 10, 4, 5)
    data1 = np.random.random(shape1).astype("float32")
    data2 = np.random.random(shape2).astype("float32")
    data3 = np.random.random(shape3).astype("float32")

    cases = [{"input": [data1, data2]}, {"input": [data2, data3]}]
    for i in range(0, batch_size):

        def compare_fn(x, y):
            x.numpy()[i, ...] == y

        opr_test(
            cases,
            F.matmul,
            compare_fn=compare_fn,
            ref_fn=lambda x, y: np.matmul(x[i, ...], y[i, ...]),
        )


# def test_logsumexp():
#     x = np.arange(10).astype(np.float32)
#     expected = np.log(np.sum(np.exp(x)))
#     cases = [{"input": x, "output": expected}]
#     compare_fn = partial(assertTensorClose, allow_special_values=True)
#     # large value check
#     n = 100
#     x = np.full(n, 10000, dtype=np.float32)
#     expected = 10000 + np.log(n)
#     cases.append({"input": x, "output": expected.astype(np.float32)})
#     opr_test(cases, F.logsumexp, axis=0, compare_fn=compare_fn)

#     # special value check
#     x = np.array([np.inf], dtype=np.float32)
#     expected = x
#     cases = [{"input": x, "output": expected}]

#     x = np.array([-np.inf, 0.0], dtype=np.float32)
#     expected = np.zeros(1).astype(np.float32)
#     cases.append({"input": x, "output": expected})
#     opr_test(cases, F.logsumexp, axis=0, compare_fn=compare_fn)

#     x = np.array([np.nan], dtype=np.float32)
#     expected = x
#     cases = [{"input": x, "output": expected}]

#     x = np.array([-np.inf, 1], dtype=np.float32)
#     expected = np.array([1.0], dtype=np.float32)
#     cases.append({"input": x, "output": expected})

#     opr_test(cases, F.logsumexp, axis=0, compare_fn=compare_fn)

#     # keepdims check
#     x = np.array([[1e10, 1e-10], [-1e10, -np.inf]], dtype=np.float32)
#     expected = np.array([[1e10], [-1e10]], dtype=np.float32)
#     cases = [{"input": x, "output": expected}]
#     x = np.array([[1e10, -1e-10, 1e-10], [1e10, 1e-10, np.inf]], dtype=np.float32)
#     expected = np.array([[1e10], [np.inf]], dtype=np.float32)
#     cases.append({"input": x, "output": expected})
#     opr_test(cases, F.logsumexp, axis=1, keepdims=True, compare_fn=compare_fn)

#     # multiple axes check
#     x = np.array([[1e10, 1e-10], [-1e10, -np.inf]], dtype=np.float32)
#     expected = np.array([1e10], dtype=np.float32)
#     cases = [{"input": x, "output": expected}]
#     x = np.array([[1e10, -1e-10, 1e-10], [1e10, 1e-10, np.inf]], dtype=np.float32)
#     expected = np.array([np.inf], dtype=np.float32)
#     cases.append({"input": x, "output": expected})
#     opr_test(cases, F.logsumexp, axis=(0, 1), keepdims=False, compare_fn=compare_fn)
