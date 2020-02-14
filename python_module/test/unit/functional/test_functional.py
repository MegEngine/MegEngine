# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np
from helpers import opr_test

import megengine.functional as F
from megengine import Buffer, jit, tensor
from megengine.test import assertTensorClose


def test_flatten():
    data0_shape = (2, 3, 4, 5)
    data1_shape = (4, 5, 6, 7)
    data0 = np.random.random(data0_shape).astype(np.float32)
    data1 = np.random.random(data1_shape).astype(np.float32)

    def compare_fn(x, y):
        assert x.numpy().shape == y

    output0 = (2 * 3 * 4 * 5,)
    output1 = (4 * 5 * 6 * 7,)
    cases = [{"input": data0, "output": output0}, {"input": data1, "output": output1}]
    opr_test(cases, F.flatten, compare_fn=compare_fn)

    output0 = (2, 3 * 4 * 5)
    output1 = (4, 5 * 6 * 7)
    cases = [{"input": data0, "output": output0}, {"input": data1, "output": output1}]
    opr_test(cases, F.flatten, compare_fn=compare_fn, start_axis=1)

    output0 = (2, 3, 4 * 5)
    output1 = (4, 5, 6 * 7)
    cases = [{"input": data0, "output": output0}, {"input": data1, "output": output1}]
    opr_test(cases, F.flatten, compare_fn=compare_fn, start_axis=2)

    output0 = (2, 3 * 4, 5)
    output1 = (4, 5 * 6, 7)
    cases = [{"input": data0, "output": output0}, {"input": data1, "output": output1}]
    opr_test(cases, F.flatten, compare_fn=compare_fn, start_axis=1, end_axis=2)


def test_where():
    maskv0 = np.array([[1, 0], [0, 1]], dtype=np.int32)
    xv0 = np.array([[1, np.inf], [np.nan, 4]], dtype=np.float32)
    yv0 = np.array([[5, 6], [7, 8]], dtype=np.float32)

    maskv1 = np.array([[1, 0, 1], [1, 0, 0], [1, 1, 0]], dtype=np.int32)
    xv1 = np.array([[1, np.inf, 2], [0, np.nan, 4], [1, 5, 7]], dtype=np.float32)
    yv1 = np.array([[5, 6, 9], [2, 7, 8], [2, 1, 9]], dtype=np.float32)

    cases = [{"input": [maskv0, xv0, yv0]}, {"input": [maskv1, xv1, yv1]}]
    opr_test(cases, F.where, ref_fn=np.where)


def test_eye():
    dtype = np.float32
    cases = [{"input": [10, 20]}, {"input": [20, 30]}]
    opr_test(cases, F.eye, ref_fn=lambda n, m: np.eye(n, m).astype(dtype), dtype=dtype)


def test_concat():
    def get_data_shape(length: int):
        return (length, 2, 3)

    data1 = np.random.random(get_data_shape(5)).astype("float32")
    data2 = np.random.random(get_data_shape(6)).astype("float32")
    data3 = np.random.random(get_data_shape(7)).astype("float32")

    def run(data1, data2):
        return F.concat([data1, data2])

    cases = [{"input": [data1, data2]}, {"input": [data1, data3]}]
    opr_test(cases, run, ref_fn=lambda x, y: np.concatenate([x, y]))


def test_matrix_mul():
    shape1 = (2, 3)
    shape2 = (3, 4)
    shape3 = (4, 5)
    data1 = np.random.random(shape1).astype("float32")
    data2 = np.random.random(shape2).astype("float32")
    data3 = np.random.random(shape3).astype("float32")

    cases = [{"input": [data1, data2]}, {"input": [data2, data3]}]
    opr_test(cases, F.matrix_mul, ref_fn=np.matmul)


def test_batched_matrix_mul():
    batch_size = 10
    shape1 = (batch_size, 2, 3)
    shape2 = (batch_size, 3, 4)
    shape3 = (batch_size, 4, 5)
    data1 = np.random.random(shape1).astype("float32")
    data2 = np.random.random(shape2).astype("float32")
    data3 = np.random.random(shape3).astype("float32")

    cases = [{"input": [data1, data2]}, {"input": [data2, data3]}]
    for i in range(0, batch_size):

        def compare_fn(x, y):
            x.numpy()[i, ...] == y

        opr_test(
            cases,
            F.batched_matrix_mul,
            compare_fn=compare_fn,
            ref_fn=lambda x, y: np.matmul(x[i, ...], y[i, ...]),
        )


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


def test_round():
    data1_shape = (15,)
    data2_shape = (25,)
    data1 = np.random.random(data1_shape).astype(np.float32)
    data2 = np.random.random(data2_shape).astype(np.float32)

    cases = [{"input": data1}, {"input": data2}]
    opr_test(cases, F.round, ref_fn=np.round)


def test_broadcast_to():
    input1_shape = (20, 30)
    output1_shape = (30, 20, 30)
    data1 = np.random.random(input1_shape).astype(np.float32)

    input2_shape = (10, 20)
    output2_shape = (20, 10, 20)
    data2 = np.random.random(input2_shape).astype(np.float32)

    def compare_fn(x, y):
        assert x.numpy().shape == y

    cases = [
        {"input": [data1, output1_shape], "output": output1_shape},
        {"input": [data2, output2_shape], "output": output2_shape},
    ]
    opr_test(cases, F.broadcast_to, compare_fn=compare_fn)


def test_add_update():
    shape = (2, 3)
    v = np.random.random(shape).astype(np.float32)
    b = Buffer(v)

    u = F.add_update(b, 1)
    assertTensorClose(u.numpy(), v + 1)
    u = F.add_update(b, 1)
    assertTensorClose(u.numpy(), v + 2)

    x = np.ones((2, 2), dtype=np.float32)
    y = x * 0.5
    dest = tensor(x)
    delta = tensor(y)
    r = F.add_update(dest, delta, alpha=tensor(0.9), beta=0.1, bias=0.1)
    assertTensorClose(r.numpy(), x * 0.9 + y * 0.1 + 0.1)


def test_add_update_params():
    b = np.random.random((2, 3)).astype(np.float32)
    y = Buffer(b)

    @jit.trace
    def f(x):
        return F.add_update(y, x)

    f(np.zeros((2, 3)).astype(np.float32))

    z = Buffer(np.zeros((2, 3)).astype(np.float32))
    F.add_update(y, z, beta=0.1)

    res = f(np.ones((2, 3)).astype(np.float32))
    assertTensorClose(res, b + 1)


def test_cross_entropy_with_softmax():
    data1_shape = (1, 2)
    label1_shape = (1,)
    data2_shape = (1, 3)
    label2_shape = (1,)

    data1 = np.array([1, 0.5], dtype=np.float32).reshape(data1_shape)
    label1 = np.array([1], dtype=np.int32).reshape(label1_shape)
    expect1 = F.cross_entropy(F.softmax(tensor(data1)), tensor(label1)).numpy()

    data2 = np.array([0.3, 0.4, 0.3], dtype=np.float32).reshape(data2_shape)
    label2 = np.array([1], dtype=np.int32).reshape(label2_shape)
    expect2 = F.cross_entropy(F.softmax(tensor(data2)), tensor(label2)).numpy()

    cases = [
        {"input": [data1, label1], "output": expect1,},
        {"input": [data2, label2], "output": expect2,},
    ]
    opr_test(cases, F.cross_entropy_with_softmax)


def test_cross_entropy():
    data1_shape = (1, 2)
    label1_shape = (1,)
    data2_shape = (1, 3)
    label2_shape = (1,)

    data1 = np.array([0.5, 0.5], dtype=np.float32).reshape(data1_shape)
    label1 = np.array([1], dtype=np.int32).reshape(label1_shape)
    expect1 = np.array([-np.log(0.5)], dtype=np.float32)

    data2 = np.array([0.3, 0.4, 0.3], dtype=np.float32).reshape(data2_shape)
    label2 = np.array([1], dtype=np.int32).reshape(label2_shape)
    expect2 = np.array([-np.log(0.4)], dtype=np.float32)

    cases = [
        {"input": [data1, label1], "output": expect1,},
        {"input": [data2, label2], "output": expect2,},
    ]
    opr_test(cases, F.cross_entropy)


def test_binary_cross_entropy():
    data1_shape = (2, 2)
    label1_shape = (2, 2)
    data2_shape = (2, 3)
    label2_shape = (2, 3)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def compare_fn(x, y):
        assertTensorClose(x.numpy(), y, max_err=5e-4)

    np.random.seed(123)
    data1 = sigmoid(np.random.uniform(size=data1_shape).astype(np.float32))
    label1 = np.random.uniform(size=label1_shape).astype(np.float32)
    expect1 = np.array([0.6361], dtype=np.float32)

    np.random.seed(123)
    data2 = sigmoid(np.random.uniform(size=data2_shape).astype(np.float32))
    label2 = np.random.uniform(size=label2_shape).astype(np.float32)
    expect2 = np.array([0.6750], dtype=np.float32)

    cases = [
        {"input": [data1, label1], "output": expect1,},
        {"input": [data2, label2], "output": expect2,},
    ]
    opr_test(cases, F.binary_cross_entropy, compare_fn=compare_fn)
