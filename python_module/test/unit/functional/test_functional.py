# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np
import pytest
from helpers import opr_test

import megengine._internal as mgb
import megengine.functional as F
from megengine import Buffer, Parameter, is_cuda_available, jit, tensor
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

    cases = [
        {"input": [maskv0, xv0, yv0]},
        {"input": [maskv1, xv1, yv1]},
    ]
    opr_test(cases, F.where, ref_fn=np.where)

    maskv2 = np.array([1, 1, 1], dtype=np.int32)
    xv2 = np.array([1, 3, 2], dtype=np.float32)
    yv2 = np.array([5, 6, 9], dtype=np.float32)

    maskv3 = np.array([0, 0, 0], dtype=np.int32)
    xv3 = np.array([1, 3, 2], dtype=np.float32)
    yv3 = np.array([5, 6, 9], dtype=np.float32)

    cases = [
        {"input": [maskv2, xv2, yv2]},
        {"input": [maskv3, xv3, yv3]},
    ]
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


def test_linspace():
    cases = [
        {"input": [1, 9, 9]},
        {"input": [3, 10, 8]},
    ]
    opr_test(
        cases,
        F.linspace,
        ref_fn=lambda start, end, step: np.linspace(start, end, step, dtype=np.float32),
    )

    cases = [
        {"input": [9, 1, 9]},
        {"input": [10, 3, 8]},
    ]
    opr_test(
        cases,
        F.linspace,
        ref_fn=lambda start, end, step: np.linspace(start, end, step, dtype=np.float32),
    )


def test_arange():
    cases = [
        {"input": [1, 9, 1]},
        {"input": [2, 10, 2]},
    ]
    opr_test(
        cases,
        F.arange,
        ref_fn=lambda start, end, step: np.arange(start, end, step, dtype=np.float32),
    )

    cases = [
        {"input": [9, 1, -1]},
        {"input": [10, 2, -2]},
    ]
    opr_test(
        cases,
        F.arange,
        ref_fn=lambda start, end, step: np.arange(start, end, step, dtype=np.float32),
    )

    cases = [
        {"input": [9.3, 1.2, -0.5]},
        {"input": [10.3, 2.1, -1.7]},
    ]
    opr_test(
        cases,
        F.arange,
        ref_fn=lambda start, end, step: np.arange(start, end, step, dtype=np.float32),
    )


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


def test_hinge_loss():
    np.random.seed(123)
    # case with L1 norm
    cases = []
    for shape in [(2, 2), (2, 3)]:
        data = np.random.uniform(size=shape).astype(np.float32)
        label = 2 * np.random.randint(0, 1, size=shape).astype(np.int32) - 1
        expect = np.clip(0, np.inf, 1 - data * label).sum(axis=1).mean()
        cases.append({"input": [data, label], "output": tensor(expect)})

    opr_test(cases, F.hinge_loss)

    # cases with L2 norm
    cases = []
    for shape in [(2, 2), (2, 3)]:
        data = np.random.uniform(size=shape).astype(np.float32)
        label = 2 * np.random.randint(0, 1, size=shape).astype(np.int32) - 1
        expect = ((np.clip(0, np.inf, 1 - data * label) ** 2).sum(axis=1)).mean()
        cases.append({"input": [data, label], "output": tensor(expect)})

    def hinge_loss_with_l2_norm(pred, label):
        return F.hinge_loss(pred, label, "L2")

    opr_test(cases, hinge_loss_with_l2_norm)


@pytest.mark.skip
def test_conv_bias():
    inp_scale = 0.01
    w_scale = 0.02
    outp_scale = 0.1
    inp_dtype = mgb.dtype.qint8(inp_scale)
    w_dtype = mgb.dtype.qint8(w_scale)
    b_dtype = mgb.dtype.qint32(inp_scale * w_scale)
    out_dtype = mgb.dtype.qint8(outp_scale)

    def run(
        N,
        IC,
        OC,
        IH,
        IW,
        KH,
        KW,
        PH,
        PW,
        SH,
        SW,
        has_bias=True,
        nonlinear_mode="IDENTITY",
    ):
        inp_v = np.random.normal(size=(N, IC, IH, IW))
        w_v = np.random.normal(size=(OC, IC, KW, KW))
        b_v = np.random.normal(size=(1, OC, 1, 1))
        inp_scale = mgb.dtype.get_scale(inp_dtype)
        w_scale = mgb.dtype.get_scale(w_dtype)
        b_scale = mgb.dtype.get_scale(b_dtype)

        inpv = mgb.dtype.convert_to_qint8(inp_v * inp_scale, inp_dtype)
        wv = mgb.dtype.convert_to_qint8(w_v * w_scale, w_dtype)
        bv = mgb.dtype.convert_to_qint32(b_v * b_scale, b_dtype)

        inp_int8 = tensor(inpv, dtype=inp_dtype)
        w_int8 = Parameter(wv, dtype=w_dtype)
        b_int32 = Parameter(bv, dtype=b_dtype)

        inp_fp32 = inp_int8.astype("float32")
        w_fp32 = w_int8.astype("float32")
        b_fp32 = b_int32.astype("float32")

        jit.trace.enabled = True
        b_symbolic = True

        def convert_to_nchw4(var):
            return var.reshape(
                var.shapeof(0), var.shapeof(1) // 4, 4, var.shapeof(2), var.shapeof(3)
            ).dimshuffle(0, 1, 3, 4, 2)

        @jit.trace(symbolic=b_symbolic)
        def run_conv2d(inp, w, b):
            O = F.conv2d(
                inp, w, b if has_bias else None, stride=(SH, SW), padding=(PH, PW),
            )
            if nonlinear_mode == "RELU":
                return F.relu(O)
            else:
                return O

        @jit.trace(symbolic=b_symbolic)
        def run_conv_bias(inp, w, b, format="NCHW"):
            b = b if has_bias else np.zeros_like(b)
            if format == "NCHW4":
                inp = convert_to_nchw4(inp)
                w = convert_to_nchw4(w)
                b = F.flatten(b)
            return F.conv_bias_activation(
                inp,
                w,
                b,
                stride=(SH, SW),
                padding=(PH, PW),
                dtype=out_dtype,
                nonlinear_mode=nonlinear_mode,
            )

        format = "NCHW4" if is_cuda_available() else "NCHW"

        expected = run_conv2d(inp_fp32, w_fp32, b_fp32)
        expected = expected.astype(out_dtype).astype("float32")
        result = run_conv_bias(inp_int8, w_int8, b_int32, format=format).astype(
            "float32"
        )
        if format == "NCHW4":
            result = result.dimshuffle(0, 1, 4, 2, 3)
        expected = F.flatten(expected)
        result = F.flatten(result)
        assertTensorClose(result.numpy(), expected.numpy())

    if not is_cuda_available():
        run(1, 4, 4, 24, 33, 1, 1, 2, 3, 1, 1, False)
        run(10, 12, 24, 46, 46, 1, 1, 2, 1, 3, 1, False)
        run(10, 36, 8, 46, 26, 2, 2, 2, 1, 1, 2, False)

        run(1, 4, 4, 24, 33, 1, 1, 2, 3, 1, 1)
        run(10, 12, 24, 46, 46, 1, 1, 2, 1, 3, 1)
        run(10, 36, 8, 46, 26, 2, 2, 2, 1, 1, 2)

        run(10, 36, 8, 46, 26, 2, 2, 2, 1, 1, 2, False, "RELU")
        run(10, 36, 8, 46, 26, 2, 2, 2, 1, 1, 2, True, "RELU")


def test_softplus():
    x = np.arange(1000).astype(np.float32)
    out = F.softplus(tensor(x))
    mask = x <= 20
    with np.errstate(over="ignore"):
        expected = np.where(mask, np.log(1 + np.exp(x)), x)
    assertTensorClose(out, expected)
    beta = 2
    out = F.softplus(tensor(x), beta=beta, threshold=30)
    mask = beta * x <= 30
    # ignore overflow
    with np.errstate(over="ignore"):
        expected = np.where(mask, np.log(1 + np.exp(x * beta)) / beta, x)
    assertTensorClose(out, expected)
