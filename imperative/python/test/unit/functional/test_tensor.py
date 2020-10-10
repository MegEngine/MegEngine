# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import os
import platform

import numpy as np
import pytest
from utils import opr_test

import megengine.functional as F
from megengine import tensor
from megengine.core._trace_option import use_tensor_shape
from megengine.core.tensor.utils import astensor1d
from megengine.distributed.helper import get_device_count_by_fork


def test_eye():
    dtype = np.float32
    cases = [{"input": [10, 20]}, {"input": [20, 30]}]
    for case in cases:
        np.testing.assert_allclose(
            F.eye(case["input"], dtype=dtype).numpy(),
            np.eye(*case["input"]).astype(dtype),
        )


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


def test_concat_device():
    data1 = tensor(np.random.random((3, 2, 2)).astype("float32"), device="cpu0")
    data2 = tensor(np.random.random((2, 2, 2)).astype("float32"), device="cpu1")

    out = F.concat([data1, data2], device="cpu0")
    assert str(out.device).split(":")[0] == "cpu0"


def test_stack():
    data1 = np.random.random((3, 2, 2)).astype("float32")
    data2 = np.random.random((3, 2, 2)).astype("float32")
    data3 = np.random.random((3, 2, 2)).astype("float32")

    cases = [{"input": [data1, data2]}, {"input": [data1, data3]}]
    for ai in range(3):

        def run(data1, data2):
            return F.stack([data1, data2], axis=ai)

        opr_test(cases, run, ref_fn=lambda x, y: np.stack([x, y], axis=ai))


def test_split():
    data = np.random.random((2, 3, 4, 5)).astype(np.float32)
    mge_out1 = F.split(tensor(data), 2, axis=3)
    mge_out2 = F.split(tensor(data), [3, 5], axis=3)

    np_out = np.split(data, [3, 5], axis=3)

    np.testing.assert_equal(mge_out1[0].numpy(), mge_out2[0].numpy())
    np.testing.assert_equal(mge_out1[0].numpy(), np_out[0])


def test_reshape():
    x = np.arange(6, dtype="float32")
    xx = tensor(x)
    y = x.reshape(1, 2, 3)

    for shape in [
        (1, 2, 3),
        (1, -1, 3),
        (1, tensor(-1), 3),
        np.array([1, -1, 3], dtype="int32"),
        tensor([1, -1, 3]),
    ]:
        yy = F.reshape(xx, shape)
        np.testing.assert_equal(yy.numpy(), y)


def test_squeeze():
    x = np.arange(6, dtype="float32").reshape(1, 2, 3, 1)
    xx = tensor(x)

    for axis in [None, 3, -4, (3, -4)]:
        y = np.squeeze(x, axis)
        yy = F.squeeze(xx, axis)
        np.testing.assert_equal(y, yy.numpy())


def test_expand_dims():
    x = np.arange(6, dtype="float32").reshape(2, 3)
    xx = tensor(x)

    for axis in [2, -3, (3, -4), (1, -4)]:
        y = np.expand_dims(x, axis)
        yy = F.expand_dims(xx, axis)
        np.testing.assert_equal(y, yy.numpy())


def test_elemwise_dtype_promotion():
    x = np.random.rand(2, 3).astype("float32")
    y = np.random.rand(1, 3).astype("float16")
    xx = tensor(x)
    yy = tensor(y)
    z = xx * yy
    np.testing.assert_equal(z.numpy(), x * y)

    z = xx + y
    np.testing.assert_equal(z.numpy(), x + y)

    z = x - yy
    np.testing.assert_equal(z.numpy(), x - y)


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


def test_round():
    data1_shape = (15,)
    data2_shape = (25,)
    data1 = np.random.random(data1_shape).astype(np.float32)
    data2 = np.random.random(data2_shape).astype(np.float32)

    cases = [{"input": data1}, {"input": data2}]
    opr_test(cases, F.round, ref_fn=np.round)


def test_flatten():
    data0_shape = (2, 3, 4, 5)
    data1_shape = (4, 5, 6, 7)
    data0 = np.random.random(data0_shape).astype(np.float32)
    data1 = np.random.random(data1_shape).astype(np.float32)

    def compare_fn(x, y):
        assert x.shape[0] == y

    output0 = (2 * 3 * 4 * 5,)
    output1 = (4 * 5 * 6 * 7,)
    cases = [
        {"input": data0, "output": output0},
        {"input": data1, "output": output1},
    ]
    opr_test(cases, F.flatten, compare_fn=compare_fn)

    output0 = (2, 3 * 4 * 5)
    output1 = (4, 5 * 6 * 7)
    cases = [
        {"input": data0, "output": output0},
        {"input": data1, "output": output1},
    ]
    opr_test(cases, F.flatten, compare_fn=compare_fn, start_axis=1)

    output0 = (2, 3, 4 * 5)
    output1 = (4, 5, 6 * 7)
    cases = [
        {"input": data0, "output": output0},
        {"input": data1, "output": output1},
    ]
    opr_test(cases, F.flatten, compare_fn=compare_fn, start_axis=2)

    output0 = (2, 3 * 4, 5)
    output1 = (4, 5 * 6, 7)
    cases = [
        {"input": data0, "output": output0},
        {"input": data1, "output": output1},
    ]
    opr_test(cases, F.flatten, compare_fn=compare_fn, start_axis=1, end_axis=2)


def test_broadcast():
    input1_shape = (20, 30)
    output1_shape = (30, 20, 30)
    data1 = np.random.random(input1_shape).astype(np.float32)

    input2_shape = (10, 1)
    output2_shape = (20, 10, 20)
    data2 = np.random.random(input2_shape).astype(np.float32)

    def compare_fn(x, y):
        assert x.shape[0] == y

    cases = [
        {"input": [data1, output1_shape], "output": output1_shape},
        {"input": [data2, output2_shape], "output": output2_shape},
    ]
    opr_test(cases, F.broadcast_to, compare_fn=compare_fn)

    x = F.ones((2, 1, 3))
    with pytest.raises(ValueError):
        F.broadcast_to(x, (2, 3, 4))

    with pytest.raises(ValueError):
        F.broadcast_to(x, (4, 1, 3))

    with pytest.raises(ValueError):
        F.broadcast_to(x, (1, 3))


def test_utils_astensor1d():
    reference = tensor(0)

    # literal
    x = [1, 2, 3]
    for dtype in [None, "float32"]:
        xx = astensor1d(x, reference, dtype=dtype)
        assert type(xx) is tensor
        np.testing.assert_equal(xx.numpy(), x)

    # numpy array
    x = np.asarray([1, 2, 3], dtype="int32")
    for dtype in [None, "float32"]:
        xx = astensor1d(x, reference, dtype=dtype)
        assert type(xx) is tensor
        np.testing.assert_equal(xx.numpy(), x.astype(dtype) if dtype else x)

    # tensor
    x = tensor([1, 2, 3], dtype="int32")
    for dtype in [None, "float32"]:
        xx = astensor1d(x, reference, dtype=dtype)
        assert type(xx) is tensor
        np.testing.assert_equal(xx.numpy(), x.numpy())

    # mixed
    x = [1, tensor(2), 3]
    for dtype in [None, "float32"]:
        xx = astensor1d(x, reference, dtype=dtype)
        assert type(xx) is tensor
        np.testing.assert_equal(xx.numpy(), [1, 2, 3])


def test_device():
    x = tensor([1, 2, 3], dtype="float32")

    y1 = F.eye(x.shape, dtype="float32")
    y2 = F.eye(x.shape, dtype="float32", device=None)
    np.testing.assert_almost_equal(y1.numpy(), y2.numpy())

    y3 = F.eye(x.shape, dtype="float32", device="xpux")
    y4 = F.eye(x.shape, dtype="float32", device=x.device.to_c())
    np.testing.assert_almost_equal(y3.numpy(), y4.numpy())

    y5 = F.full((3, 2), 4, device=x.device)
    y6 = F.full((3, 2), 4, device="xpux")
    np.testing.assert_almost_equal(y5.numpy(), y6.numpy())


def test_identity():
    x = tensor(np.random.random((5, 10)).astype(np.float32))
    y = F.copy(x)
    np.testing.assert_equal(y.numpy(), x)


def copy_test(dst, src):
    data = np.random.random((2, 3)).astype(np.float32)
    x = tensor(data, device=src)
    y = F.copy(x, dst)
    assert np.allclose(data, y.numpy())
    z = x.to(dst)
    assert np.allclose(data, z.numpy())


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.skipif(get_device_count_by_fork("gpu") == 0, reason="CUDA is disabled")
def test_copy_h2d():
    copy_test("cpu0", "gpu0")


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.skipif(get_device_count_by_fork("gpu") == 0, reason="CUDA is disabled")
def test_copy_d2h():
    copy_test("gpu0", "cpu0")


@pytest.mark.skipif(
    platform.system() == "Darwin", reason="do not imp GPU mode at macos now"
)
@pytest.mark.skipif(
    platform.system() == "Windows", reason="do not imp GPU mode at Windows now"
)
@pytest.mark.skipif(get_device_count_by_fork("gpu") < 2, reason="need more gpu device")
def test_copy_d2d():
    copy_test("gpu0", "gpu1")
    copy_test("gpu0:0", "gpu0:1")
