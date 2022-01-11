# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import os
import platform

import numpy as np
import pytest
from utils import get_var_value, make_tensor, opr_test

import megengine.functional as F
from megengine import tensor
from megengine.core._trace_option import use_symbolic_shape
from megengine.core.tensor import megbrain_graph as G
from megengine.core.tensor.utils import astensor1d
from megengine.jit import trace
from megengine.utils.network import Network, set_symbolic_shape
from megengine.utils.network_node import VarNode


def test_eye():
    dtypes = [np.float32, np.bool]
    cases = [{"input": [10, 20]}, {"input": [30]}]
    for dtype in dtypes:
        for case in cases:
            np.testing.assert_allclose(
                F.eye(case["input"], dtype=dtype).numpy(),
                np.eye(*case["input"]).astype(dtype),
            )
            np.testing.assert_allclose(
                F.eye(*case["input"], dtype=dtype).numpy(),
                np.eye(*case["input"]).astype(dtype),
            )
            np.testing.assert_allclose(
                F.eye(tensor(case["input"]), dtype=dtype).numpy(),
                np.eye(*case["input"]).astype(dtype),
            )


def test_full():
    shape = (2, 3)
    values = [True, 4, 5.0]
    for value in values:
        np.testing.assert_allclose(F.full(shape, value).numpy(), np.full(shape, value))
        assert F.full(shape, value).dtype == tensor(value).dtype


@pytest.mark.parametrize("is_varnode", [True, False])
def test_concat(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    def get_data_shape(length: int):
        return (length, 2, 3)

    data1 = np.random.random(get_data_shape(5)).astype("float32")
    data2 = np.random.random(get_data_shape(6)).astype("float32")
    data3 = np.random.random(get_data_shape(7)).astype("float32")

    def run(data1, data2):
        return F.concat([data1, data2])

    cases = [{"input": [data1, data2]}, {"input": [data1, data3]}]
    opr_test(cases, run, ref_fn=lambda x, y: np.concatenate([x, y]), network=network)


@pytest.mark.parametrize("is_varnode", [True, False])
def test_condtake(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    x = np.array([[1, 2, 3], [4, 5, 6]]).astype("float32")
    y = np.array([[True, False, True], [False, True, True]])
    xx = make_tensor(x, network)
    yy = make_tensor(y, network)
    val, idx = F.cond_take(yy, xx)
    if is_varnode:
        np.testing.assert_equal(get_var_value(val), x[y])
        np.testing.assert_equal(get_var_value(idx), np.where(y.reshape(-1))[0])
    else:
        np.testing.assert_equal(val.numpy(), x[y])
        np.testing.assert_equal(idx.numpy(), np.where(y.reshape(-1))[0])


@pytest.mark.parametrize("is_varnode", [True, False])
def test_concat_device(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    data1 = make_tensor(np.random.random((3, 2, 2)).astype("float32"), network, "cpu0")
    data2 = make_tensor(np.random.random((2, 2, 2)).astype("float32"), network, "cpu1")

    out = F.concat([data1, data2], device="cpu0")
    assert str(out.device).split(":")[0] == "cpu0"


@pytest.mark.parametrize("is_varnode", [True, False])
def test_stack(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    data1 = np.random.random((3, 2, 2)).astype("float32")
    data2 = np.random.random((3, 2, 2)).astype("float32")
    data3 = np.random.random((3, 2, 2)).astype("float32")

    cases = [{"input": [data1, data2]}, {"input": [data1, data3]}]
    for ai in range(3):

        def run(data1, data2):
            return F.stack([data1, data2], axis=ai)

        opr_test(
            cases, run, ref_fn=lambda x, y: np.stack([x, y], axis=ai), network=network
        )


@pytest.mark.parametrize("is_varnode", [True, False])
def test_split_basic(is_varnode):
    if is_varnode:
        network = Network()
        saved_symbolic_shape = set_symbolic_shape(False)
    else:
        network = None

    data = np.random.random((2, 3, 4, 5)).astype(np.float32)
    inp = make_tensor(data, network)

    mge_out0 = F.split(inp, 2, axis=3)
    mge_out1 = F.split(inp, [3], axis=3)

    np_out = np.split(data, [3, 5], axis=3)

    assert len(mge_out0) == 2
    assert len(mge_out1) == 2

    np.testing.assert_equal(mge_out0[0].numpy(), np_out[0])
    np.testing.assert_equal(mge_out1[0].numpy(), np_out[0])

    np.testing.assert_equal(mge_out0[1].numpy(), np_out[1])
    np.testing.assert_equal(mge_out1[1].numpy(), np_out[1])

    try:
        F.split(inp, 4)
        assert False
    except ValueError as e:
        pass

    try:
        F.split(inp, [3, 2, 5], axis=3)
        assert False
    except ValueError as e:
        assert str(e) == "Invalid nsplits_or_secions: [3, 2, 5]"

    if is_varnode:
        set_symbolic_shape(saved_symbolic_shape)


@pytest.mark.parametrize("symbolic", [None, False, True])
def test_split(symbolic):
    inp1 = np.random.random((3, 4, 5, 6)).astype(np.float32)
    inp2 = np.random.random((0, 4, 5, 6)).astype(np.float32)

    def ref(inp, nsplits_or_sections, axis):
        return np.split(inp, nsplits_or_sections, axis)

    def func(inp, nsplits_or_sections, axis):
        return F.split(inp, nsplits_or_sections, axis)

    cases = [
        (inp1, 2, 3),
        (inp1, [3], 3),
        (inp1, [3, 3, 5], 3),
        (inp2, 2, 3),
        (inp2, [3], 3),
        (inp2, [3, 3, 5], 3),
    ]

    for case in cases:
        if symbolic is None:
            fn = func
        else:
            fn = trace(symbolic=symbolic)(func)
        for i in range(3 if symbolic is not None else 1):
            ref_out = ref(*case)
            out = fn(tensor(case[0]), case[1], case[2])
            assert len(ref_out) == len(out)
            for idx in range(len(ref_out)):
                np.testing.assert_equal(ref_out[idx], out[idx].numpy())


@pytest.mark.parametrize("is_varnode", [True, False])
def test_reshape(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    x = np.arange(6, dtype="float32")
    xx = make_tensor(x, network)
    y = x.reshape(1, 2, 3)

    for shape in [
        (1, 2, 3),
        (1, -1, 3),
        (1, make_tensor(-1, network), 3),
        np.array([1, -1, 3], dtype="int32"),
        make_tensor([1, -1, 3], network),
    ]:
        yy = F.reshape(xx, shape)
        np.testing.assert_equal(yy.numpy(), y)


@pytest.mark.parametrize("is_varnode", [True, False])
def test_broadcast_auto_infer(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    x = np.random.random((1, 2, 3)).astype(np.float32)
    xx = make_tensor(x, network)

    for shape in [
        (1, 2, 3),
        (1, None, 3),
    ]:
        yy = F.broadcast_to(xx, shape)
        np.testing.assert_equal(yy.numpy(), x)

    with pytest.raises(ValueError):
        F.broadcast_to(xx, (1, -1, 3))

    with pytest.raises(ValueError):
        F.broadcast_to(xx, (None, 1, 2, 3))

    F.broadcast_to(xx, (1, None, 2, 3))
    t = tensor(2, dtype=np.int32)
    F.broadcast_to(xx, (t, None, 2, 3))


@pytest.mark.parametrize("is_trace", [True, False])
def test_reshape_on_empty_tensor(is_trace):
    input1_shape = (100, 0, 1)
    output1_shape = (100, 0, 10)
    data1 = tensor(np.random.random(input1_shape).astype(np.float32))

    input2_shape = (10, 0)
    output2_shape = (0,)
    data2 = tensor(np.random.random(input2_shape).astype(np.float32))

    input3_shape = (10, 0, 10)
    output3_shape = (0, 1, 2, 3)
    data3 = tensor(np.random.random(input3_shape).astype(np.float32))

    def comp(out, target_shp):
        assert out._tuple_shape == target_shp

    def func(x, shp):
        return F.reshape(x, shp)

    cases = [
        [data1, output1_shape],
        [data2, output2_shape],
        [data3, output3_shape],
    ]

    def test(func, inp, comp, target_shp):
        out = func(inp, target_shp)
        comp(out, target_shp)

    if is_trace:
        for symbolic in [False, True]:
            for inp, target_shp in cases:
                func_traced = trace(symbolic=symbolic)(func)
                test(func_traced, inp, comp, target_shp)
                test(func_traced, inp, comp, target_shp)
                test(func_traced, inp, comp, target_shp)
    else:
        for inp, target_shp in cases:
            test(func, inp, comp, target_shp)


@pytest.mark.parametrize("is_varnode", [True, False])
def test_reshape_shape_inference(is_varnode):
    if is_varnode:
        network = Network()
        saved_symbolic_shape = set_symbolic_shape(False)
    else:
        network = None

    x_shape_known = make_tensor([1, 2, 3, 4], network)
    x_shape_unknown = F.broadcast_to(
        make_tensor([1.0], network), shape=make_tensor([1, 1, 1, 1], network).sum()
    )
    tshp_unknown = astensor1d(
        (make_tensor([2], network), make_tensor([2], network)), x_shape_known
    )
    tshp_known = astensor1d((2, 2), x_shape_known)
    tshp_known_unspec = astensor1d((2, -1), x_shape_known)

    def check_shape(output, target):
        source = output.shape
        if isinstance(source, tensor):
            source = source.numpy()
        np.testing.assert_equal(source, target)

    def func(x, target_shape):
        return x.reshape(target_shape)

    cases = [
        {"input": [x_shape_known, tshp_unknown], "output": [(2, 2),]},
        {"input": [x_shape_unknown, tshp_unknown], "output": [(2, 2),]},
        {"input": [x_shape_known, tshp_known], "output": [(2, 2),]},
        {"input": [x_shape_known, tshp_known_unspec], "output": [(2, 2),]},
        {"input": [x_shape_unknown, tshp_known], "output": [(2, 2),]},
        {"input": [x_shape_unknown, tshp_known_unspec], "output": [(2, 2),]},
    ]
    opr_test(cases, func, compare_fn=check_shape, test_trace=True, network=network)
    if is_varnode:
        set_symbolic_shape(saved_symbolic_shape)


@pytest.mark.parametrize("is_varnode", [True, False])
def test_squeeze(is_varnode):
    if is_varnode:
        network = Network()
        saved_symbolic_shape = set_symbolic_shape(False)
    else:
        network = None

    x = np.arange(6, dtype="float32").reshape(1, 2, 3, 1)
    xx = make_tensor(x, network)

    for axis in [None, 3, -4, (3, -4)]:
        y = np.squeeze(x, axis)
        yy = F.squeeze(xx, axis)
        np.testing.assert_equal(y, yy.numpy())

    if is_varnode:
        set_symbolic_shape(saved_symbolic_shape)


@pytest.mark.parametrize("is_varnode", [True, False])
def test_expand_dims(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    x = np.arange(6, dtype="float32").reshape(2, 3)
    xx = make_tensor(x, network)

    for axis in [2, -3, (3, -4), (1, -4)]:
        y = np.expand_dims(x, axis)
        yy = F.expand_dims(xx, axis)
        np.testing.assert_equal(y, yy.numpy())


def test_expand_dims_for_scalar():
    x = np.array(1, dtype="float32")
    xx = make_tensor(x, None)
    for axis in [0, -1, (0, 1), (-1, -2), (0, -1)]:
        y = np.expand_dims(x, axis)
        yy = F.expand_dims(xx, axis)
        np.testing.assert_equal(y, yy.numpy())

    for axis in [1, -2, (1, 2), (-2, -3)]:
        np.testing.assert_raises(np.AxisError, np.expand_dims, x, axis)
        np.testing.assert_raises(RuntimeError, F.expand_dims, xx, axis)


@pytest.mark.parametrize("is_varnode", [True, False])
def test_elemwise_dtype_promotion(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    x = np.random.rand(2, 3).astype("float32")
    y = np.random.rand(1, 3).astype("float16")
    xx = make_tensor(x, network)
    yy = make_tensor(y, network)
    z = xx * yy
    np.testing.assert_equal(z.numpy(), x * y)

    z = xx + y
    np.testing.assert_equal(z.numpy(), x + y)

    z = x - yy
    np.testing.assert_equal(z.numpy(), x - y)


@pytest.mark.parametrize("is_varnode", [True, False])
def test_linspace(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    cases = [
        {"input": [1, 9, 9]},
        {"input": [3, 10, 8]},
    ]
    opr_test(
        cases,
        F.linspace,
        ref_fn=lambda start, end, step: np.linspace(start, end, step, dtype=np.float32),
        network=network,
    )

    cases = [
        {"input": [9, 1, 9]},
        {"input": [10, 3, 8]},
    ]
    opr_test(
        cases,
        F.linspace,
        ref_fn=lambda start, end, step: np.linspace(start, end, step, dtype=np.float32),
        network=network,
    )

    cases = [
        {"input": [1, make_tensor(9, network), 9]},
        {"input": [make_tensor(1, network), 9, make_tensor(9, network)]},
    ]
    opr_test(
        cases,
        F.linspace,
        ref_fn=lambda start, end, step: np.linspace(1, 9, 9, dtype=np.float32),
        network=network,
    )


@pytest.mark.parametrize("is_varnode", [True, False])
def test_arange(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    cases = [
        {"input": [1, 9, 1]},
        {"input": [2, 10, 2]},
    ]
    opr_test(
        cases,
        F.arange,
        ref_fn=lambda start, end, step: np.arange(start, end, step, dtype=np.float32),
        network=network,
    )

    cases = [
        {"input": [9, 1, -1]},
        {"input": [10, 2, -2]},
    ]
    opr_test(
        cases,
        F.arange,
        ref_fn=lambda start, end, step: np.arange(start, end, step, dtype=np.float32),
        network=network,
    )

    cases = [
        {"input": [9.3, 1.2, -0.5]},
        {"input": [10.3, 2.1, -1.7]},
    ]
    opr_test(
        cases,
        F.arange,
        ref_fn=lambda start, end, step: np.arange(start, end, step, dtype=np.float32),
        network=network,
    )


@pytest.mark.parametrize("is_varnode", [True, False])
def test_round(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    data1_shape = (15,)
    data2_shape = (25,)
    data1 = np.random.random(data1_shape).astype(np.float32)
    data2 = np.random.random(data2_shape).astype(np.float32)

    cases = [{"input": data1}, {"input": data2}]
    opr_test(cases, F.round, ref_fn=np.round, network=network)


@pytest.mark.parametrize("is_varnode", [True, False])
def test_flatten(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    data0_shape = (2, 3, 4, 5)
    data1_shape = (4, 5, 6, 7)
    data0 = np.random.random(data0_shape).astype(np.float32)
    data1 = np.random.random(data1_shape).astype(np.float32)

    def compare_fn(x, y):
        assert x._tuple_shape[0] == y

    output0 = (2 * 3 * 4 * 5,)
    output1 = (4 * 5 * 6 * 7,)
    cases = [
        {"input": data0, "output": output0},
        {"input": data1, "output": output1},
    ]
    opr_test(cases, F.flatten, compare_fn=compare_fn, network=network)

    output0 = (2, 3 * 4 * 5)
    output1 = (4, 5 * 6 * 7)
    cases = [
        {"input": data0, "output": output0},
        {"input": data1, "output": output1},
    ]
    opr_test(cases, F.flatten, compare_fn=compare_fn, start_axis=1, network=network)

    output0 = (2, 3, 4 * 5)
    output1 = (4, 5, 6 * 7)
    cases = [
        {"input": data0, "output": output0},
        {"input": data1, "output": output1},
    ]
    opr_test(cases, F.flatten, compare_fn=compare_fn, start_axis=2, network=network)

    output0 = (2, 3 * 4, 5)
    output1 = (4, 5 * 6, 7)
    cases = [
        {"input": data0, "output": output0},
        {"input": data1, "output": output1},
    ]
    opr_test(
        cases,
        F.flatten,
        compare_fn=compare_fn,
        start_axis=1,
        end_axis=2,
        network=network,
    )


@pytest.mark.parametrize("is_varnode", [True, False])
def test_broadcast(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    input1_shape = (20, 30)
    output1_shape = (30, 20, 30)
    data1 = np.random.random(input1_shape).astype(np.float32)

    input2_shape = (10, 1)
    output2_shape = (20, 10, 20)
    data2 = np.random.random(input2_shape).astype(np.float32)

    input3_shape = (10, 10)
    output3_shape = (10, 10)
    data3 = np.random.random(input3_shape).astype(np.float32)

    def compare_fn(x, y):
        assert x._tuple_shape[0] == y

    cases = [
        {"input": [data1, output1_shape], "output": output1_shape},
        {"input": [data2, output2_shape], "output": output2_shape},
        {"input": [data3, output3_shape], "output": output3_shape},
    ]
    opr_test(cases, F.broadcast_to, compare_fn=compare_fn, network=network)

    x = F.ones((2, 1, 3))
    with pytest.raises(RuntimeError):
        F.broadcast_to(x, (2, 3, 4))

    with pytest.raises(RuntimeError):
        F.broadcast_to(x, (4, 1, 3))

    with pytest.raises(RuntimeError):
        F.broadcast_to(x, (1, 3))


@pytest.mark.parametrize("is_trace", [True, False])
def test_broadcast_on_empty_tensor(is_trace):
    input1_shape = (100, 0, 1)
    output1_shape = (100, 0, 10)
    data1 = tensor(np.random.random(input1_shape).astype(np.float32))

    input2_shape = (10, 0)
    output2_shape = (10, 10, 0)
    data2 = tensor(np.random.random(input2_shape).astype(np.float32))

    input3_shape = (0, 0, 1, 10)
    output3_shape = (10, 0, 0, 10, 10)
    data3 = tensor(np.random.random(input3_shape).astype(np.float32))

    def comp(out, target_shp):
        assert out._tuple_shape == target_shp

    def func(x, shp):
        return F.broadcast_to(x, shp)

    cases = [
        [data1, output1_shape],
        [data2, output2_shape],
        [data3, output3_shape],
    ]

    def test(func, inp, comp, target_shp):
        out = func(inp, target_shp)
        comp(out, target_shp)

    if is_trace:
        for symbolic in [False, True]:
            for inp, target_shp in cases:
                func_traced = trace(symbolic=symbolic)(func)
                test(func_traced, inp, comp, target_shp)
                test(func_traced, inp, comp, target_shp)
                test(func_traced, inp, comp, target_shp)
    else:
        for inp, target_shp in cases:
            test(func, inp, comp, target_shp)


@pytest.mark.parametrize("is_varnode", [True, False])
def test_utils_astensor1d(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    reference = make_tensor(0, network)

    # literal
    x = [1, 2, 3]
    for dtype in [None, "float32"]:
        xx = astensor1d(x, reference, dtype=dtype)
        assert isinstance(xx, type(reference))
        np.testing.assert_equal(xx.numpy(), x)

    # numpy array
    x = np.asarray([1, 2, 3], dtype="int32")
    for dtype in [None, "float32"]:
        xx = astensor1d(x, reference, dtype=dtype)
        assert isinstance(xx, type(reference))
        np.testing.assert_equal(xx.numpy(), x.astype(dtype) if dtype else x)

    # tensor
    x = make_tensor([1, 2, 3], network)
    for dtype in [None, "float32"]:
        xx = astensor1d(x, reference, dtype=dtype)
        assert isinstance(xx, type(reference))
        np.testing.assert_equal(xx.numpy(), x.numpy())

    # mixed
    x = [1, make_tensor(2, network), 3]
    for dtype in [None, "float32"]:
        xx = astensor1d(x, reference, dtype=dtype)
        assert isinstance(xx, type(reference))
        np.testing.assert_equal(xx.numpy(), [1, 2, 3])


def test_device():
    x = tensor([1, 2, 3], dtype="float32")

    y1 = F.eye(x.shape, dtype="float32")
    y2 = F.eye(x.shape, dtype="float32", device=None)
    np.testing.assert_almost_equal(y1.numpy(), y2.numpy())

    y3 = F.eye(x.shape, dtype="float32", device="xpux")
    y4 = F.eye(x.shape, dtype="float32", device=x.device)
    np.testing.assert_almost_equal(y3.numpy(), y4.numpy())

    y5 = F.full((3, 2), 4, device=x.device)
    y6 = F.full((3, 2), 4, device="xpux")
    np.testing.assert_almost_equal(y5.numpy(), y6.numpy())


@pytest.mark.parametrize("is_varnode", [True, False])
def test_identity(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    x = make_tensor(np.random.random((5, 10)).astype(np.float32), network)
    y = F.copy(x)
    np.testing.assert_equal(y.numpy(), x)


def copy_test(dst, src, network):
    data = np.random.random((2, 3)).astype(np.float32)
    x = make_tensor(data, device=src, network=network)
    y = F.copy(x, dst)
    assert np.allclose(data, y.numpy())
    if network is None:
        z = x.to(dst)
        assert np.allclose(data, z.numpy())


@pytest.mark.require_ngpu(1)
@pytest.mark.parametrize("is_varnode", [True, False])
def test_copy_h2d(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    copy_test("cpu0", "gpu0", network=network)


@pytest.mark.require_ngpu(1)
@pytest.mark.parametrize("is_varnode", [True, False])
def test_copy_d2h(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    copy_test("gpu0", "cpu0", network=network)


@pytest.mark.require_ngpu(2)
@pytest.mark.parametrize("is_varnode", [True, False])
def test_copy_d2d(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    copy_test("gpu0", "gpu1", network=network)
    copy_test("gpu0:0", "gpu0:1", network=network)


@pytest.mark.require_ngpu(2)
@pytest.mark.parametrize(
    "shape, device_src, device_dst",
    [
        ((0,), "cpu0", "cpu0"),
        ((10, 0), "cpu0", "cpu1"),
        ((2, 0, 3), "cpu0", "gpu0"),
        ((1, 0, 1, 0), "gpu0", "cpu0"),
        ((2, 3, 4, 5, 0), "gpu0", "gpu1"),
    ],
)
@pytest.mark.parametrize("is_symbolic", [None, True, False])
def test_copy_empty(shape, device_src, device_dst, is_symbolic):
    inp = tensor(np.random.randn(*shape).astype("float32"), device=device_src)

    def func(inp):
        return F.copy(inp, device_dst)

    if is_symbolic is not None:
        func = trace(symbolic=is_symbolic)(func)

    for _ in range(3):
        out = func(inp)
        assert out.numpy().shape == shape
        assert out.device == device_dst
        if is_symbolic is None:
            break


@pytest.mark.parametrize(
    "shape, repeats, axis",
    [
        ((2,), 2, 0),
        ((2, 3, 4, 5), 3, 0),
        ((2, 3, 4, 5), 4, 3),
        ((2,), 2, None),
        ((2, 3, 4, 5), 3, None),
        ((), 1, None),
        ((), 10, None),
    ],
)
@pytest.mark.parametrize("is_varnode", [True, False])
def test_repeat(shape, repeats, axis, is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    def repeat_func(inp):
        return F.repeat(inp=inp, repeats=repeats, axis=axis)

    if shape != ():
        cases = [
            {"input": np.random.randn(*shape).astype("float32")},
        ]
    else:
        cases = [{"input": np.array(1.23)}]

    opr_test(
        cases,
        repeat_func,
        ref_fn=lambda inp: np.repeat(inp, repeats, axis),
        network=network,
    )


@pytest.mark.parametrize(
    "shape, reps",
    [
        ((2,), (2,)),
        ((2, 3, 4, 5), (1, 1, 1, 1)),
        ((2, 3, 4, 5), (1, 2, 3, 4)),
        ((2, 3, 4, 5), (2, 2, 2, 2, 2, 2, 2)),
    ],
)
@pytest.mark.parametrize("is_varnode", [True])
def test_tile(shape, reps, is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    def tile_func(inp):
        return F.tile(inp=inp, reps=reps)

    cases = [{"input": np.random.randn(*shape).astype("float32")}]

    opr_test(cases, tile_func, ref_fn=lambda inp: np.tile(inp, reps), network=network)


@pytest.mark.parametrize(
    "shape, shifts, axis",
    [
        ((2, 3), 0, None),
        ((2, 3), 1, 0),
        ((2, 3), 100, 0),
        ((2, 3), -100, 0),
        ((2, 3, 4, 5), (-1, 1), (0, 1)),
        ((2, 3, 4, 5), (-2, 1, 2), (1, 2, 3)),
    ],
)
@pytest.mark.parametrize("is_varnode", [True, False])
def test_roll(shape, shifts, axis, is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    inp = np.random.randn(*shape).astype("float32")

    def func(inp):
        return F.roll(inp, shifts, axis)

    cases = [
        {"input": inp},
    ]

    opr_test(
        cases, func, ref_fn=lambda inp: np.roll(inp, shifts, axis), network=network
    )


@pytest.mark.parametrize(
    "shape, shifts, axis", [((10, 0), 5, 1), ((10, 0), -10, 1),],
)
@pytest.mark.parametrize("is_symbolic", [None, True, False])
def test_roll_empty_tensor(shape, shifts, axis, is_symbolic):
    inp = tensor(np.random.randn(*shape).astype("float32"))

    def func(inp):
        return F.roll(inp, shifts, axis)

    if is_symbolic is not None:
        func = trace(symbolic=is_symbolic)(func)

    out_ref = np.roll(inp.numpy(), shifts, axis)
    for _ in range(3):
        out = F.roll(inp, shifts, axis)
        np.testing.assert_equal(out.numpy(), out_ref)
        if is_symbolic is None:
            break
