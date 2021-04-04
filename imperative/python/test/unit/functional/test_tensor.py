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
from utils import make_tensor, opr_test

import megengine.functional as F
from megengine import tensor
from megengine.core._trace_option import use_symbolic_shape
from megengine.core.tensor import megbrain_graph as G
from megengine.core.tensor.utils import astensor1d
from megengine.distributed.helper import get_device_count_by_fork
from megengine.utils.network import Network
from megengine.utils.network_node import VarNode


def test_eye():
    dtype = np.float32
    cases = [{"input": [10, 20]}, {"input": [30]}]
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
def test_split(is_varnode):
    if is_varnode:
        network = Network()
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
        F.split(inp, [3, 3, 5], axis=3)
        assert False
    except ValueError as e:
        assert str(e) == "Invalid nsplits_or_secions: [3, 3, 5]"


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
def test_reshape_shape_inference(is_varnode):
    if is_varnode:
        network = Network()
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


@pytest.mark.parametrize("is_varnode", [True, False])
def test_squeeze(is_varnode):
    if is_varnode:
        network = Network()
    else:
        network = None

    x = np.arange(6, dtype="float32").reshape(1, 2, 3, 1)
    xx = make_tensor(x, network)

    for axis in [None, 3, -4, (3, -4)]:
        y = np.squeeze(x, axis)
        yy = F.squeeze(xx, axis)
        np.testing.assert_equal(y, yy.numpy())


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
        assert x.shape[0] == y

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
        assert x.shape[0] == y

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
