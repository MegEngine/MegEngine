# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import io
from tempfile import mkstemp

import numpy as np
import pytest

import megengine.core.tensor.megbrain_graph as G
import megengine.functional as F
from megengine import cgtools, tensor
from megengine.core._trace_option import set_tensor_shape
from megengine.core.ops import builtin as ops
from megengine.core.tensor.core import apply
from megengine.core.tensor.raw_tensor import as_raw_tensor
from megengine.functional import exp, log
from megengine.jit import exclude_from_trace, trace


def test_trace():
    for symbolic in [False, True]:

        @trace(symbolic=symbolic)
        def f(x):
            op = ops.Elemwise(mode="negate")
            (y,) = apply(op, x)
            return y

        x = as_raw_tensor([1]).numpy()
        y = f.__wrapped__(as_raw_tensor(x)).numpy()

        for i in range(3):
            np.testing.assert_equal(f(as_raw_tensor(x)).numpy(), y)


def test_exclude_from_trace():
    for symbolic in [False, True]:

        @trace(symbolic=symbolic)
        def f(x):
            neg = ops.Elemwise(mode="negate")
            (x,) = apply(neg, x)
            with exclude_from_trace():
                if i % 2:
                    (x,) = apply(neg, x)
            (x,) = apply(neg, x)
            return x

        x = as_raw_tensor([1]).numpy()

        for i in range(3):
            y = f.__wrapped__(as_raw_tensor(x)).numpy()
            np.testing.assert_equal(f(as_raw_tensor(x)).numpy(), y)


def test_print_in_trace():
    for symbolic in [False]:  # cannot read value in symbolic mode

        @trace(symbolic=symbolic)
        def f(x):
            nonlocal buf
            neg = ops.Elemwise(mode="negate")
            (x,) = apply(neg, x)
            buf = x.numpy()
            (x,) = apply(neg, x)
            return x

        buf = None
        x = as_raw_tensor([1]).numpy()

        for i in range(3):
            y = f.__wrapped__(as_raw_tensor(x)).numpy()
            z = buf
            buf = None
            np.testing.assert_equal(f(as_raw_tensor(x)).numpy(), y)
            np.testing.assert_equal(z, buf)


def test_dump():
    @trace(symbolic=True, capture_as_const=True)
    def f(a, b):
        op = ops.Elemwise(mode="add")
        (y,) = apply(op, a, b)
        return y

    a = as_raw_tensor([2]).numpy()
    b = as_raw_tensor([4]).numpy()
    y = f.__wrapped__(as_raw_tensor(a), as_raw_tensor(b)).numpy()

    for i in range(3):
        np.testing.assert_equal(f(as_raw_tensor(a), as_raw_tensor(b)).numpy(), y)

    file = io.BytesIO()
    dump_info = f.dump(file)
    assert dump_info.nr_opr == 3
    np.testing.assert_equal(dump_info.inputs, ["h2d[0]", "h2d[2]"])
    np.testing.assert_equal(dump_info.outputs, ["ADD(h2d[0],h2d[2])[4]"])
    file.seek(0)
    result = cgtools.load_and_inference(file, [a, b])
    np.testing.assert_equal(result[0], y)


def test_capture_dump():
    a = as_raw_tensor([2])

    @trace(symbolic=True, capture_as_const=True)
    def f(x):
        op = ops.Elemwise(mode="mul")
        (y,) = apply(op, x, a)
        return y

    x = as_raw_tensor([3]).numpy()
    y = f.__wrapped__(as_raw_tensor(x)).numpy()

    for i in range(3):
        np.testing.assert_equal(f(as_raw_tensor(x)).numpy(), y)

    file = io.BytesIO()
    f.dump(file)
    file.seek(0)
    result = cgtools.load_and_inference(file, [x])
    np.testing.assert_equal(result[0], y)


def test_dump_volatile():
    p = as_raw_tensor([2])

    @trace(symbolic=True, capture_as_const=True)
    def f(x):
        op = ops.Elemwise(mode="mul")
        (y,) = apply(op, x, p)
        return y

    x = as_raw_tensor([3]).numpy()
    y = f.__wrapped__(as_raw_tensor(x)).numpy()

    for i in range(3):
        np.testing.assert_equal(f(as_raw_tensor(x)).numpy(), y)

    file = io.BytesIO()
    f.dump(file, optimize_for_inference=False)
    file.seek(0)
    cg, _, outputs = G.load_graph(file)
    (out,) = outputs
    assert (
        cgtools.get_owner_opr_type(cgtools.get_owner_opr_inputs(out)[1])
        == "ImmutableTensor"
    )


def test_trace_profiler():
    for symbolic in [False, True]:

        @trace(symbolic=symbolic, profiling=True)
        def f(x):
            op = ops.Elemwise(mode="negate")
            (y,) = apply(op, x)
            return y

        x = as_raw_tensor([1]).numpy()
        y = f.__wrapped__(as_raw_tensor(x)).numpy()

        f(as_raw_tensor(x))
        f(as_raw_tensor(x))  # XXX: has to run twice

        out = f.get_profile()
        assert out.get("profiler")


@pytest.mark.skip(reason="could not disable opt_level")
def test_goptions_log_exp():
    @trace(symbolic=True, opt_level=0, capture_as_const=True)
    def f(x):
        return log(exp(x))

    @trace(symbolic=True, opt_level=1, capture_as_const=True)
    def g(x):
        return log(exp(x))

    f(tensor(1.0))
    _, out = mkstemp()
    f.dump(out, optimize_for_inference=False)
    *_, outputs = G.load_graph(out)
    oprs_1 = cgtools.get_oprs_seq(outputs)

    g(tensor(1.0))
    g.dump(out, optimize_for_inference=False)
    *_, outputs = G.load_graph(out)
    oprs_2 = cgtools.get_oprs_seq(outputs)

    assert len(oprs_1) - len(oprs_2) == 2


@pytest.mark.skip(reason="could not disable opt_level")
def test_goptions_log_sum_exp():
    @trace(symbolic=True, opt_level=0, capture_as_const=True)
    def f(x, y):
        return log(exp(x) + exp(y))

    @trace(symbolic=True, opt_level=1, capture_as_const=True)
    def g(x, y):
        return log(exp(x) + exp(y))

    f(tensor(1.0), tensor(2.0))
    _, out = mkstemp()
    f.dump(out, optimize_for_inference=False)
    *_, outputs = G.load_graph(out)
    oprs_1 = cgtools.get_oprs_seq(outputs)

    g(tensor(1.0), tensor(2.0))
    g.dump(out, optimize_for_inference=False)
    *_, outputs = G.load_graph(out)
    oprs_2 = cgtools.get_oprs_seq(outputs)

    assert len(oprs_1) - len(oprs_2) == 2


def test_optimize_for_inference():
    @trace(symbolic=True, capture_as_const=True)
    def f(x):
        return exp(x)

    _, out = mkstemp()
    f(tensor(5.0))
    f.dump(out, enable_io16xc32=True)

    res = G.load_graph(out)
    computing_input = res.output_vars_list[0].owner.inputs[0]
    assert computing_input.dtype == np.float16


def test_optimize_for_inference_broadcast():
    a = tensor(np.ones(1, dtype=np.float32))

    @trace(capture_as_const=True, tensor_shape=True)
    def f():
        (b,) = apply(ops.Broadcast(), a, tensor([1, 10], dtype=np.int32))
        return b

    f()
    f.dump(io.BytesIO())


def test_trace_cvt_bool():
    set_tensor_shape(True)
    x = tensor([0], dtype=np.int32)

    @trace(symbolic=True)
    def f(x):
        return x.shape[0] == 0

    for i in range(3):
        np.testing.assert_equal(f(x).numpy()[0], False)


def test_trace_reshape():
    for symbolic in [False, True]:
        set_tensor_shape(True)
        x1 = tensor(np.random.randn(2, 10, 10))
        x2 = tensor(np.random.randn(4, 10, 10))
        x3 = tensor(np.random.randn(8, 10, 10))

        @trace(symbolic=symbolic, capture_as_const=True)
        def f(x):
            y = x.reshape(x.shape[0], 100)
            return y

        f(x1)
        f(x2)
        f(x3)


def test_trace_topk():
    x = tensor([5, 2, 7, 1, 0, 3, 2])

    @trace(symbolic=True)
    def f(x):
        y = F.topk(x, 3)
        np.testing.assert_equal(y[0].shape.numpy(), np.array([3,]))
        return y

    for i in range(3):
        f(x)


def test_trace_warp_perspective():
    inp_shape = (1, 1, 4, 4)
    x = tensor(np.arange(16, dtype=np.float32).reshape(inp_shape))
    M_shape = (1, 3, 3)
    M = tensor(
        np.array(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ).reshape(M_shape)
    )

    @trace(symbolic=True)
    def f(x, M):
        out = F.warp_perspective(x, M, (2, 2))
        np.testing.assert_equal(out.shape.numpy(), np.array([1, 1, 2, 2]))
        return out

    for i in range(1):
        f(x, M)


def test_raise_on_trace():
    step_count = 0
    catch_count = 0
    bad_step = 10

    class CatchMe(Exception):
        pass

    a = tensor([1, 2, 3, 4])
    b = tensor([5, 6, 7, 8])
    c = tensor([9, 0, 1, 2])

    @trace
    def add_abc(a, b, c):
        print("Hello")
        ps = a + b
        result = ps + c
        if step_count == bad_step:
            raise CatchMe("catch me")
        return result

    for i in range(100):
        try:
            d = add_abc(a, b, c)
        except CatchMe as e:
            catch_count += 1
        else:
            np.testing.assert_equal(d.numpy(), (a + b + c).numpy())
        step_count += 1

    assert catch_count == 1


def test_trace_broadcast():
    for symbolic in [False, True]:
        set_tensor_shape(True)
        x1 = tensor(np.random.randn(3, 1, 1))
        x2 = tensor(np.random.randn(1, 4, 1))
        x3 = tensor(np.random.randn(1, 1, 5))

        @trace(symbolic=symbolic, capture_as_const=True)
        def f(x):
            y = F.broadcast_to(x, (3, 4, 5))
            return y

        f(x1)
        f(x2)
        f(x3)


def test_trace_nms():
    def make_inputs(n):
        boxes = np.zeros((n, 4))
        boxes[:, :2] = np.random.rand(n, 2) * 100
        boxes[:, 2:] = np.random.rand(n, 2) * 100 + 100

        scores = np.random.rand(n)

        return tensor(boxes), tensor(scores)

    @trace(symbolic=False)
    def f(boxes, scores):
        results = F.nn.nms(boxes, scores=scores, iou_thresh=0.5, max_output=20)
        with exclude_from_trace():
            _ = F.nn.nms(boxes, scores=scores, iou_thresh=0.5)
        return results

    f(*make_inputs(10))
    f(*make_inputs(20))
    f(*make_inputs(30))


def test_trace_valid_broadcast():
    set_tensor_shape(True)
    x1 = tensor(np.random.randn(1, 1))
    x2 = tensor(np.random.randn(1, 2))
    shape = (tensor([2]), tensor([2]))

    @trace(symbolic=False)
    def f(x, shape):
        y = F.broadcast_to(x, shape)
        return y

    f(x1, shape)
    f(x2, shape)
