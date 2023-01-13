# -*- coding: utf-8 -*-
import inspect
import io
import itertools
import random
from tempfile import mkstemp

import numpy as np
import pytest

import megengine.core.tensor.megbrain_graph as G
import megengine.functional as F
import megengine.optimizer as optim
import megengine.utils.comp_graph_tools as cgtools
from megengine import Parameter, tensor
from megengine.autodiff import GradManager
from megengine.core.ops import builtin as ops
from megengine.core.ops.builtin import Elemwise
from megengine.core.tensor.utils import isscalar
from megengine.functional import exp, log
from megengine.jit import GraphOptimizationConfig, TraceError, exclude_from_trace, trace
from megengine.module import Module
from megengine.random import normal, uniform
from megengine.utils.naming import AutoNaming


@pytest.mark.parametrize("trace_mode", [False, True])
@pytest.mark.parametrize("return_mode", ["Value", "Tuple", "List", "Dict"])
def test_trace(trace_mode, return_mode):
    @trace(symbolic=trace_mode)
    def f(x):
        if return_mode == "Tuple":
            return (-x,)
        elif return_mode == "List":
            return [-x]
        elif return_mode == "Dict":
            return {"neg": -x}
        else:
            return -x

    def get_numpy(y):
        if return_mode == "Tuple" or return_mode == "List":
            return y[0].numpy()
        elif return_mode == "Dict":
            return y["neg"].numpy()
        return y.numpy()

    x = tensor([1])
    y = get_numpy(f(x))

    for i in range(3):
        np.testing.assert_equal(get_numpy(f(x)), y)


def test_output_copy_trace():
    class Simple(Module):
        def __init__(self):
            super().__init__()
            self.a = Parameter([1.0], dtype=np.float32)

        def forward(self, x):
            x = x * self.a
            # will result into a copy of output in grad
            x = F.exp(x)
            return x

    ys = {False: [], True: []}

    for symbolic in [False, True]:
        net = Simple()
        gm = GradManager().attach(net.parameters())
        opt = optim.SGD(net.parameters(), 1e-3, momentum=0.9)
        data = tensor(np.arange(4).reshape(2, 2), dtype="float32")

        @trace(symbolic=symbolic)
        def train_func(d):
            with gm:
                loss = net(d)
                gm.backward(loss)
                opt.step().clear_grad()
            return loss

        for i in range(3):
            y = train_func(data).numpy()
            ys[symbolic].append(y)

    for i in range(3):
        np.testing.assert_equal(ys[False][i], ys[True][i])


@pytest.mark.parametrize("trace_mode", [False, True])
def test_tensor_detach(trace_mode):
    @trace(symbolic=True)
    def f(x):
        y = x.detach() ** 2
        z = y.detach() + 1
        return z.detach()

    x = tensor([1, 2, 3, 4])
    for _ in range(3):
        f(x).numpy()


@pytest.mark.parametrize("trace_mode", [False, True])
def test_exclude_from_trace(trace_mode):
    @trace(symbolic=trace_mode)
    def f(x):
        x = -x
        with exclude_from_trace():
            if i % 2:
                x = -x
        x = -x
        return x

    x = tensor([1])

    for i in range(3):
        y = f(x).numpy()
        np.testing.assert_equal(f(x).numpy(), y)


@pytest.mark.parametrize("trace_mode", [False, True])
def test_elemwise_fuse(trace_mode):
    # explicitly declare opt_level as 2
    @trace(symbolic=trace_mode, opt_level=2)
    def f(a, b):
        base = 0
        c = b - a
        _, idx = F.topk(c, 3)
        # internally, biased_idx will be idx as gopt will ignore the addition
        biased_idx = base + idx
        return biased_idx

    a = tensor(np.ones((7, 2)), dtype=np.int32)
    b = tensor(2 * np.ones((7, 2)), dtype=np.float32)

    for i in range(3):
        y = f(a, b)
        y.numpy()


@pytest.mark.parametrize("trace_mode", [False, True])
def test_elemwise_fuse_in_grad(trace_mode):
    w = Parameter(np.ones([4, 6]), dtype="float32")

    gm = GradManager().attach(w)
    opt = optim.SGD([w], lr=0.01, momentum=0.9, weight_decay=5e-4)

    # explicitly declare opt_level as 2
    @trace(symbolic=trace_mode, opt_level=2)
    def f():
        with gm:
            wm = F.sum(w ** 2, axis=1) ** 0.5
            loss = wm.mean()
            gm.backward(loss)
            opt.step().clear_grad()
        return loss

    for i in range(3):
        y = f()
        y.numpy()


def test_repeat_in_trace():
    @trace(symbolic=False)
    def fun(data, repeats):
        F.repeat(data, repeats)

    data = tensor(np.random.random([1, 2, 3]).astype(np.float32))

    for i in range(1, 5):
        repeats = tensor(i)
        fun(data, repeats)


def test_print_in_trace():
    for symbolic in [False]:  # cannot read value in symbolic mode

        @trace(symbolic=symbolic)
        def f(x):
            nonlocal buf
            x = -x
            buf = x.numpy()
            x = -x
            return x

        buf = None
        x = tensor([1])

        for i in range(3):
            y = f(x).numpy()
            z = buf
            buf = None
            np.testing.assert_equal(f(x).numpy(), y)
            np.testing.assert_equal(z, buf)


@pytest.mark.parametrize(
    "dump_format",
    [
        "FBS",
    ],
)
def test_dump(dump_format):
    @trace(symbolic=True, capture_as_const=True)
    def f(a, b):
        return a + b

    # prevent from remaining scope from exception test
    AutoNaming.clear()
    a = tensor([2])
    b = tensor([4])
    y = f(a, b).numpy()

    for i in range(3):
        np.testing.assert_equal(f(a, b).numpy(), y)

    file = io.BytesIO()
    dump_info = f.dump(file, dump_format=dump_format)
    assert dump_info.nr_opr == 3
    np.testing.assert_equal(dump_info.inputs, ["arg_0", "arg_1"])
    np.testing.assert_equal(dump_info.outputs, ["ADD"])
    file.seek(0)
    infer_cg = cgtools.GraphInference(file)
    result = list((infer_cg.run(a, b)).values())[0]
    np.testing.assert_equal(result[0], y)


def test_capture_dump():
    a = tensor([2])

    @trace(symbolic=True, capture_as_const=True)
    def f(x):
        return x * a

    x = tensor([3])
    y = f(x).numpy()

    for i in range(3):
        np.testing.assert_equal(f(x).numpy(), y)

    file = io.BytesIO()
    f.dump(file)
    file.seek(0)
    infer_cg = cgtools.GraphInference(file)
    result = list((infer_cg.run(x)).values())[0]
    np.testing.assert_equal(result[0], y)


def test_dump_volatile():
    p = tensor([2])

    @trace(symbolic=True, capture_as_const=True)
    def f(x):
        return x * p

    x = tensor([3])
    y = f(x).numpy()

    for i in range(3):
        np.testing.assert_equal(f(x).numpy(), y)

    file = io.BytesIO()
    f.dump(file, optimize_for_inference=False)
    file.seek(0)
    (out,) = G.load_graph(file).output_vars_list
    assert (
        cgtools.get_owner_opr_type(cgtools.get_owner_opr_inputs(out)[1])
        == "ImmutableTensor"
    )


def test_dump_backward_graph():
    x0 = tensor(np.random.randn(3, 4))
    x1 = tensor(np.random.randn(3, 4))

    gm = GradManager().attach(x0)

    @trace(symbolic=True, capture_as_const=True)
    def f(x0, x1):
        with gm:
            y = x0 * x1
            gm.backward(y, F.ones_like(y))
            dx0 = x0.grad
        return y, dx0

    y, dx0 = f(x0, x1)
    np.testing.assert_equal(dx0.numpy(), x1)

    file = io.BytesIO()
    f.dump(file, optimize_for_inference=False)
    file.seek(0)

    infer_cg = cgtools.GraphInference(file)
    results = list((infer_cg.run(x0, x1)).values())

    np.testing.assert_equal(results[0], y)
    np.testing.assert_equal(results[1], dx0)


def test_dump_with_testcase():
    @trace(symbolic=True, capture_as_const=True)
    def f(x):
        return exp(x)

    f(tensor(1.0))
    file = io.BytesIO()
    f.dump(file, input_data=["#rand(0, 255, 1)"])


def test_split_dump():
    class SimpleNet(Module):
        def __init__(self, num_segments: int = 3):
            super().__init__()
            self.num_segments = num_segments

        def forward(self, x):
            x = F.split(x, self.num_segments, axis=1)
            return x

    model = SimpleNet()
    model.eval()
    data = tensor(np.random.random((1, 12, 224, 224)))

    @trace(symbolic=True, capture_as_const=True)
    def fun(data, *, net):
        return net(data)

    x = fun(data, net=model)
    fun.dump(io.BytesIO(), arg_names=["data"])


@pytest.mark.parametrize("trace_mode", [False, True])
def test_trace_profiler(trace_mode):
    @trace(symbolic=trace_mode, profiling=True)
    def f(x):
        return -x

    x = tensor([1])
    y = f(x).numpy()

    f(x)
    f(x)  # XXX: has to run twice

    out = f.get_profile()
    assert out.get("profiler")


def test_goptions():
    @trace(symbolic=True, opt_level=0, capture_as_const=True)
    def f(x):
        # directly return x / x will not trigger gopt
        # since there's no way to tell the two x are the same
        y = 2.0 * x
        return y / y

    @trace(symbolic=True, opt_level=1, capture_as_const=True)
    def g(x):
        y = 2.0 * x
        return y / y

    d = tensor(0.0)
    assert not np.isfinite(f(d).numpy())
    np.testing.assert_equal(g(d).numpy().item(), 1.0)


def test_goptions_log_sum_exp():
    @trace(symbolic=True, opt_level=0, capture_as_const=True)
    def f(x, y):
        return log(exp(x) + exp(y))

    @trace(symbolic=True, opt_level=1, capture_as_const=True)
    def g(x, y):
        return log(exp(x) + exp(y))

    val = 1.0e4
    d = tensor(val)
    o = tensor(0.0)
    assert not np.isfinite(f(d, o).numpy())
    np.testing.assert_almost_equal(g(d, o), val)


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
    outputs = G.load_graph(out).output_vars_list
    oprs_1 = cgtools.get_oprs_seq(outputs)

    g(tensor(1.0))
    g.dump(out, optimize_for_inference=False)
    outputs = G.load_graph(out).output_vars_list
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

    @trace(capture_as_const=True, symbolic_shape=True)
    def f():
        return a._broadcast(tensor([1, 10], dtype=np.int32))

    f()
    f.dump(io.BytesIO())


def test_trace_cvt_bool():
    x = tensor([0], dtype=np.int32)

    @trace(symbolic=True)
    def f(x):
        a = x.shape
        b = a[0]
        assert isscalar(b)
        return b == 0

    for i in range(3):
        np.testing.assert_equal(f(x).numpy(), False)


@pytest.mark.parametrize("trace_mode", [False, True])
def test_trace_reshape(trace_mode):
    x1 = tensor(np.random.randn(2, 10, 10))
    x2 = tensor(np.random.randn(4, 10, 10))
    x3 = tensor(np.random.randn(8, 10, 10))

    @trace(symbolic=trace_mode, capture_as_const=True)
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
        out = F.vision.warp_perspective(x, M, (2, 2))
        np.testing.assert_equal(out.shape.numpy(), np.array([1, 1, 2, 2]))
        return out

    for i in range(3):
        f(x, M)


@pytest.mark.parametrize(
    "normal_expr, mismatch_expr, reason",
    [
        ("a + b + c", "a + b - c", "operator mismatch"),
        ("a + b + 1", "a + b + 2", "tensors not equals"),
        ("((a + b), (b + c))[0]", "a + b", "mismature end"),
        ("a + b + c", "c + (a + b)", "expect internal node, got external"),
        ("c + (a + b)", "a + b + c", "expect external node, got internal"),
        ("a + b + c", "a + b + c + c", "too many instructions"),
        ("((a + b), (b + c))[1]", "((a + b), (b + c))[0]", "data unreadable"),
        ("((a + b), (b + c))[1] + a", "((a + b), (b + c))[0] + a", "input id mismatch"),
    ],
)
def test_trace_mismatch(normal_expr, mismatch_expr, reason):
    a = tensor([1, 2, 3, 4])
    b = tensor([5, 6, 7, 8])
    c = tensor([9, 0, 1, 2])

    mismatch = False

    @trace(symbolic=True)
    def fn(a, b, c):
        if not mismatch:
            result = eval(normal_expr)
        else:
            result = eval(mismatch_expr)
        return result

    for i in range(20):
        try:
            d = fn(a, b, c)
        except TraceError as e:
            assert mismatch
            assert str(e) == "trace error because {}".format(reason)
        except:
            pytest.fail("unexpected trace error")
        else:
            assert not mismatch
            np.testing.assert_equal(d.numpy(), eval(normal_expr).numpy())
        mismatch = random.random() > 0.8


def test_exception_in_trace():
    a = tensor([1, 2, 3, 4])
    b = tensor([5, 6, 7, 8])
    c = tensor([9, 0, 1, 2])

    mismatch = False

    exc = Exception()

    @trace(symbolic=True)
    def fn(a, b, c):
        result = a + b
        if not mismatch:
            result += c
        else:
            raise exc
        return result

    for i in range(20):
        try:
            d = fn(a, b, c)
        except TraceError as e:
            pytest.fail("unexpected trace error")
        except Exception as e:
            assert mismatch
            assert e is exc
        else:
            assert not mismatch
            np.testing.assert_equal(d.numpy(), (a + b + c).numpy())
        mismatch = random.random() > 0.8


def test_graph_error():
    a = tensor(np.arange(8).reshape((2, 4)))
    b = tensor(np.arange(8).reshape((2, 4)))

    @trace(symbolic=True)
    def fn(a, b):
        return a + b

    fn(a, b)
    with pytest.raises(RuntimeError):
        fn(a, b.transpose())
    fn(a, b)


@pytest.mark.parametrize("trace_mode", [False, True])
def test_trace_broadcast(trace_mode):
    x1 = tensor(np.random.randn(3, 1, 1))
    x2 = tensor(np.random.randn(1, 4, 1))
    x3 = tensor(np.random.randn(1, 1, 5))

    @trace(symbolic=trace_mode, capture_as_const=True)
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
        # with tracing, max_output must be specified
        results = F.vision.nms(boxes, scores=scores, iou_thresh=0.5, max_output=20)
        # without tracing, max output can be inferred inside nms
        with exclude_from_trace():
            _ = F.vision.nms(boxes, scores=scores, iou_thresh=0.5)
        return results

    f(*make_inputs(10))
    f(*make_inputs(20))
    f(*make_inputs(30))


def test_trace_valid_broadcast():
    x1 = tensor(np.random.randn(1, 1))
    x2 = tensor(np.random.randn(1, 2))
    shape = (tensor([2]), tensor([2]))

    @trace(symbolic=False)
    def f(x, shape):
        y = F.broadcast_to(x, shape)
        return y

    f(x1, shape)
    f(x2, shape)


@pytest.mark.parametrize("trace_mode", [False, True])
def test_clip(trace_mode):
    x = tensor(np.random.randn(10, 10))

    @trace(symbolic=trace_mode)
    def f(x, lower, upper):
        y = F.clip(x, lower, upper)
        return y

    for i in range(3):
        f(x, tensor([0]), tensor([1]))

    for i in range(3):
        f(x, tensor([5]), tensor([4]))


# test returning noncontiguous tensor from trace
def test_slice():
    @trace
    def f(x):
        return x[:, 1::2]

    x = F.arange(8).reshape(2, 4)
    f(x)
    y = f(x)
    np.testing.assert_array_equal(y.numpy(), x.numpy()[:, 1::2])
    y + y


@pytest.mark.parametrize("shape_mode", [False, True])
def test_random(shape_mode):
    def run_test(op):
        @trace(symbolic=True, symbolic_shape=shape_mode)
        def f():
            out = op(size=[10, 10])
            out_shape = out.shape
            assert out_shape is not None
            if not isinstance(out_shape, tuple):
                assert out.shape.numpy() is not None
            return out

        for _ in range(3):
            f()

    run_test(uniform)
    run_test(normal)


@pytest.mark.parametrize("shape_mode", [False, True])
def test_trace_advance_indexing(shape_mode):
    funcs = [
        lambda x, i: x[i],
        lambda x, i, j: x[i, j],
        lambda x, i, j: x[i, :, j, ...],
        lambda x, start, end: x[start:end],
        lambda x, start, end: x[:, 0, start:end, ..., 1],
        lambda x, vec: x[vec],
        lambda x, vec: x[vec, ..., 0, 1:3],
        lambda x, vec: x[vec, vec[0], vec[1]],
        # lambda x, i, start, end, vec: x[i, ..., :, vec, start:end],  # FIXME
        lambda x, mask: x[mask],
    ]

    inputs = {
        "x": np.random.randn(5, 5, 5, 5, 5).astype("float32"),
        "i": 4,
        "j": 2,
        "start": 1,
        "end": 3,
        "vec": [1, 2, 3],
        "mask": np.random.randn(5, 5, 5, 5, 5) >= 0,
    }
    for f in funcs:
        sig = inspect.signature(f)
        param_names = list(sig._parameters.keys())
        params = {}
        params_np = {}
        f_traced = trace(f, symbolic=False, symbolic_shape=shape_mode)
        for name in param_names:
            params[name] = tensor(inputs[name])
            params_np[name] = inputs[name]
        expected = f(**params_np)
        result_imperative = f(**params)
        np.testing.assert_equal(expected, result_imperative.numpy())
        for _ in range(3):
            result_trace = f_traced(**params)
            np.testing.assert_equal(expected, result_trace.numpy())


@pytest.mark.require_ngpu(1)  # nvrtc backend
def test_trace_jit_config():
    def run(fuse_dimshuffle, fuse_reduce):
        config = GraphOptimizationConfig()
        config.jit_fuse_dimshuffle = fuse_dimshuffle
        config.jit_fuse_reduce = fuse_reduce

        # set opt_level = 1 to avoid fusing dimshuffle and reduce at the same time
        @trace(opt_level=1, graph_opt_config=config)
        def func(x):
            return x + 1

        x = tensor(2)
        y = func(x)
        y = func(x)
        # func._compile()

        options = func._trace.options
        mapping = {None: 0, False: 1, True: 2}
        assert options.graph_opt.jit == 0
        assert options.graph_opt.jit_config.fuse_dimshuffle == mapping[fuse_dimshuffle]
        assert options.graph_opt.jit_config.fuse_reduce == mapping[fuse_reduce]

    for fuse_dimshuffle in [None, False, True]:
        for fuse_reduce in [None, False, True]:
            run(fuse_dimshuffle, fuse_reduce)
