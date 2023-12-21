# -*- coding: utf-8 -*-
import math

import numpy as np
import pytest

import megengine as mge
import megengine.autodiff as ad
import megengine.functional as F
import megengine.functional.elemwise as elemwise
from megengine import tensor
from megengine.core.tensor import dtype
from megengine.core.tensor.utils import subgraph_fn
from megengine.functional.elemwise import Elemwise
from megengine.jit import trace


def test_abs():
    np.testing.assert_allclose(
        F.abs(tensor([-3.0, -4.0, -5.0])).numpy(),
        np.abs(np.array([-3.0, -4.0, -5.0], dtype=np.float32)),
    )

    np.testing.assert_allclose(F.abs(-3.0).numpy(), np.abs(np.float32(-3.0)))


def test_elemwise_mode_string():
    for key, mode in vars(Elemwise.Mode).items():
        if isinstance(mode, Elemwise.Mode):
            assert key == mode
            assert Elemwise(mode=key) == Elemwise(mode=mode)


def test_multiply():
    np.testing.assert_allclose(
        F.mul(-3.0, -4.0).numpy(), np.multiply(np.float32(-3.0), np.float32(-4.0))
    )

    np.testing.assert_allclose(
        F.mul(tensor([3.0, 4.0]), 4.0).numpy(),
        np.multiply(np.array([3.0, 4.0], dtype=np.float32), 4.0),
    )

    np.testing.assert_allclose(
        F.mul(4.0, tensor([3.0, 4.0])).numpy(),
        np.multiply(4.0, np.array([3.0, 4.0], dtype=np.float32)),
    )

    np.testing.assert_allclose(
        F.mul(tensor([3.0, 4.0]), tensor([3.0, 4.0])).numpy(),
        np.multiply(
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        ),
    )


def test_multiply_uint16():
    x = np.random.rand(1, 4, 512, 512) * 255
    y_mge = F.mul(mge.tensor(x.astype("uint16")), mge.tensor(2, dtype="uint16")).numpy()
    y_np = x.astype("uint16") * np.array([2], dtype="uint16")
    np.testing.assert_allclose(y_mge, y_np)
    assert y_mge.dtype == np.uint16


def test_div():
    np.testing.assert_allclose(
        F.div(tensor([3.0, 4.0]), 2).numpy(),
        np.divide(np.array([3, 4], dtype=np.float32), 2),
    )

    np.testing.assert_allclose(
        (tensor([3, 4]) / 2).numpy(), np.divide(np.array([3, 4], dtype=np.float32), 2),
    )

    np.testing.assert_allclose(
        F.floor_div(tensor([-5.0, -7.0]), 2).numpy(),
        np.floor_divide(np.array([-5.0, -7.0], dtype=np.float32), 2),
    )

    np.testing.assert_allclose(
        (tensor([-5, -7]) // 2).numpy(),
        np.floor_divide(np.array([-5, -7], dtype=np.int32), 2),
    )

    np.testing.assert_allclose(
        (tensor([[5, 4, 3], [4, 2, 6]]) // [1, 2, 1]).numpy(),
        np.floor_divide(np.array([[5, 4, 3], [4, 2, 6]], dtype=np.int32), [1, 2, 1]),
    )


def test_clamp():
    """Fix an issue when `lower` or `upper` is 0, it will be recognized as `False` and
    `F.clip` will fall into wrong conditions unexpectedly.
    """
    x = np.linspace(-6, 6, dtype="float32")
    np.testing.assert_allclose(
        F.clip(tensor(x) + 3, 0, 6).numpy(), np.clip(x + 3, 0, 6)
    )
    np.testing.assert_allclose(
        F.clip(tensor(x) - 3, -6, 0).numpy(), np.clip(x - 3, -6, 0)
    )


def test_erf():
    def numpy_erf(x):
        shape = x.shape
        flatten_x = x.flatten()
        erf_arr = np.vectorize(math.erf)
        result = erf_arr(flatten_x).reshape(shape)
        return result

    x = np.array([-1.5, 0.0, 1.0, 1.5]).astype("float32")
    y_np = numpy_erf(x)
    y_mge = F.erf(tensor(x)).numpy()
    np.testing.assert_allclose(y_np, y_mge, rtol=1e-5)


def test_isnan():
    for case in [[1, float("nan"), 0]]:
        np.testing.assert_allclose(F.isnan(tensor(case)).numpy(), np.isnan(case))


def test_isinf():
    for case in [[1, float("inf"), 0]]:
        np.testing.assert_allclose(F.isinf(tensor(case)).numpy(), np.isinf(case))


def test_sign():
    for case in [[1, -1, 0]]:
        x = tensor(case)
        np.testing.assert_allclose(F.sign(x).numpy(), np.sign(case).astype(x.dtype))


def test_cosh():
    np.random.seed(42)
    x = np.random.randn(100).astype("float32")
    y_np = np.cosh(x)
    y_mge = F.cosh(tensor(x)).numpy()
    np.testing.assert_allclose(y_np, y_mge, rtol=1e-5)


def test_sinh():
    np.random.seed(42)
    x = np.random.randn(100).astype("float32")
    y_np = np.sinh(x)
    y_mge = F.sinh(tensor(x)).numpy()
    np.testing.assert_allclose(y_np, y_mge, rtol=1e-5)


def test_asinh():
    np.random.seed(42)
    x = np.random.randn(100).astype("float32")
    y_np = np.arcsinh(x)
    y_mge = F.asinh(tensor(x)).numpy()
    np.testing.assert_almost_equal(y_np, y_mge, decimal=5)


def test_acosh():
    x = np.arange(0, 10000).astype("float32") / 100 + 1
    y_np = np.arccosh(x)
    y_mge = F.acosh(tensor(x)).numpy()
    np.testing.assert_almost_equal(y_np, y_mge, decimal=6)


def test_atanh():
    np.random.seed(42)
    x = np.random.rand(100).astype("float32") * 2 - 1
    y_np = np.arctanh(x)
    y_mge = F.atanh(tensor(x)).numpy()
    np.testing.assert_almost_equal(y_np, y_mge, decimal=5)


def test_hswish():
    np.random.seed(42)
    x = np.random.randn(100).astype("float32")
    y_np = x * np.minimum(np.maximum(x + 3, 0), 6) / 6
    y_mge = F.hswish(tensor(x)).numpy()
    np.testing.assert_almost_equal(y_np, y_mge, decimal=6)


def test_silu():
    x = np.array([-1.5, 0.0, 1.0, 1.5]).astype("float32")
    y_np = x / (1 + np.exp(-x))
    y_mge = F.silu(tensor(x)).numpy()
    np.testing.assert_almost_equal(y_np, y_mge, decimal=6)


def test_hsigmoid():
    np.random.seed(42)
    x = np.random.randn(100).astype("float32")
    y_np = np.minimum(np.maximum(x + 3, 0), 6) / 6
    y_mge = F.hsigmoid(tensor(x)).numpy()
    np.testing.assert_almost_equal(y_np, y_mge, decimal=6)


def test_logical_oprs():
    x = np.array([[True, False], [False, True]])
    y = np.array([[True, True], [False, False]])
    xx = tensor(x)
    yy = tensor(y)
    np.testing.assert_equal(~x, (F.logical_not(xx)).numpy())
    np.testing.assert_equal(x & y, F.logical_and(xx, yy).numpy())
    np.testing.assert_equal(x | y, F.logical_or(xx, yy).numpy())
    np.testing.assert_equal(x ^ y, F.logical_xor(xx, yy).numpy())

    x = np.random.randint(0, 2, (1, 32), dtype="bool")
    y = np.random.randint(0, 2, (1, 31), dtype="bool")
    u = x[:, 1:]
    xx = tensor(x)
    yy = tensor(y)
    uu = xx[:, 1:]
    np.testing.assert_equal(u ^ y, F.logical_xor(uu, yy).numpy())
    np.testing.assert_equal(y ^ u, F.logical_xor(yy, uu).numpy())


def test_logaddexp():
    x = np.random.randn(2, 100)
    y = np.random.randn(2, 100)
    xx = tensor(x)
    yy = tensor(y)
    out_np = np.log(np.exp(x) + np.exp(y))
    out_mge = F.logaddexp(xx, yy)
    np.testing.assert_almost_equal(out_np, out_mge.numpy(), decimal=6)


def test_qadd():
    inp_scale = 0.5
    outp_scale = 0.2
    x = np.arange(6).reshape(2, 3).astype("float32")
    y = np.arange(6).reshape(2, 3).astype("float32")
    x = tensor(x, dtype=dtype.qint8(inp_scale))
    y = tensor(y, dtype=dtype.qint8(inp_scale))
    result_mge = F.elemwise._elemwise_multi_type(
        x, y, mode="qadd", dtype=dtype.qint8(outp_scale)
    )
    result_mge = result_mge.astype("float32").numpy()
    result_expect = x.astype("float32").numpy() + y.astype("float32").numpy()
    np.testing.assert_almost_equal(result_mge, result_expect, decimal=6)


def test_int32_input():
    x = tensor(np.array([1, 2, 3, 4, 5]), dtype="int32")
    for op_name in elemwise.__all__:
        op = getattr(elemwise, op_name)
        nargs = op.__code__.co_argcount
        if op_name == "clip":
            inp = (x, 0, 1)
        elif op_name == "erf":
            continue
        elif op_name.endswith("_shift"):
            inp = (x, 1)
        elif op_name.startswith("logical_"):
            continue
        else:
            inp = (x,) * nargs
        y = op(*inp)
        y.numpy()


@pytest.mark.parametrize("is_trace", [True, False])
def test_empty_tensor(is_trace):
    binary_func = []
    unary_func = []
    for op_name in elemwise.__all__:
        op = getattr(elemwise, op_name)
        nargs = op.__code__.co_argcount
        if op_name == "clip":
            unary_func.append(["clip", lambda x, f=op: f(x, lower=0, upper=1)])
        elif op_name.endswith("_shift"):
            unary_func.append(
                [op_name, lambda x, f=op: f(tensor(x.numpy(), dtype="int32"), 1)]
            )
        elif op_name.startswith("logical_"):  # logical_xxx op only accept boolean type
            if nargs == 1:
                unary_func.append(
                    [op_name, lambda x, f=op: f(tensor(x.numpy(), dtype="bool"))]
                )
            else:
                assert nargs == 2
                binary_func.append(
                    [
                        op_name,
                        lambda x, y, f=op: f(
                            tensor(x.numpy(), dtype="bool"),
                            tensor(y.numpy(), dtype="bool"),
                        ),
                    ]
                )
        elif nargs == 1:
            unary_func.append([op_name, op])
        elif nargs == 2:
            binary_func.append([op_name, op])
        else:
            raise NotImplementedError("nargs {}".format(nargs))

    def run_test(func, args, ref_shape, is_trace, sym=False):
        args = [tensor(t, dtype="float32") for t in args]
        if is_trace:
            func = trace(symbolic=sym)(func)
            for _ in range(3):
                out = func(*args)
                assert out.numpy().shape == ref_shape
        else:
            out = func(*args)
            assert out.numpy().shape == ref_shape, out.numpy().shape

    inps = [
        np.array([]).astype("float32"),
        np.random.randn(2, 0, 3).astype("float32"),
        123,
    ]
    for op_name, op in unary_func:
        if is_trace:
            for sym in [True, False]:
                run_test(op, [inps[0],], inps[0].shape, True, sym)
                run_test(op, [inps[1],], inps[1].shape, True, sym)
        else:
            run_test(op, [inps[0],], inps[0].shape, False)
            run_test(op, [inps[1],], inps[1].shape, False)

    for op_name, op in binary_func:
        if is_trace:
            for sym in [True, False]:
                run_test(op, [inps[0], inps[0]], (inps[0] + inps[0]).shape, True, sym)
                run_test(op, [inps[1], inps[1]], (inps[1] + inps[1]).shape, True, sym)
                run_test(op, [inps[0], inps[2]], (inps[0] + inps[2]).shape, True, sym)
                run_test(op, [inps[1], inps[2]], (inps[1] + inps[2]).shape, True, sym)
        else:
            run_test(op, [inps[0], inps[0]], (inps[0] + inps[0]).shape, False)
            run_test(op, [inps[1], inps[1]], (inps[1] + inps[1]).shape, False)
            run_test(op, [inps[0], inps[2]], (inps[0] + inps[2]).shape, False)
            run_test(op, [inps[1], inps[2]], (inps[1] + inps[2]).shape, False)


@pytest.mark.parametrize("is_trace", [True, False])
def test_maximum_grad_consistency(is_trace):
    def f(x):
        with ad.GradManager() as gm:
            gm.attach(x)
            gm.backward(F.maximum(x, x))
        dx = x.grad
        x.grad = None
        return dx

    def run(f):
        x = F.arange(10)
        for i in range(3):
            np.testing.assert_equal(f(x).numpy(), np.ones(10))

    if is_trace:
        for symbolic in [False, True]:
            run(trace(symbolic=symbolic)(f))
    else:
        run(f)


def _get_logsigmoid_op(dtype=None, device=None):
    @subgraph_fn(
        "LogSigmoid",
        dtype=dtype,
        device=device,
        nr_inputs=1,
        jit_fusion=False,
        custom_grad=True,
    )
    def logsigmoid(inputs, f, c):
        (inp,) = inputs[0:1]
        neg_abs = f("-", f("abs", inp))
        exp = f("exp", neg_abs)
        oup0 = f("log1p", exp)
        oup1 = f("relu", f("-", inp))
        oup = f("+", oup0, oup1)
        oup = f("-", oup)
        (oup_grad,) = yield (oup,)
        oup_grad = f("-", oup_grad)
        inp_grad_0 = f("switch_gt0", oup1, oup_grad)
        inp_grad_0 = f("-", inp_grad_0)
        inp_grad_1 = oup_grad
        inp_grad_1 = f("/", inp_grad_1, f("+", exp, c(1)))
        inp_grad_1 = f("*", inp_grad_1, exp)
        inp_grad_1 = f("-", inp_grad_1)
        inp_grad_1 = f("abs_grad", inp, inp_grad_1)
        inp_grad = f("+", inp_grad_0, inp_grad_1)
        yield (inp_grad,)

    return logsigmoid


def origin_logsigmoid(inp: mge.tensor) -> mge.tensor:
    logsigmoid = _get_logsigmoid_op(inp.dtype, inp.device)
    (oup,) = logsigmoid(inp)
    return oup


def _get_softplus_op(dtype=None, device=None):
    @subgraph_fn(
        "Softplus",
        dtype=dtype,
        device=device,
        nr_inputs=1,
        jit_fusion=False,
        custom_grad=True,
    )
    def softplus(inputs, f, c):
        (inp,) = inputs[0:1]
        neg_abs = f("-", f("abs", inp))
        exp = f("exp", neg_abs)
        oup0 = f("log1p", exp)
        oup1 = f("relu", inp)
        oup = f("+", oup0, oup1)
        (oup_grad,) = yield (oup,)
        inp_grad_0 = f("switch_gt0", oup1, oup_grad)
        inp_grad_1 = oup_grad
        inp_grad_1 = f("/", oup_grad, f("+", exp, c(1)))
        inp_grad_1 = f("*", inp_grad_1, exp)
        inp_grad_1 = f("-", inp_grad_1)
        inp_grad_1 = f("abs_grad", inp, inp_grad_1)
        inp_grad = f("+", inp_grad_0, inp_grad_1)
        yield (inp_grad,)

    return softplus


def origin_softplus(inp: mge.tensor) -> mge.tensor:

    softplus = _get_softplus_op(inp.dtype, inp.device)
    (oup,) = softplus(inp)
    return oup


def test_subgraph_elemwise_mode():
    def _test_allclose(func, ori_func):
        targets = np.array(2)
        inp = np.random.uniform(size=(2, 16, 10, 16)).astype(np.float32)
        ori_inp = mge.tensor(inp)
        mge_inp = mge.tensor(inp)

        mge_gm = mge.autodiff.GradManager().attach(mge_inp)
        ori_gm = mge.autodiff.GradManager().attach(ori_inp)

        for _ in range(2):
            with mge_gm:
                mge_output = func(mge_inp)
                loss = F.loss.square_loss(
                    mge_output.sum(), mge.tensor(targets, dtype=np.float32)
                )
                mge_gm.backward(loss)

            with ori_gm:
                ori_output = ori_func(ori_inp)
                loss = F.loss.square_loss(
                    ori_output.sum(), mge.tensor(targets, dtype=np.float32)
                )
                ori_gm.backward(loss)

            np.testing.assert_allclose(
                mge_output.numpy(), ori_output.numpy(), rtol=1e-06
            )
            np.testing.assert_allclose(
                ori_inp.grad.numpy(), mge_inp.grad.numpy(), rtol=1e-06
            )

    _test_allclose(F.logsigmoid, origin_logsigmoid)
    _test_allclose(F.softplus, origin_softplus)
