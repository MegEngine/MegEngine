# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np
import pytest

import megengine.functional as F
import megengine.functional.elemwise as elemwise
from megengine import tensor
from megengine.core.tensor import dtype
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
