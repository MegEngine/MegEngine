import functools
import os
import platform
import subprocess
import sys

import numpy as np
import pytest

import megengine
from megengine.autodiff.grad_manager import GradManager
from megengine.core.ops.builtin import GetVarShape, Reduce, TypeCvt
from megengine.core.tensor.utils import subgraph_fn
from megengine.device import CompNode, get_default_device
from megengine.jit import trace

_assert_allclose = functools.partial(np.testing.assert_allclose, atol=5e-6, rtol=5e-6)


@functools.lru_cache(maxsize=None)
def _get_batch_norm_fn(dtype, device, channels, ndim, interpret, gopt_level):
    @subgraph_fn(
        "BatchNormNd",
        dtype=dtype,
        device=device,
        nr_inputs=4,
        interpret=interpret,
        gopt_level=gopt_level,
    )
    def batch_norm_nd(inputs, f, c):
        input, eps, weight, bias = inputs[0:4]
        reduce_shape = c(
            (1, channels) + (1,) * (ndim - 2), dtype="int32", device=device
        )
        input_shape = f(GetVarShape(), input)
        input_elems = f(Reduce(mode="product", axis=0), input_shape)
        reduce_elems = f(Reduce(mode="product", axis=0), reduce_shape)
        reduce_size = f("//", input_elems, reduce_elems)
        reduce_size = f(TypeCvt(dtype=dtype), reduce_size)
        channel_x1s = f(Reduce(mode="sum"), input, reduce_shape)
        channel_x2s = f(Reduce(mode="sum_sqr"), input, reduce_shape)
        channel_mean = f("/", channel_x1s, reduce_size)
        channel_var = f(
            "-", f("/", channel_x2s, reduce_size), f("*", channel_mean, channel_mean),
        )
        invsqrt_channel_var = f("**", f("+", channel_var, eps), c(-0.5))
        inv_var_wt = f("*", invsqrt_channel_var, weight)
        neg_channel_mean = f("-", channel_mean)
        outvar = f(
            "fma3", input, inv_var_wt, f("fma3", neg_channel_mean, inv_var_wt, bias),
        )
        return (outvar,), (True,)

    return batch_norm_nd


@pytest.mark.parametrize("device", [get_default_device(), "cpux"])
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("channels", [3])
@pytest.mark.parametrize(
    "use_trace, symbolic", [(False, None), (True, False), (True, True)]
)
@pytest.mark.parametrize("gopt_level", [None, 1, 2])
@pytest.mark.parametrize("dtype", ["float32"])
def test_subgraph(device, batch_size, channels, use_trace, symbolic, gopt_level, dtype):
    device = CompNode(device)

    def subgraph_batch_norm(inp, weight, bias, eps, diff):
        inp = inp.detach()
        with GradManager().attach(inp) as gm:
            batch_norm_fn = _get_batch_norm_fn(
                dtype, device, channels, ndim, interpret=False, gopt_level=gopt_level
            )
            out, *_ = batch_norm_fn(inp, eps, weight, bias)
            gm.backward(out * 1e3 + 1e3, diff)
            return out, inp.grad

    def primitive_batch_norm(inp, weight, bias, eps, diff):
        inp = inp.detach()
        with GradManager().attach(inp) as gm:
            batch_norm_fn = _get_batch_norm_fn(
                dtype, device, channels, ndim, interpret=True, gopt_level=gopt_level
            )
            (out,) = batch_norm_fn(inp, eps, weight, bias)
            gm.backward(out * 1e3 + 1e3, diff)
            return out, inp.grad

    if use_trace:
        subgraph_batch_norm = trace(symbolic=symbolic)(subgraph_batch_norm)
        primitive_batch_norm = trace(symbolic=symbolic)(primitive_batch_norm)

    def rand_tensor(shape, dtype=dtype, device=device):
        return megengine.tensor(np.random.random(shape), dtype=dtype, device=device)

    # skip this test because could not do several reduce sequentially with opr cache
    return

    # test shape change
    for image_shape in [(223, 223), (10, 20)]:
        ndim = len(image_shape) + 2
        input_shape = (batch_size, channels) + image_shape
        param_shape = (1, channels) + (1,) * len(image_shape)

        inp = rand_tensor(input_shape) * 1e3 + 1e3
        weight = rand_tensor(param_shape)
        bias = rand_tensor(param_shape)
        eps = megengine.tensor(1e-5, dtype=dtype, device=device)

        diff = rand_tensor(input_shape)

        out1, grad1 = subgraph_batch_norm(inp, weight, bias, eps, diff)
        out2, grad2 = primitive_batch_norm(inp, weight, bias, eps, diff)

        _assert_allclose(out1.numpy(), out2.numpy())
        _assert_allclose(grad1.numpy(), grad2.numpy())


@functools.lru_cache(maxsize=None)
def _get_mul_fn(dtype, device):
    @subgraph_fn(
        "Mul",
        dtype=dtype,
        device=device,
        nr_inputs=2,
        gopt_level=None,
        jit_fusion=False,
        custom_grad=True,
    )
    def mul(inputs, f, c):
        x, y = inputs[0:2]
        z = f("*", x, y)
        (dz,) = yield (z,)
        dx = f("*", dz, y)
        dy = f("*", dz, x)
        yield (dx, dy)

    return mul


def test_subgraph_jit_backward():
    x_np = np.random.rand(3, 4, 5).astype("float32")
    x1 = megengine.Tensor(x_np)
    x2 = megengine.Tensor(x_np)
    mul = _get_mul_fn(x1.dtype, x1.device)
    gm = GradManager()
    gm.attach([x1, x2])
    with gm:
        y1 = x1 * x1
        y2 = mul(x2, x2)
        gm.backward(y1)
    with gm:
        y1 = x1 * x1
        y2 = mul(x2, x2)
        gm.backward(y1 + y2)
    with gm:
        y1 = x1 * x1
        y2 = mul(x2, x2)
        gm.backward(y2)


@pytest.mark.skipif(
    platform.system() != "Linux", reason="jit fusion is only available on Linux",
)
def test_subgraph_jit():
    prog = """
import megengine
import numpy as np
from megengine.core.tensor.utils import subgraph_fn

# 3 * 4 * 5 > MEGDNN_MAX_NDIM
x_np = np.random.rand(3, 4, 5).astype("float32")
x1 = megengine.Tensor(x_np)
x2 = megengine.Tensor(x_np)

@subgraph_fn(
    "Mul",
    dtype=x1.dtype,
    device=x1.device,
    nr_inputs=2,
    gopt_level=None,
    jit_fusion=True,
    custom_grad=True,
)
def mul(inputs, f, c):
    x, y = inputs[0:2]
    z = f("*", x, y)
    (dz,) = yield (z,)
    dx = f("*", dz, y)
    dy = f("*", dz, x)
    yield (dx, dy)

y, = mul(x1, x2)

# ensure execution
y.numpy()
"""
    env = dict(os.environ)
    if "PATH" in env:
        # remove nvcc from environ["PATH"]
        path = env["PATH"]
        paths = path.split(os.pathsep)
        paths = [
            path
            for path in paths
            if not (os.path.isdir(path) and "nvcc" in os.listdir(path))
        ]
        path = os.pathsep.join(paths)
        env["PATH"] = path
    # previous program may be stored in persistent cache
    env["MGE_FASTRUN_CACHE_TYPE"] = "MEMORY"
    subprocess.check_call([sys.executable, "-c", prog], env=env)
