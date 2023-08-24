import os
import platform
import shutil
import sys

import numpy as np
import pytest

import megengine.functional as F
from megengine import jit
from megengine.autodiff import Function, GradManager
from megengine.core._imperative_rt.core2 import apply
from megengine.core.ops import custom
from megengine.device import get_device_count
from megengine.tensor import Tensor
from megengine.utils import custom_op_tools

build_path = os.path.join(
    custom_op_tools._get_default_build_root(), "custom_opsrc", "build"
)
cur_dir_path = os.path.dirname(os.path.abspath(__file__))
mgb_root_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(cur_dir_path))))
)
extra_include_paths = [os.path.join(mgb_root_path, "src", "custom", "include")]

extra_ld_flags = []
if sys.platform != "win32":
    ld_path = os.environ.get("LD_LIBRARY_PATH")
    if ld_path != None:
        ld_dirs = ld_path.split(":")
        for ld_dir in ld_dirs:
            if os.path.exists(ld_dir) and os.path.isdir(ld_dir):
                for lib in os.listdir(ld_dir):
                    if "megengine_shared" in lib:
                        extra_ld_flags += ["-L{} -Wl,-rpath,{}".format(ld_dir, ld_dir)]
                        break


def build_and_clean(*srcs):
    def deco(test_func):
        custom_op_srcs = [os.path.join(cur_dir_path, "custom_opsrc", s) for s in srcs]

        def wrapper(*args, **kwargs):
            lib_path = custom_op_tools.build_and_load(
                "test_op",
                custom_op_srcs,
                extra_include_paths=extra_include_paths,
                build_dir=build_path,
                extra_ldflags=extra_ld_flags,
                verbose=True,
            )
            test_func(*args, **kwargs)
            custom.unload(lib_path)

        return wrapper

    return deco


@pytest.mark.skipif(
    get_device_count("gpu") > 0, reason="elem_add operator is only supported on CPU"
)
@build_and_clean("elem_add.cpp")
def test_cpu_func():
    class ElemAddSmooth(Function):
        def __init__(self, smooth):
            super().__init__()
            self.smooth = smooth

        def forward(self, lhs, rhs):
            op = custom.ElemAddSmoothForward(smooth=self.smooth)
            return apply(op, lhs, rhs)[0]

        def backward(self, ograd):
            op = custom.ElemAddSmoothBackward()
            return apply(op, ograd)

    def gen_elemadd_data(seed, shape, low=-1, high=1):
        rng = np.random.RandomState(seed=seed)
        lhs_np = rng.uniform(low=low, high=high, size=shape).astype(np.float32)
        rhs_np = rng.uniform(low=low, high=high, size=shape).astype(np.float32)
        ograd_np = rng.uniform(low=low, high=high, size=shape).astype(np.float32)
        return lhs_np, rhs_np, ograd_np

    def builtin_func(lhs, rhs, smooth):
        out = lhs + rhs
        return F.where(out < 0, out + smooth, out - smooth)

    def test_elemadd_smooth_train(smooth=0.5, m=4, n=2, seed=2021):
        lhs_np, rhs_np, ograd_np = gen_elemadd_data(seed, (m, n))
        custom_lhs, custom_rhs = Tensor(lhs_np), Tensor(rhs_np)
        builtin_lhs, builtin_rhs = Tensor(lhs_np), Tensor(rhs_np)
        ograd_tensor = Tensor(ograd_np)

        custom_func = ElemAddSmooth(smooth=smooth)
        gm = GradManager().attach([custom_lhs, custom_rhs])
        with gm:
            custom_out = custom_func(custom_lhs, custom_rhs)
            gm.backward(custom_out, ograd_tensor)

        gm = GradManager().attach([builtin_lhs, builtin_rhs])
        with gm:
            builtin_out = builtin_func(builtin_lhs, builtin_rhs, smooth)
            gm.backward(builtin_out, ograd_tensor)

        np.testing.assert_allclose(custom_out, builtin_out, rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(
            custom_lhs.grad.numpy(), builtin_lhs.grad.numpy(), rtol=1e-3, atol=1e-5
        )
        np.testing.assert_allclose(
            custom_rhs.grad.numpy(), builtin_rhs.grad.numpy(), rtol=1e-3, atol=1e-5
        )

    def test_elemadd_smooth_trace(smooth=0.5, m=4, n=2, seed=2021):
        @jit.trace(capture_as_const=True)
        def func_dumper(lhs, rhs, *, net):
            return net(lhs, rhs)

        lhs_np, rhs_np, _ = gen_elemadd_data(seed, (m, n))
        lhs_tensor = Tensor(lhs_np)
        rhs_tensor = Tensor(rhs_np)
        func = ElemAddSmooth(smooth=smooth)
        real = func_dumper(lhs_tensor, rhs_tensor, net=func)
        real = func_dumper(lhs_tensor, rhs_tensor, net=func)

        ref = builtin_func(Tensor(lhs_np), Tensor(rhs_np), smooth)
        np.testing.assert_allclose(real.numpy(), ref.numpy(), rtol=1e-3, atol=1e-5)

    test_elemadd_smooth_train(0.2, 128, 256, 2027)
    test_elemadd_smooth_train(0.3, 256, 128, 2028)
    test_elemadd_smooth_train(0.4, 128, 512, 2029)

    test_elemadd_smooth_trace(0.2, 256, 64, 2030)


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="GPU kernel is only support on Linux and Windows",
)
@pytest.mark.skipif(
    get_device_count("gpu") < 1, reason="matmul scale operator is only supported on GPU"
)
@build_and_clean("matmul_scale.cpp", "matmul_scale.cu")
def test_gpu_func():
    class MatMulScale(Function):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

        def forward(self, lhs, rhs):
            op = custom.MatMulScaleForward(scale=self.scale)
            self.lhs = lhs
            self.rhs = rhs
            return apply(op, lhs, rhs)[0]

        def backward(self, ograd):
            op = custom.MatMulScaleBackward(scale=self.scale)
            return apply(op, ograd, self.lhs, self.rhs)

    def gen_matmul_data(seed, m, k, n, low=-0.5, high=0.5, dtype=np.float32):
        rng = np.random.RandomState(seed=seed)
        lhs_np = rng.uniform(low=low, high=high, size=(m, k)).astype(dtype)
        rhs_np = rng.uniform(low=low, high=high, size=(k, n)).astype(dtype)
        ograd_np = rng.uniform(low=low, high=high, size=(m, n)).astype(dtype)
        scale = rng.uniform(low=0.1, high=0.9, size=(1)).astype(np.float32)[0]

        return lhs_np, rhs_np, ograd_np, scale

    def builtin_func(lhs, rhs, scale):
        out = F.matmul(lhs, rhs) * scale
        return out

    def test_matmul_scale(m=1, k=1, n=1, seed=2021):
        lhs_np, rhs_np, _, scale = gen_matmul_data(seed, m, k, n)
        custom_lhs, custom_rhs = Tensor(lhs_np), Tensor(rhs_np)
        builtin_lhs, builtin_rhs = Tensor(lhs_np), Tensor(rhs_np)

        custom_func = MatMulScale(scale=scale)
        custom_out = custom_func(custom_lhs, custom_rhs)
        builtin_out = builtin_func(builtin_lhs, builtin_rhs, scale)

        np.testing.assert_allclose(custom_out, builtin_out, rtol=1e-3, atol=1e-5)

    def test_matmul_scale_trace(m=1, k=1, n=1, seed=2021):
        @jit.trace(capture_as_const=True)
        def func_dumper(lhs, rhs, *, net):
            return net(lhs, rhs)

        lhs_np, rhs_np, _, scale = gen_matmul_data(seed, m, k, n)
        lhs_tensor, rhs_tensor = Tensor(lhs_np), Tensor(rhs_np)
        func = MatMulScale(scale=scale)
        real = func_dumper(lhs_tensor, rhs_tensor, net=func)
        real = func_dumper(lhs_tensor, rhs_tensor, net=func)

        ref = builtin_func(Tensor(lhs_np), Tensor(rhs_np), scale)
        np.testing.assert_allclose(real.numpy(), ref.numpy(), rtol=1e-3, atol=1e-5)

    test_matmul_scale(128, 256, 64, 2028)
    test_matmul_scale(64, 32, 16, 2029)

    test_matmul_scale_trace(64, 32, 16, 2030)


@pytest.mark.skipif(
    get_device_count("gpu") < 1, reason="matmul scale operator is only supported on GPU"
)
def test_custom_op():
    org_op_list = custom._get_custom_op_list()
    assert len(custom._get_custom_op_lib_info()) == 0

    assert "ElemAddSmoothForward" not in custom._get_custom_op_list()
    assert not hasattr(custom, "ElemAddSmoothForward")
    assert "MatMulScaleForward" not in custom._get_custom_op_list()
    assert not hasattr(custom, "MatMulScaleForward")

    srcs1 = [os.path.join(cur_dir_path, "custom_opsrc", "elem_add.cpp")]
    lib_path1 = custom_op_tools.build_and_load(
        "elem",
        srcs1,
        extra_include_paths=extra_include_paths,
        build_dir=build_path,
        extra_ldflags=extra_ld_flags,
        verbose=True,
    )
    assert "ElemAddSmoothForward" in custom._get_custom_op_list()
    assert hasattr(custom, "ElemAddSmoothForward")
    assert lib_path1 in custom._get_custom_op_lib_info()
    assert "ElemAddSmoothForward" in custom._get_custom_op_lib_info()[lib_path1]

    srcs2 = [
        os.path.join(cur_dir_path, "custom_opsrc", src)
        for src in ["matmul_scale.cpp", "matmul_scale.cu"]
    ]
    lib_path2 = custom_op_tools.build_and_load(
        "matmul",
        srcs2,
        extra_include_paths=extra_include_paths,
        build_dir=build_path,
        extra_ldflags=extra_ld_flags,
        verbose=True,
    )

    assert "MatMulScaleForward" in custom._get_custom_op_list()
    assert hasattr(custom, "MatMulScaleForward")
    assert lib_path2 in custom._get_custom_op_lib_info()
    assert "MatMulScaleForward" in custom._get_custom_op_lib_info()[lib_path2]

    assert len(custom._get_custom_op_list()) == len(org_op_list) + 4

    custom.unload(lib_path1)
    assert "ElemAddSmoothForward" not in custom._get_custom_op_list()
    assert not hasattr(custom, "ElemAddSmoothForward")
    assert lib_path1 not in custom._get_custom_op_lib_info()

    custom.unload(lib_path2)
    assert "MatMulScaleForward" not in custom._get_custom_op_list()
    assert not hasattr(custom, "MatMulScaleForward")
    assert lib_path1 not in custom._get_custom_op_lib_info()

    assert len(custom._get_custom_op_lib_info()) == 0
    assert custom._get_custom_op_list() == org_op_list

    custom.load(lib_path2)
    assert "MatMulScaleForward" in custom._get_custom_op_list()
    assert hasattr(custom, "MatMulScaleForward")
    assert lib_path2 in custom._get_custom_op_lib_info()
    assert "MatMulScaleForward" in custom._get_custom_op_lib_info()[lib_path2]

    custom.unload(lib_path2)
    assert "MatMulScaleForward" not in custom._get_custom_op_list()
    assert not hasattr(custom, "MatMulScaleForward")
    assert lib_path1 not in custom._get_custom_op_lib_info()

    assert len(custom._get_custom_op_lib_info()) == 0
    assert custom._get_custom_op_list() == org_op_list
