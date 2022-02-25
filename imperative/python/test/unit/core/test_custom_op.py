import os
import platform
import shutil
import sys

import numpy as np
import pytest

import megengine
import megengine.functional as F
import megengine.optimizer as optim
from megengine import jit
from megengine.autodiff import Function, GradManager
from megengine.core._imperative_rt.core2 import apply
from megengine.core.ops import custom
from megengine.device import get_device_count
from megengine.module import Conv2d, Linear, Module
from megengine.random import normal
from megengine.tensor import Parameter, Tensor
from megengine.utils import custom_op_tools


def compare(ref, real):
    if ref.shape != real.shape:
        real = real.T
    np.testing.assert_allclose(ref, real, rtol=1e-3, atol=1e-5)


def build_and_clean(test_func):
    def wrapper():
        cur_dir_path = os.path.dirname(os.path.abspath(__file__))
        build_root_dir = custom_op_tools._get_default_build_root()
        build_path = os.path.join(build_root_dir, "custom_opsrc", "build")

        if os.path.exists(build_path):
            shutil.rmtree(build_path)

        mgb_root_path = os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(cur_dir_path)))
            )
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
                                extra_ld_flags += [
                                    "-L{} -Wl,-rpath,{}".format(ld_dir, ld_dir)
                                ]
                                break

        if get_device_count("gpu") > 0:
            custom_opsrc = [
                os.path.join(cur_dir_path, "custom_opsrc", "matmul_scale.cpp"),
                os.path.join(cur_dir_path, "custom_opsrc", "matmul_scale.cu"),
            ]
        else:
            custom_opsrc = [os.path.join(cur_dir_path, "custom_opsrc", "elem_add.cpp")]

        try:
            lib_path = custom_op_tools.build_and_load(
                "test_op",
                custom_opsrc,
                extra_include_paths=extra_include_paths,
                extra_ldflags=extra_ld_flags,
                build_dir=build_path,
                verbose=False,
            )
            test_func()
            custom.unload(lib_path)

        finally:
            if os.path.exists(build_path):
                shutil.rmtree(build_path)

    return wrapper


@pytest.mark.skipif(
    get_device_count("gpu") > 0, reason="elem_add operator is only supported on CPU"
)
@build_and_clean
def test_custom_op_cpu_build():
    assert "ElemAddSmoothForward" in custom._get_custom_op_list()
    assert "ElemAddSmoothBackward" in custom._get_custom_op_list()
    assert hasattr(custom, "ElemAddSmoothForward")
    assert hasattr(custom, "ElemAddSmoothBackward")


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="GPU kernel is only support on Linux and Windows",
)
@pytest.mark.skipif(
    get_device_count("gpu") < 1, reason="matmul scale operator is only supported on GPU"
)
@build_and_clean
def test_custom_op_gpu_build():
    assert "MatMulScaleForward" in custom._get_custom_op_list()
    assert "MatMulScaleBackward" in custom._get_custom_op_list()
    assert hasattr(custom, "MatMulScaleForward")
    assert hasattr(custom, "MatMulScaleBackward")
