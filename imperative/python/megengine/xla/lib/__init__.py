try:
    import mge_xlalib as mge_xlalib
except ModuleNotFoundError as err:
    msg = (
        "mge-xla requires mge_xlalib to be installed. if this problem happened when "
        "pytest, maybe you have set --doctest-modules for pytest. you can close it "
        "by setup `norecursedirs = megengine/xla` in `pytest.ini`"
    )
    raise ModuleNotFoundError(msg)

import gc
import pathlib
import platform
import warnings
from typing import Optional

import mge_xlalib.cpu_feature_guard as cpu_feature_guard
import mge_xlalib.ducc_fft as ducc_fft
import mge_xlalib.gpu_linalg as gpu_linalg
import mge_xlalib.gpu_prng as gpu_prng
import mge_xlalib.gpu_rnn as gpu_rnn
import mge_xlalib.gpu_solver as gpu_solver
import mge_xlalib.gpu_sparse as gpu_sparse
import mge_xlalib.lapack as lapack
import mge_xlalib.xla_client as xla_client

from ...core._imperative_rt.common import get_cudnn_version as _get_cudnn_version

if int(platform.python_version_tuple()[1]) < 8:
    raise RuntimeError(
        f"xla backend requires Python version >= 3.8, got {platform.python_version()}"
    )

if _get_cudnn_version() < 8600:
    warnings.warn(
        f"xla backend can get the max speed up with CUDNN version >= 8.6.0, "
        f"but current cudnn version is {_get_cudnn_version()}"
    )


cpu_feature_guard.check_cpu_features()


xla_extension = xla_client._xla
pytree = xla_client._xla.pytree
jax_jit = xla_client._xla.jax_jit
pmap_lib = xla_client._xla.pmap_lib


def _xla_gc_callback(*args):
    xla_client._xla.collect_garbage()


gc.callbacks.append(_xla_gc_callback)


xla_extension_version: int = getattr(xla_client, "_version", 0)
mlir_api_version = xla_client.mlir_api_version


def _cuda_path() -> Optional[str]:
    _mgexlalib_path = pathlib.Path(mge_xlalib.__file__).parent
    path = _mgexlalib_path.parent / "nvidia" / "cuda_nvcc"
    if path.is_dir():
        return str(path)
    path = _mgexlalib_path / "cuda"
    if path.is_dir():
        return str(path)
    return None


cuda_path = _cuda_path()

transfer_guard_lib = xla_client._xla.transfer_guard_lib
