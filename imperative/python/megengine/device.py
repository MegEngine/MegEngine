# -*- coding: utf-8 -*-
import os
import re
from typing import Optional

from .core._imperative_rt.common import CompNode, DeviceType
from .core._imperative_rt.common import get_cuda_version as _get_cuda_version
from .core._imperative_rt.common import get_cudnn_version as _get_cudnn_version
from .core._imperative_rt.common import get_device_prop as _get_device_prop
from .core._imperative_rt.common import get_tensorrt_version as _get_tensorrt_version
from .core._imperative_rt.common import set_prealloc_config as _set_prealloc_config
from .core._imperative_rt.common import what_is_xpu as _what_is_xpu
from .core._imperative_rt.utils import _try_coalesce_all_free_memory

__all__ = [
    "is_cuda_available",
    "is_cambricon_available",
    "is_atlas_available",
    "is_rocm_available",
    "get_device_count",
    "get_default_device",
    "set_default_device",
    "get_mem_status_bytes",
    "get_cuda_compute_capability",
    "get_cuda_device_property",
    "get_cuda_version",
    "get_cudnn_version",
    "get_tensorrt_version",
    "get_allocated_memory",
    "get_reserved_memory",
    "get_max_reserved_memory",
    "get_max_allocated_memory",
    "reset_max_memory_stats",
    "set_prealloc_config",
    "coalesce_free_memory",
]


class _stream_helper:
    def __init__(self):
        self.stream = 1

    def get_next(self):
        out = self.stream
        self.stream = self.stream + 1
        return out


_sh = _stream_helper()


def _valid_device(inp):
    if isinstance(inp, str) and re.match(
        "^([cxg]pu|rocm|multithread)(x|\d+)(:\d+)?$", inp
    ):
        return True
    return False


def _str2device_type(type_str: str, allow_unspec: bool = True):
    type_str = type_str.upper()
    if type_str == "CPU":
        return DeviceType.CPU
    elif type_str == "GPU" or type_str == "CUDA":
        return DeviceType.CUDA
    elif type_str == "CAMBRICON":
        return DeviceType.CAMBRICON
    elif type_str == "ATLAS":
        return DeviceType.ATLAS
    elif type_str == "ROCM" or type_str == "AMDGPU":
        return DeviceType.ROCM
    else:
        assert (
            allow_unspec and type_str == "XPU"
        ), "device type can only be cpu, gpu or xpu"
        return DeviceType.UNSPEC


_device_type_set = {"cpu", "gpu", "xpu", "rocm"}


def get_device_count(device_type: str) -> int:
    r"""Gets number of devices installed on this system.

    Args:
        device_type: device type, one of 'gpu' or 'cpu'
    """
    assert device_type in _device_type_set, "device must be one of {}".format(
        _device_type_set
    )
    device_type = _str2device_type(device_type)
    return CompNode._get_device_count(device_type, False)


def is_cuda_available() -> bool:
    r"""Returns whether cuda device is available on this system."""
    t = _str2device_type("gpu")
    return CompNode._get_device_count(t, False) > 0


def is_cambricon_available() -> bool:
    r"""Returns whether cambricon device is available on this system."""
    t = _str2device_type("cambricon")
    return CompNode._get_device_count(t, False) > 0


def is_atlas_available() -> bool:
    r"""Returns whether atlas device is available on this system."""
    t = _str2device_type("atlas")
    return CompNode._get_device_count(t, False) > 0


def is_rocm_available() -> bool:
    r"""Returns whether rocm device is available on this system."""
    t = _str2device_type("rocm")
    return CompNode._get_device_count(t, False) > 0


def set_default_device(device: str = "xpux"):
    r"""Sets default computing node.

    Args:
        device: default device type.

    Note:
        * The type can be 'cpu0', 'cpu1', etc., or 'gpu0', 'gpu1', etc.,
          to specify the particular CPU or GPU to use.
        * 'cpux' and  'gpux' can also be used to specify any number of CPU or GPU devices.
        * The default value is 'xpux' to specify any device available.
        * The priority of using GPU is higher when both GPU and CPU are available.
        * 'multithread' device type is avaliable when inference,
          which implements multi-threading parallelism at the operator level.
          For example, 'multithread4' will compute with 4 threads.
        * It can also be set by environment variable ``MGE_DEFAULT_DEVICE``.
    """
    assert _valid_device(device), "Invalid device name {}".format(device)
    CompNode._set_default_device(device)


def get_default_device() -> str:
    r"""Gets default computing node.
    It returns the value set by :func:`~.set_default_device`.
    """
    return CompNode._get_default_device()


def get_mem_status_bytes(device: Optional[str] = None):
    r"""Get total and free memory on the computing device in bytes."""
    if device is None:
        device = get_default_device()
    tot, free = CompNode(device).get_mem_status_bytes
    return tot, free


def get_cuda_compute_capability(device: int, device_type=DeviceType.CUDA) -> int:
    r"""Gets compute capability of the specified device.

    Args:
        device: device number.

    Returns:
        a version number, or `SM version`.
    """
    prop = _get_device_prop(device, device_type)
    return prop.major * 10 + prop.minor


def get_cuda_device_property(device: int, device_type=DeviceType.CUDA):
    return _get_device_prop(device, device_type)


def get_allocated_memory(device: Optional[str] = None):
    r"""Returns the current memory occupied by tensors on the computing device in bytes.

    Due to the asynchronous execution of MegEngine, please call megengine._full_sync
    before calling this function in order to get accurate value.
    """
    if device is None:
        device = get_default_device()
    return CompNode(device).get_used_memory


def get_reserved_memory(device: Optional[str] = None):
    r"""Returns the current memory managed by the caching allocator on the computing device in bytes.

    Due to the asynchronous execution of MegEngine, please call megengine._full_sync
    before calling this function in order to get accurate value.
    """
    if device is None:
        device = get_default_device()
    return CompNode(device).get_reserved_memory


def get_max_reserved_memory(device: Optional[str] = None):
    r"""Returns the maximum memory managed by the caching allocator on the computing device in bytes.

    Due to the asynchronous execution of MegEngine, please call megengine._full_sync
    before calling this function in order to get accurate value.
    """
    if device is None:
        device = get_default_device()
    return CompNode(device).get_max_reserved_memory


def get_max_allocated_memory(device: Optional[str] = None):
    r"""Returns the maximum memory occupied by tensors on the computing device in bytes.

    Due to the asynchronous execution of MegEngine, please call megengine._full_sync
    before calling this function in order to get accurate value.
    """
    if device is None:
        device = get_default_device()
    return CompNode(device).get_max_used_memory


def reset_max_memory_stats(device: Optional[str] = None):
    r"""Resets the maximum stats on the computing device.

    Due to the asynchronous execution of MegEngine, please call megengine._full_sync
    before calling this function in order to properly reset memory stats.
    """
    if device is None:
        device = get_default_device()
    CompNode.reset_max_memory_stats(device)


set_default_device(os.getenv("MGE_DEFAULT_DEVICE", "xpux"))


def set_prealloc_config(
    alignment: int = 1,
    min_req: int = 32 * 1024 * 1024,
    max_overhead: int = 0,
    growth_factor=2.0,
    device_type=DeviceType.CUDA,
):
    r"""Specifies how to pre-allocate from raw device allocator.

    Args:
        alignment: specifies the alignment in bytes.
        min_req: min request size in bytes.
        max_overhead: max overhead above required size in bytes.
        growth_factor: request size / cur allocated`
        device_type: the device type
        alignment: int:
        min_req: int:
        max_overhead: int:
    """
    assert alignment > 0
    assert min_req > 0
    assert max_overhead >= 0
    assert growth_factor >= 1
    _set_prealloc_config(alignment, min_req, max_overhead, growth_factor, device_type)


def what_is_xpu():
    r"""Return the precise device type like ``cpu``, ``cuda`` and so on."""
    return _what_is_xpu().name.lower()


def coalesce_free_memory():
    r"""This function will try it best to free all consecutive free chunks back to operating system,
    small pieces may not be returned.

    because of the async processing of megengine, the effect of this func may not be reflected
    immediately. if you want to see the effect immediately, you can call megengine._full_sync after
    this func was called

    .. note::

       * This function will not move any memory in-use;
       * This function may do nothing if there are no chunks that can be freed.
    """
    return _try_coalesce_all_free_memory()


def get_cuda_version():
    r"""Gets the CUDA version used when compiling MegEngine.

    Returns:
        a version number, indicating `CUDA_VERSION_MAJOR * 1000 + CUDA_VERSION_MINOR * 10`.
    """
    return _get_cuda_version()


def get_cudnn_version():
    r"""Get the Cudnn version used when compiling MegEngine.

    Returns:
        a version number, indicating `CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL`.
    """
    return _get_cudnn_version()


def get_tensorrt_version():
    r"""Get the TensorRT version used when compiling MegEngine.

    Returns:
        a version number, indicating `NV_TENSORRT_MAJOR * 1000 + NV_TENSORRT_MINOR * 100 + NV_TENSORRT_PATCH`.
    """
    return _get_tensorrt_version()
