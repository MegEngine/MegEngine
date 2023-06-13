import itertools as it
from typing import Sequence, Tuple, Union

import numpy as np

from ..core._imperative_rt.common import CompNode
from ..tensor import Parameter as MgeParameter
from ..tensor import Tensor as MgeTensor
from .dtype import (
    _np_types,
    _python_scalar_dtypes,
    _scalar_type_to_dtype,
    canonicalize_arg,
)
from .lib import xla_bridge as xb
from .lib import xla_client as xc
from .utils import safe_zip

xla_extention = xc._xla
xe = xla_extention

Backend = xe.Client

device_put_handlers = {}


def _device_put_nparray(x, device):
    backend = xb.get_device_backend(device)
    return (backend.buffer_from_pyval(x, device),)


def _device_put_scalar(x, device):
    def cvt_scalar_to_nparray(x, dtype=None):
        if dtype is None and type(x) in _python_scalar_dtypes:
            dtype = _scalar_type_to_dtype(type(x), x)
        return np.asarray(x, dtype)

    return _device_put_nparray(cvt_scalar_to_nparray(x), device)


def _device_put_device_array(x, device):
    assert False


def _device_put_mge_tensor(x, device):
    x = x.numpy()
    return _device_put_nparray(x, device)


for nt in _np_types:
    device_put_handlers[nt] = _device_put_nparray
for sc in _python_scalar_dtypes:
    device_put_handlers[nt] = _device_put_scalar
device_put_handlers[xc._xla.DeviceArray] = _device_put_device_array
device_put_handlers[MgeTensor] = _device_put_mge_tensor
device_put_handlers[MgeParameter] = _device_put_mge_tensor


def _device_put_impl(x, device):
    x = canonicalize_arg(x)
    return device_put_handlers[type(x)](x, device)


def device_put(x, devices: Sequence[xb.xla_client.Device], replicate: bool = False):
    if replicate:
        return list(
            it.chain.from_iterable(_device_put_impl(x, device) for device in devices)
        )
    else:
        return list(
            it.chain.from_iterable(
                _device_put_impl(val, device) for val, device in safe_zip(x, devices)
            )
        )


def get_xla_backend_and_device(device=None) -> Tuple[Backend, Sequence[xc.Device]]:
    assert device is None, "device assignment is not supported yet"
    device_assignment = [xb.local_devices()[0]]
    backend = xb.get_device_backend(device_assignment[0])
    platform = backend.platform
    platform = xb.canonicalize_platform(platform)

    assert xb.is_known_platform(platform), f"{platform} is not known yet"
    assert platform == "cuda", f"only cuda platfrom is supportted, but get {platform}"
    return backend, device_assignment, platform
