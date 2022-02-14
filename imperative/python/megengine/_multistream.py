from .core._imperative_rt.core2 import apply
from .core.ops.builtin import Barrier
from .tensor import Tensor

_dummy_tensors = {}


def _get_dummy_tensor(device):
    if device not in _dummy_tensors:
        _dummy_tensors[device] = Tensor([], device=device)
    return _dummy_tensors[device]


def record_event(device):
    x = _get_dummy_tensor(device)
    (x,) = apply(Barrier(device, 1), x)
    return x


def wait_event(device, event):
    apply(Barrier(device, 0), event)
