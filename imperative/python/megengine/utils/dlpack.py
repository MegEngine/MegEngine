from typing import Any

from ..core._imperative_rt.core2 import _from_dlpack


def to_dlpack(tensor, stream=None):
    if stream is not None and stream != -1:
        return tensor.__dlpack__(stream)
    else:
        return tensor.__dlpack__()


def from_dlpack(ext_tensor: Any, stream=None):
    if isinstance(stream, int):
        assert stream >= 0, "device stream should be a positive value"
    stream = 0 if stream is None else stream
    return _from_dlpack(ext_tensor, stream)
