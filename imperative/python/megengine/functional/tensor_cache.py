from ..core._imperative_rt.core2 import Const
from ..jit.tracing import is_tracing

small_tensor_cache = {}


def _get_scalar_tensor_with_value(value, dtype=None, device=None):
    global small_tensor_cache
    if is_tracing():
        ret = Const(value, dtype, device)
    else:
        cache_key = (value, dtype, device)
        if cache_key not in small_tensor_cache:
            ret = Const(value, dtype, device)
            small_tensor_cache[cache_key] = ret
        else:
            ret = small_tensor_cache[cache_key]
    return ret


def get_scalar_zero(dtype=None, device=None):
    return _get_scalar_tensor_with_value(0, dtype, device)


def get_scalar_zero_point_five(dtype=None, device=None):
    return _get_scalar_tensor_with_value(0.5, dtype, device)


def get_scalar_one(dtype=None, device=None):
    return _get_scalar_tensor_with_value(1, dtype, device)


def get_scalar_two(dtype=None, device=None):
    return _get_scalar_tensor_with_value(2, dtype, device)
