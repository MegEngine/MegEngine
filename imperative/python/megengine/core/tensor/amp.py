from .._imperative_rt.core2 import (
    _get_amp_dtype_autocast,
    _get_amp_high_prec_dtype,
    _get_amp_low_prec_dtype,
    _set_amp_dtype_autocast,
    _set_amp_high_prec_dtype,
    _set_amp_low_prec_dtype,
)

_enabled = False
_set_amp_dtype_autocast(_enabled)

__all__ = [
    "enabled",
    "high_prec_dtype",
    "low_prec_dtype",
]


@property
def enabled(mod):
    r"""Get or set amp autocast mode enabled or not.

    Examples:

        .. code-block::

           import megengine as mge
           mge.amp.enabled = True
    """
    return _enabled


@enabled.setter
def enabled(mod, enabled: bool):
    global _enabled
    _enabled = enabled
    _set_amp_dtype_autocast(_enabled)


@property
def high_prec_dtype(mod):
    r"""Get or set amp autocast mode's higher precision dtype. It will change the
    target dtype in tensor casting for better precision. Default: float32.

    Examples:

        .. code-block::

           import megengine as mge
           mge.amp.high_prec_dtype = "float32"
    """
    return _get_amp_high_prec_dtype()


@high_prec_dtype.setter
def high_prec_dtype(mod, dtype: str):
    _set_amp_high_prec_dtype(dtype)


@property
def low_prec_dtype(mod):
    r"""Get or set amp autocast mode's lower precision dtype. It will change the
    target dtype in tensor casting for better speed and memory. Default: float16.

    Examples:

        .. code-block::

           import megengine as mge
           mge.amp.low_prec_dtype = "float16"
    """
    return _get_amp_low_prec_dtype()


@low_prec_dtype.setter
def low_prec_dtype(mod, dtype: str):
    _set_amp_low_prec_dtype(dtype)
