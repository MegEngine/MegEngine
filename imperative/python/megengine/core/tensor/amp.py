# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

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
