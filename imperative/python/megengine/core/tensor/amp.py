# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
_enabled = False
_high_prec_dtype = "float32"
_low_prec_dtype = "float16"


@property
def enabled(mod):
    r"""
    Get or set amp autocast mode enabled or not.

    Examples:

    ..code-block::

        import megengine as mge
        mge.amp.enabled = True

    """
    return _enabled


@enabled.setter
def enabled(mod, enabled: bool):
    global _enabled
    _enabled = enabled


@property
def high_prec_dtype(mod):
    r"""
    Get or set amp autocast mode's higher precision dtype. It will change the
    target dtype in tensor casting for better precision. Default: float32.

    Examples:

    ..code-block::

        import megengine as mge
        mge.amp.high_prec_dtype = "float32"

    """
    return _high_prec_dtype


@high_prec_dtype.setter
def high_prec_dtype(mod, dtype: str):
    global _high_prec_dtype
    _high_prec_dtype = dtype


@property
def low_prec_dtype(mod):
    r"""
    Get or set amp autocast mode's lower precision dtype. It will change the
    target dtype in tensor casting for better speed and memory. Default: float16.

    Examples:

    ..code-block::

        import megengine as mge
        mge.amp.low_prec_dtype = "float16"

    """
    return _low_prec_dtype


@low_prec_dtype.setter
def low_prec_dtype(mod, dtype: str):
    global _low_prec_dtype
    _low_prec_dtype = dtype
