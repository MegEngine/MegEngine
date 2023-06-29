# -*- coding: utf-8 -*-

import os

from ._imperative_rt.core2 import set_cpp_use_symbolic_shape

_use_symbolic_shape = False
if os.environ.get("MEGENGINE_USE_SYMBOLIC_SHAPE"):
    _use_symbolic_shape = True


def use_symbolic_shape() -> bool:
    r"""Returns whether tensor.shape returns a tensor instead of a tuple"""
    return _use_symbolic_shape


def set_symbolic_shape(option: bool):
    r"""Sets whether tensor.shape returns a tensor instead of a tuple"""
    global _use_symbolic_shape
    _org = _use_symbolic_shape
    _use_symbolic_shape = option
    return _org


set_cpp_use_symbolic_shape(use_symbolic_shape)
