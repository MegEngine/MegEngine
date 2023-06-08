# -*- coding: utf-8 -*-

import os

from ._imperative_rt.core2 import set_cpp_use_symbolic_shape

_use_symbolic_shape = False
if os.environ.get("MEGENGINE_USE_SYMBOLIC_SHAPE"):
    _use_symbolic_shape = True

_use_xla_backend = False


def use_symbolic_shape() -> bool:
    r"""Returns whether tensor.shape returns a tensor instead of a tuple"""
    return _use_symbolic_shape


def set_symbolic_shape(option: bool):
    r"""Sets whether tensor.shape returns a tensor instead of a tuple"""
    global _use_symbolic_shape
    _org = _use_symbolic_shape
    _use_symbolic_shape = option
    return _org


def use_xla_backend() -> bool:
    return _use_xla_backend


def set_use_xla_backend(option: bool) -> bool:
    global _use_xla_backend
    _org = _use_xla_backend
    _use_xla_backend = option
    return _org


set_cpp_use_symbolic_shape(use_symbolic_shape)
