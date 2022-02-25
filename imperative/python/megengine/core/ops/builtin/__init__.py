# -*- coding: utf-8 -*-
from ..._imperative_rt import OpDef, ops

__all__ = ["OpDef"]

for k, v in ops.__dict__.items():
    if isinstance(v, type) and issubclass(v, OpDef):
        globals()[k] = v
        __all__.append(k)
