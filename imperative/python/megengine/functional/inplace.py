# -*- coding: utf-8 -*-
from ..core._imperative_rt.core2 import apply
from ..core.ops import builtin
from ..core.ops.builtin import InplaceAdd


def _inplace_add_(dest, delta, alpha, beta):
    dest._reset(apply(InplaceAdd(), dest, delta, alpha, beta)[0])
    return dest
