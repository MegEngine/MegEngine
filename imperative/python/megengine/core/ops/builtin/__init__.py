# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import warnings
from typing import Union

from ..._imperative_rt import OpDef, ops
from ...tensor.core import OpBase, TensorBase, TensorWrapperBase, apply
from .._internal import all_ops
from .._internal.helper import PodOpVisitor

# register OpDef as a "virtual subclass" of OpBase, so any of registered
# apply(OpBase, ...) rules could work well on OpDef
OpBase.register(OpDef)

# forward to apply(OpDef, ...)
@apply.add
def _(op: PodOpVisitor, *args: Union[TensorBase, TensorWrapperBase]):
    return apply(op.to_c(), *args)


__all__ = ["OpDef", "PodOpVisitor"]

for k, v in all_ops.__dict__.items():
    if isinstance(v, type) and issubclass(v, PodOpVisitor):
        globals()[k] = v
        __all__.append(k)

for k, v in ops.__dict__.items():
    if isinstance(v, type) and issubclass(v, OpDef):
        globals()[k] = v
        __all__.append(k)
