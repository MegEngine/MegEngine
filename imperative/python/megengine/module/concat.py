# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Iterable

from ..functional import concat
from ..tensor import Tensor
from .module import Module


class Concat(Module):
    r"""
    A :class:`~.Module` to do functional concat. Could be replaced with :class:`~.QATModule`
    version :class:`~.qat.concat.Concat` using :func:`~.quantize.quantize_qat`.
    """

    def forward(self, inps: Iterable[Tensor], axis: int = 0):
        return concat(inps, axis)
