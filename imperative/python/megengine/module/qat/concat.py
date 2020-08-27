# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Iterable

from ...tensor import Tensor
from .. import concat as Float
from .module import QATModule


class Concat(Float.Concat, QATModule):
    r"""
    A :class:`~.QATModule` to do functional concat with QAT support.
    Could be applied with :class:`~.Observer` and :class:`~.FakeQuantize`.
    """

    def forward(self, inps: Iterable[Tensor], axis: int = 0):
        return self.apply_quant_activation(super().forward(inps, axis))

    @classmethod
    def from_float_module(cls, float_module):
        r"""
        Return a :class:`~.QATModule` instance converted from
        a float :class:`~.Module` instance.
        """
        return cls()
