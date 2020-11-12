# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from .. import elemwise as Float
from .module import QATModule


class Elemwise(Float.Elemwise, QATModule):
    r"""
    A :class:`~.QATModule` to do elemwise operator with QAT support.
    Could be applied with :class:`~.Observer` and :class:`~.FakeQuantize`.

    :param method: the elemwise method, see :class:`~.module.elemwise.Elemwise` for detail.
    """

    with_weight = False

    def forward(self, *inps):
        return self.apply_quant_activation(super().forward(*inps))

    @classmethod
    def from_float_module(cls, float_module: Float.Elemwise):
        r"""
        Return a :class:`~.QATModule` instance converted from
        a float :class:`~.Module` instance.
        """
        return cls(float_module.method)
