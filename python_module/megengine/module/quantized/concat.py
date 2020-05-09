# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Iterable

from ... import _internal as mgb
from ... import functional as F
from ... import module as Float
from ...core.tensor import Tensor
from ...quantization.utils import register_method_to_class
from ..module import Module


class Concat(Module):
    r"""
    A :class:`~.Module` to do quantized concat, inference only.
    """

    def __init__(self, dtype=None):
        super().__init__()
        self.output_dtype = dtype

    def forward(self, inps: Iterable[Tensor], axis: int = 0):
        if self.training:
            raise ValueError("quantized module only support inference.")
        new_inps = (x.astype(self.output_dtype) for x in inps)
        return F.concat(new_inps, axis)


@register_method_to_class(Float.Concat)
def to_quantized(float_module):
    r"""
    Replace :class:`~.module.QATModule`'s ``to_quantized`` method.
    implemented here to avoid circular import.
    """
    return Concat(float_module.act_observer.get_dtype())
