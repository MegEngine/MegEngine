# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Iterable

from ... import functional as F
from ...tensor import Tensor
from ..qat import concat as QAT
from .module import QuantizedModule


class Concat(QuantizedModule):
    r"""
    A :class:`~.QuantizedModule` to do quantized concat, used for inference only.
    """

    def __init__(self, dtype=None):
        super().__init__()
        self.output_dtype = dtype

    def forward(self, inps: Iterable[Tensor], axis: int = 0):
        new_inps = (x.astype(self.output_dtype) for x in inps)
        return F.concat(new_inps, axis)

    @classmethod
    def from_qat_module(cls, qat_module: QAT.Concat):
        r"""
        Return a :class:`~.QuantizedModule` instance converted from a
        :class:`~.QATModule` instance.
        """
        return cls(qat_module.get_activation_dtype())
