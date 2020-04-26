# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Iterable

from .. import functional as F
from ..core.tensor import Tensor
from .module import QATModule


class Concat(QATModule):
    r"""
    A :class:`~.QATModule` to do functional concat, should replace concat with this module,
    supporting ``qat`` mode and ``quantized`` mode.
    """

    def forward(self, inps: Iterable[Tensor], axis: int = 0):
        return F.concat(inps, axis)

    def forward_qat(self, inps: Iterable[Tensor], axis: int = 0):
        return self.apply_fakequant_with_observer(
            self.forward(inps, axis), self.act_fake_quant, self.act_observer
        )
