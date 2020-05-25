# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from .. import functional as F
from .._internal.dtype import _metadata_dict
from ..module import Module
from .observer import ObserverMode, Round


class FakeQuantize(Module):
    r"""
    A module to do quant and dequant according to observer's scale and zero_point.
    """

    def __init__(self, dtype: str, enable: bool = True):
        super().__init__()
        if not dtype in _metadata_dict.keys():
            raise ValueError(
                "unknown dtype: {}, only support {}".format(
                    dtype, _metadata_dict.keys()
                )
            )
        self.dtype = dtype
        self.qmin = _metadata_dict[dtype].qmin
        self.qmax = _metadata_dict[dtype].qmax
        self.enabled = enable

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def forward(self, inp, q_dict):
        if self.enabled:
            if q_dict["mode"] == ObserverMode.SYMMERTIC:
                scale = q_dict["scale"]
                # Quant
                oup = Round()(inp / scale)
                # clip
                oup = F.minimum(F.maximum(oup, self.qmin), self.qmax)
                # DeQuant
                oup = (oup) * scale
                return oup
            else:
                scale = q_dict["scale"]
                zero_point = q_dict["zero_point"]
                # Quant
                oup = Round()(inp / scale) + zero_point
                # clip
                oup = F.minimum(F.maximum(oup, self.qmin), self.qmax)
                # DeQuant
                oup = (oup - zero_point) * scale
                return oup
        return inp
