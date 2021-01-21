# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ..core._imperative_rt.core2 import apply
from ..core.ops import builtin
from ..tensor import Tensor

__all__ = [
    "cvt_color",
]


def cvt_color(inp: Tensor, mode: str = ""):
    r"""
    Convert images from one format to another

    :param inp: input images.
    :param mode: format mode.
    :return: convert result.

    Examples:

    .. testcode::

        import numpy as np
        import megengine as mge
        import megengine.functional as F

        x = mge.tensor(np.array([[[[-0.58675045, 1.7526233, 0.10702174]]]]).astype(np.float32))
        y = F.img_proc.cvt_color(x, mode="RGB2GRAY")
        print(y.numpy())

    Outputs:

    .. testoutput::

        [[[[0.86555195]]]]

    """
    assert mode in builtin.CvtColor.Mode.__dict__, "unspport mode for cvt_color"
    mode = getattr(builtin.CvtColor.Mode, mode)
    assert isinstance(mode, builtin.CvtColor.Mode)
    op = builtin.CvtColor(mode=mode)
    (out,) = apply(op, inp)
    return out
