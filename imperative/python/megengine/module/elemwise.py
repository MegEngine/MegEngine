# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ..functional.elemwise import _elwise
from ..tensor import Tensor
from .module import Module


class Elemwise(Module):
    r"""A :class:`~.Module` to do :mod:`~.functional.elemwise` operator. Could be replaced with :class:`~.QATModule`
    version :class:`~.qat.Elemwise` using :func:`~.quantize.quantize_qat`.

    Args:
        method: the elemwise method, support the following string.
                It will do the normal elemwise operator for float.
    """

    def __init__(self, method, **kwargs):
        super().__init__(**kwargs)
        self.method = method

    def forward(self, *inps):
        return _elwise(*inps, mode=self.method)
