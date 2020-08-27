# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from .module import Module


class QuantStub(Module):
    r"""
    A helper :class:`~.Module` simply returning input. Could be replaced with :class:`~.QATModule`
    version :class:`~.qat.QuantStub` using :func:`~.quantize.quantize_qat`.
    """

    def forward(self, inp):
        return inp


class DequantStub(Module):
    r"""
    A helper :class:`~.Module` simply returning input. Could be replaced with :class:`~.QATModule`
    version :class:`~.qat.DequantStub` using :func:`~.quantize.quantize_qat`.
    """

    def forward(self, inp):
        return inp
