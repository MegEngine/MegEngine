# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from abc import abstractmethod

from ..module import Module
from ..qat import QATModule


class QuantizedModule(Module):
    r"""
    Base class of quantized Module, which should be converted from QATModule
    and not support traning.
    """

    def __call__(self, *inputs, **kwargs):
        if self.training:
            raise ValueError("quantized module only support inference.")
        return super().__call__(*inputs, **kwargs)

    @classmethod
    @abstractmethod
    def from_qat_module(cls, qat_module: QATModule):
        r"""
        Return a :class:`~.QuantizedModule` instance converted from a
        :class:`~.QATModule` instance.
        """
