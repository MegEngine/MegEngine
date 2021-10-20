# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from abc import abstractmethod

from ..module import Module
from ..qat import QATModule


class QuantizedModule(Module):
    r"""Base class of quantized :class:`~.Module`,
    which should be converted from :class:`~.QATModule` and not support traning.
    """

    def __call__(self, *inputs, **kwargs):
        if self.training:
            raise ValueError("quantized module only support inference.")
        return super().__call__(*inputs, **kwargs)

    def __repr__(self):
        return "Quantized." + super().__repr__()

    @classmethod
    @abstractmethod
    def from_qat_module(cls, qat_module: QATModule):
        r"""
        Return a :class:`~.QATModule` instance converted from
        a float :class:`~.Module` instance.
        """
