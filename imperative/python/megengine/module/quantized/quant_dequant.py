# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ..qat import quant_dequant as QAT
from .module import QuantizedModule


class QuantStub(QuantizedModule):
    r"""
    Quantized version of :class:`~.qat.QuantStub`,
    will convert input to quantized dtype.
    """

    def __init__(self, dtype=None, **kwargs):
        super().__init__(**kwargs)
        self.output_dtype = dtype

    def forward(self, inp):
        return inp.astype(self.output_dtype)

    @classmethod
    def from_qat_module(cls, qat_module: QAT.QuantStub):
        r"""
        Return a :class:`~.QuantizedModule` instance converted from a
        :class:`~.QATModule` instance.
        """
        return cls(qat_module.get_activation_dtype(), name=qat_module.name)


class DequantStub(QuantizedModule):
    r"""
    Quantized version of :class:`~.qat.DequantStub`,
    will restore quantized input to float32 dtype.
    """

    def forward(self, inp):
        return inp.astype("float32")

    @classmethod
    def from_qat_module(cls, qat_module: QAT.DequantStub):
        r"""
        Return a :class:`~.QuantizedModule` instance converted from a
        :class:`~.QATModule` instance.
        """
        return cls(name=qat_module.name)
