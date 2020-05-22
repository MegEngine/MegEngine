# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ... import _internal as mgb
from ... import module as Float
from ...quantization.utils import register_method_to_class
from ..module import Module


class QuantStub(Module):
    r"""
    A helper quantize operation on input and inference only.
    """

    def __init__(self):
        super().__init__()
        self.scale = 1.0
        self.zero_point = 0.0
        self.output_dtype = mgb.dtype.qint8(self.scale)

    def forward(self, inp):
        if self.training:
            raise ValueError("quantized module only support inference.")
        return inp.astype(self.output_dtype)


class DequantStub(Module):
    r"""
    A helper de-quantize operation and inference only.
    """

    def forward(self, inp):
        if self.training:
            raise ValueError("quantized module only support inference.")
        return inp.astype("float32")


@register_method_to_class(Float.QuantStub)
def to_quantized(float_module):
    r"""
    Replace :class:`~.module.QATModule`'s ``to_quantized`` method.
    implemented here to avoid circular import.
    """
    qmod = QuantStub()
    qmod.output_dtype = float_module.act_observer.get_dtype()
    qmod.scale, qmod.zero_point = float_module.act_observer.get_qparams()
    return qmod


@register_method_to_class(Float.DequantStub)
def to_quantized(float_module):
    r"""
    Replace :class:`~.module.QATModule`'s ``to_quantized`` method.
    implemented here to avoid circular import.
    """
    qmod = DequantStub()
    return qmod
