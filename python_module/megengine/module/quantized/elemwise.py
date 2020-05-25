# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ... import _internal as mgb
from ...core import Tensor, wrap_io_tensor
from ...core.graph import _use_default_if_none
from ..qat import elemwise as QAT
from .module import QuantizedModule


@wrap_io_tensor
def _elemwise_multi_type(mode, *inputs, **kwargs) -> Tensor:
    if all(isinstance(i, (int, float)) for i in inputs):
        device, comp_graph = _use_default_if_none(None, None)
        ret = mgb.opr.elemwise_multi_type(
            *inputs, mode=mode, comp_node=device, comp_graph=comp_graph, **kwargs,
        )
        return ret.inferred_value[0]
    return mgb.opr.elemwise_multi_type(*inputs, mode=mode, **kwargs)


class Elemwise(QuantizedModule):
    r"""quantized version of :class:`~.qat.elemwise.Elemwise`."""

    _elemwise_multi_type_mode = mgb.opr_param_defs.ElemwiseMultiType.Mode

    def __init__(self, method, dtype=None):
        super().__init__()
        self.method = self._elemwise_multi_type_mode.convert("Q" + method)
        self.output_dtype = dtype

    def forward(self, *inps):
        if self.training:
            raise ValueError("quantized module only support inference.")
        return _elemwise_multi_type(self.method, *inps, dtype=self.output_dtype)

    @classmethod
    def from_qat_module(cls, qat_module: QAT.Elemwise):
        r"""
        return a :class:`~.QuantizedModule` instance converted from a
        :class:`~.QATModule` instance.
        """
        return cls(qat_module.method.name, qat_module.get_activation_dtype())
