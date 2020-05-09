# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from ... import _internal as mgb
from ... import module as Float
from ...core import Tensor, wrap_io_tensor
from ...core.graph import _use_default_if_none
from ...quantization.utils import register_method_to_class
from ..module import Module


@wrap_io_tensor
def _elemwise_multi_type(mode, *inputs, **kwargs) -> Tensor:
    if all(isinstance(i, (int, float)) for i in inputs):
        device, comp_graph = _use_default_if_none(None, None)
        ret = mgb.opr.elemwise_multi_type(
            *inputs, mode=mode, comp_node=device, comp_graph=comp_graph, **kwargs,
        )
        return ret.inferred_value[0]
    return mgb.opr.elemwise_multi_type(*inputs, mode=mode, **kwargs)


class Elemwise(Module):
    r"""
    quantized module for elemwise operator, inference only.

    :param method: the elemwise method, supported string refer to :class:`~.module.elemwise.Elemwise`.
        it will do quantized operator with specified output quantized dtype.
    """

    _elemwise_multi_type_mode = mgb.opr_param_defs.ElemwiseMultiType.Mode

    def __init__(self, method, dtype=None):
        super().__init__()
        self.method = self._elemwise_multi_type_mode.convert("Q" + method)
        self.output_dtype = dtype

    def forward(self, *inps):
        if self.training:
            raise ValueError("quantized module only support inference.")
        return _elemwise_multi_type(self.method, *inps, dtype=self.output_dtype)


@register_method_to_class(Float.Elemwise)
def to_quantized(float_module):
    r"""
    Replace :class:`~.module.QATModule`'s ``to_quantized`` method.
    implemented here to avoid circular import.
    """
    return Elemwise(float_module.method.name, float_module.act_observer.get_dtype())
