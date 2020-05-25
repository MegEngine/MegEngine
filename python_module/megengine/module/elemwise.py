# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from .. import _internal as mgb
from ..core import Tensor, wrap_io_tensor
from ..core.graph import _use_default_if_none
from .module import Module


@wrap_io_tensor
def _elemwise_func(mode, *inputs, **kwargs) -> Tensor:
    if all(isinstance(i, (int, float)) for i in inputs):
        device, comp_graph = _use_default_if_none(None, None)
        ret = mgb.opr.elemwise(
            *inputs, mode=mode, comp_node=device, comp_graph=comp_graph, **kwargs
        )
        return ret.inferred_value[0]
    return mgb.opr.elemwise(*inputs, mode=mode, **kwargs)


class Elemwise(Module):
    r"""
    A :class:`~.Module` to do elemwise operator. Could be replaced with :class:`~.QATModule`
    version :class:`~.qat.elemwise.Elemwise` using :func:`~.quantize.quantize_qat`.

    :param method: the elemwise method, support the following string.
        It will do the normal elemwise operator for float.

        * "ADD": a + b
        * "FUSE_ADD_RELU": max(x+y, 0)
        * "MUL": x * y
        * "MIN": min(x, y)
        * "MAX": max(x, y)
        * "SUB": x - y
        * "TRUE_DIV": x / y
        * "FUSE_ADD_SIGMOID": sigmoid(x + y)
        * "FUSE_ADD_TANH": tanh(x + y)
        * "RELU": x > 0 ? x : 0
        * "ABS": x > 0 ? x : -x
        * "SIGMOID": sigmoid(x)
        * "EXP": exp(x)
        * "TANH": tanh(x)
        * "FUSE_MUL_ADD3": x * y + z
        * "FAST_TANH": fast_tanh(x)
        * "NEGATE": -x
        * "ACOS": acos(x)
        * "ASIN": asin(x)
        * "CEIL": ceil(x)
        * "COS": cos(x)
        * "EXPM1": expm1(x)
        * "FLOOR": floor(x)
        * "LOG": log(x)
        * "LOG1P": log1p(x)
        * "SIN": sin(x)
        * "ROUND": round(x)
        * "ERF": erf(x)
        * "ERFINV": erfinv(x)
        * "ERFC": erfc(x)
        * "ERFCINV": erfcinv(x)
        * "ABS_GRAD": abs_grad
        * "FLOOR_DIV": floor_div
        * "MOD": mod
        * "SIGMOID_GRAD": sigmoid_grad
        * "SWITCH_GT0": switch_gt0
        * "TANH_GRAD": tanh_grad
        * "LT": lt
        * "LEQ": leq
        * "EQ": eq
        * "POW": pow
        * "LOG_SUM_EXP": log_sum_exp
        * "FAST_TANH_GRAD": fast_tanh_grad
        * "ATAN2": atan2
        * "COND_LEQ_MOV": cond_leq_mov
        * "H_SWISH": h_swish
        * "FUSE_ADD_H_SWISH": h_swish(x+y)
        * "H_SWISH_GRAD": h_swish_grad
    """

    _elemwise_mode_type = mgb.opr_param_defs.Elemwise.Mode

    def __init__(self, method):
        super().__init__()
        self.method = self._elemwise_mode_type.convert(method)

    def forward(self, *inps):
        return _elemwise_func(self.method, *inps)
