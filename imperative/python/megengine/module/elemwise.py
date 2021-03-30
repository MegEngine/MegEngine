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
    r"""
    A :class:`~.Module` to do :mod:`~.functional.elemwise` operator. Could be replaced with :class:`~.QATModule`
    version :class:`~.qat.Elemwise` using :func:`~.quantize.quantize_qat`.

    :param method: the elemwise method, support the following string.
        It will do the normal elemwise operator for float.

        * "add": a + b
        * "fuse_add_relu": max(x+y, 0)
        * "mul": x * y
        * "min": min(x, y)
        * "max": max(x, y)
        * "sub": x - y
        * "true_div": x / y
        * "fuse_add_sigmoid": sigmoid(x + y)
        * "fuse_add_tanh": tanh(x + y)
        * "relu": x > 0 ? x : 0
        * "abs": x > 0 ? x : -x
        * "sigmoid": sigmoid(x)
        * "exp": exp(x)
        * "tanh": tanh(x)
        * "fuse_mul_add3": x * y + z
        * "fast_tanh": x * (27. + x * x) / (27. + 9. * x * x)
        * "negate": -x
        * "acos": acos(x)
        * "asin": asin(x)
        * "ceil": ceil(x)
        * "cos": cos(x)
        * "expm1": expm1(x)
        * "floor": floor(x)
        * "log": log(x)
        * "log1p": log1p(x)
        * "sin": sin(x)
        * "round": round(x)
        * "erf": erf(x)
        * "erfinv": erfinv(x)
        * "erfc": erfc(x)
        * "erfcinv": erfcinv(x)
        * "abs_grad": abs_grad
        * "floor_div": floor_div
        * "mod": mod
        * "sigmoid_grad": sigmoid_grad
        * "switch_gt0": switch_gt0
        * "tanh_grad": tanh_grad
        * "lt": less
        * "leq": leq
        * "eq": equal
        * "pow": pow
        * "log_sum_exp": log_sum_exp
        * "fast_tanh_grad": fast_tanh_grad
        * "atan2": atan2
        * "cond_leq_mov": cond_leq_mov
        * "h_swish": h_swish
        * "fuse_add_h_swish": h_swish(x+y)
        * "h_swish_grad": h_swish_grad
        * "and": bool binary: x && y
        * "or": bool binary: x || y
        * "xor": bool binary: x ^ y
        * "not": bool unary: ~x
    """

    def __init__(self, method, **kwargs):
        super().__init__(**kwargs)
        self.method = method

    def forward(self, *inps):
        return _elwise(*inps, mode=self.method)
