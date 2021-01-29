# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from enum import Enum

from ..functional.elemwise import _elwise
from ..tensor import Tensor
from .module import Module


class ElemwiseMethod(Enum):
    r"""
    Enumeration of elemwise operators, including:
      * ``ADD``: a + b
      * ``FUSE_ADD_RELU``: max(x+y, 0)
      * ``MUL``: x * y
      * ``MIN``: min(x, y)
      * ``MAX``: max(x, y)
      * ``SUB``: x - y
      * ``TRUE_DIV``: x / y
      * ``FUSE_ADD_SIGMOID``: sigmoid(x + y)
      * ``FUSE_ADD_TANH``: tanh(x + y)
      * ``RELU``: x > 0 ? x : 0
      * ``ABS``: x > 0 ? x : -x
      * ``SIGMOID``: sigmoid(x)
      * ``EXP``: exp(x)
      * ``TANH``: tanh(x)
      * ``FUSE_MUL_ADD3``: x * y + z
      * ``FAST_TANH``: x * (27. + x * x) / (27. + 9. * x * x)
      * ``NEGATE``: -x
      * ``ACOS``: acos(x)
      * ``ASIN``: asin(x)
      * ``CEIL``: ceil(x)
      * ``COS``: cos(x)
      * ``EXPM1``: expm1(x)
      * ``FLOOR``: floor(x)
      * ``LOG``: log(x)
      * ``LOG1P``: log1p(x)
      * ``SIN``: sin(x)
      * ``ROUND``: round(x)
      * ``ERF``: erf(x)
      * ``ERFINV``: erfinv(x)
      * ``ERFC``: erfc(x)
      * ``ERFCINV``: erfcinv(x)
      * ``ABS_GRAD``: abs_grad
      * ``FLOOR_DIV``: floor_div
      * ``MOD``: mod
      * ``SIGMOID_GRAD``: sigmoid_grad
      * ``SWITCH_GT0``: switch_gt0
      * ``TANH_GRAD``: tanh_grad
      * ``LT``: less
      * ``LEQ``: leq
      * ``EQ``: equal
      * ``POW``: pow
      * ``LOG_SUM_EXP``: log_sum_exp
      * ``FAST_TANH_GRAD``: fast_tanh_grad
      * ``ATAN2``: atan2
      * ``COND_LEQ_MOV``: cond_leq_mov
      * ``H_SWISH``: h_swish
      * ``FUSE_ADD_H_SWISH``: h_swish(x+y)
      * ``H_SWISH_GRAD``: h_swish_grad
      * ``AND``: bool binary: x && y
      * ``OR``: bool binary: x || y
      * ``XOR``: bool binary: x ^ y
      * ``NOT``: bool unary: ~x
    """

    ADD = "ADD"
    FUSE_ADD_RELU = "FUSE_ADD_RELU"
    MUL = "MUL"
    MIN = "MIN"
    MAX = "MAX"
    SUB = "SUB"
    TRUE_DIV = "TRUE_DIV"
    FUSE_ADD_SIGMOID = "FUSE_ADD_SIGMOID"
    FUSE_ADD_TANH = "FUSE_ADD_TANH"
    RELU = "RELU"
    ABS = "ABS"
    SIGMOID = "SIGMOID"
    EXP = "EXP"
    TANH = "TANH"
    FUSE_MUL_ADD3 = "FUSE_MUL_ADD3"
    FAST_TANH = "FAST_TANH"
    NEGATE = "NEGATE"
    ACOS = "ACOS"
    ASIN = "ASIN"
    CEIL = "CEIL"
    COS = "COS"
    EXPM1 = "EXPM1"
    FLOOR = "FLOOR"
    LOG = "LOG"
    LOG1P = "LOG1P"
    SIN = "SIN"
    ROUND = "ROUND"
    ERF = "ERF"
    ERFINV = "ERFINV"
    ERFC = "ERFC"
    ERFCINV = "ERFCINV"
    ABS_GRAD = "ABS_GRAD"
    FLOOR_DIV = "FLOOR_DIV"
    MOD = "MOD"
    SIGMOID_GRAD = "SIGMOID_GRAD"
    SWITCH_GT0 = "SWITCH_GT0"
    TANH_GRAD = "TANH_GRAD"
    LT = "LT"
    LEQ = "LEQ"
    EQ = "EQ"
    POW = "POW"
    LOG_SUM_EXP = "LOG_SUM_EXP"
    FAST_TANH_GRAD = "FAST_TANH_GRAD"
    ATAN2 = "ATAN2"
    COND_LEQ_MOV = "COND_LEQ_MOV"
    H_SWISH = "H_SWISH"
    FUSE_ADD_H_SWISH = "FUSE_ADD_H_SWISH"
    H_SWISH_GRAD = "H_SWISH_GRAD"
    AND = "AND"
    OR = "OR"
    XOR = "XOR"
    NOT = "NOT"


class Elemwise(Module):
    r"""
    A :class:`~.Module` to do elemwise operator. Could be replaced with :class:`~.QATModule`
    version :class:`~.qat.elemwise.Elemwise` using :func:`~.quantize.quantize_qat`.

    :param method: the elemwise method, see :class:`~.ElemwiseMethod` for a detailed list.
    """

    def __init__(self, method: ElemwiseMethod):
        super().__init__()
        if isinstance(method, str):
            try:
                method = ElemwiseMethod[method]
            except KeyError:
                raise ValueError("Unknown elemwise operation: {}".format(method))
        self.method: ElemwiseMethod = method

    def forward(self, *inps):
        return _elwise(*inps, mode=self.method.value)
