import math
from functools import partial
from typing import Sequence, Union

import numpy as np

from ...core._imperative_rt import ops as mops
from .. import ir_utils
from ..lib.mlir.dialects import hlo
from .hlotensor import HLOTensor
from .utils import register_lower_rule


def _infer_elemwise_oshape(inp_shapes):
    def _infer_binary_elemwise_oshape(lhs_shape, rhs_shape):
        if len(lhs_shape) == 0:
            return rhs_shape
        if len(rhs_shape) == 0:
            return lhs_shape

        if np.prod(lhs_shape) == 1 and len(rhs_shape) != 0:
            return rhs_shape
        if np.prod(rhs_shape) == 1 and len(rhs_shape) != 0:
            return lhs_shape

        oshape = []
        if len(lhs_shape) == len(rhs_shape):
            for l, r in zip(lhs_shape, rhs_shape):
                if l == r:
                    oshape.append(l)
                elif l == 1:
                    oshape.append(r)
                elif r == 1:
                    oshape.append(l)
                else:
                    assert False, f"infer elemwise shape error: {lhs_shape} {rhs_shape}"
        else:
            shorter = lhs_shape if len(lhs_shape) < len(rhs_shape) else rhs_shape
            longer = lhs_shape if len(lhs_shape) > len(rhs_shape) else rhs_shape

            right_part = longer[-len(shorter) :]
            for l, s in zip(right_part, shorter):
                assert (
                    l == s or s == 1
                ), f"infer elemwise shape error: {lhs_shape} {rhs_shape}"
            oshape = longer

        return oshape

    oshape = tuple()
    for ishape in inp_shapes:
        oshape = _infer_binary_elemwise_oshape(ishape, oshape)
    return oshape


def _infer_elemwise_odtype(inp_dtypes):
    oup_dtype = inp_dtypes[0]
    for inp_dtype in inp_dtypes:
        assert (
            inp_dtype == oup_dtype
        ), f"elemwise inputs has different dtype {inp_dtypes}"
    return oup_dtype


def _compare(lhs, rhs, mode, comparison_type=None):
    """
    mod: can be 
        'EQ' (equal-to),
        'NE' (not equal-to),
        'GE' (greater-or-equal-than),
        'GT' (greater-than),
        'LE' (less-or-equal-than),
        'LT' (less-than)
    comparision_type: can be 'UNSIGNED', 'SIGNED', 'FLOAT'
    """
    lhs = HLOTensor(lhs) if not isinstance(lhs, HLOTensor) else lhs
    rhs = HLOTensor(rhs) if not isinstance(rhs, HLOTensor) else rhs
    oshape = _infer_elemwise_oshape([lhs.shape, rhs.shape])

    lhs = lhs.broadcast_to(oshape)
    rhs = rhs.broadcast_to(oshape)

    if comparison_type is None:
        if lhs.dtype in [np.int64, np.int32, np.int16, np.int8]:
            assert rhs.dtype in [np.int64, np.int32, np.int16, np.int8]
            comparison_type = "SIGNED"
        elif lhs.dtype in [np.uint64, np.uint32, np.uint16, np.uint8]:
            assert rhs.dtype in [np.uint64, np.uint32, np.uint16, np.uint8]
            comparison_type = "UNSIGNED"
        elif lhs.dtype in [np.float64, np.float32, np.float16]:
            assert rhs.dtype in [np.float64, np.float32, np.float16]
            comparison_type = "FLOAT"
        else:
            assert False, f"invalid dtype for compare {lhs.dtype} .vs {rhs.dtype}"

    return HLOTensor(
        hlo.CompareOp(
            lhs.tensor,
            rhs.tensor,
            hlo.ComparisonDirectionAttr.get(mode),
            compare_type=hlo.ComparisonTypeAttr.get(comparison_type),
        ).result
    )


def _elemwise(hlo_op, inps):
    hinps = [HLOTensor(inp) if not isinstance(inp, HLOTensor) else inp for inp in inps]

    ishapes = [inp.shape for inp in hinps]
    idtypes = [inp.dtype for inp in hinps]

    oshape = _infer_elemwise_oshape(ishapes)
    odtype = _infer_elemwise_odtype(idtypes)

    broadcasted_inps = [hinp.broadcast_to(oshape) for hinp in hinps]
    results = hlo_op(*[binp.tensor for binp in broadcasted_inps]).results
    assert len(results) == 1, f"elemwise op {hlo_op} should have only one output"
    return HLOTensor(results[0], oshape, odtype)


def _elemwise_unary(hlo_op, a):
    return _elemwise(hlo_op, [a])


def _elemwise_binary(hlo_op, a, b):
    return _elemwise(hlo_op, [a, b])


neg = partial(_elemwise_unary, hlo.NegOp)
abs = partial(_elemwise_unary, hlo.AbsOp)
tanh = partial(_elemwise_unary, hlo.TanhOp)
exp = partial(_elemwise_unary, hlo.ExpOp)
sqrt = partial(_elemwise_unary, hlo.SqrtOp)
log = partial(_elemwise_unary, hlo.LogOp)

add = partial(_elemwise_binary, hlo.AddOp)
sub = partial(_elemwise_binary, hlo.SubtractOp)
mul = partial(_elemwise_binary, hlo.MulOp)
div = partial(_elemwise_binary, hlo.DivOp)
pow = partial(_elemwise_binary, hlo.PowOp)


equal = partial(_compare, mode="EQ")
not_equal = partial(_compare, mode="NE")
greater = partial(_compare, mode="GT")
greater_equal = partial(_compare, mode="GE")
less = partial(_compare, mode="LT")
less_equal = partial(_compare, mode="LE")


def abs_grad(x, dy):
    return (x / abs(x)) * dy


def tanh_grad(x, dy):
    return (1.0 - tanh(x) ** 2.0) * dy


def bitcast(inp, oshape, odtype):
    odtype = np.dtype(odtype) if isinstance(odtype, str) else odtype
    return HLOTensor(
        hlo.BitcastConvertOp(
            ir_utils.make_ir_type_according_meta(oshape, odtype), inp.tensor
        ).result
    )


def typecvt(inp, odtype):
    odtype = np.dtype(odtype) if isinstance(odtype, str) else odtype
    return HLOTensor(
        hlo.ConvertOp(
            ir_utils.make_ir_type_according_meta(inp.shape, odtype), inp.tensor
        ).result
    )


def gelu(inp, approximate: bool = True):
    if approximate:
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        a = inp ** 3.0
        b = 0.044715 * a
        c = inp + b
        d = sqrt_2_over_pi * c
        e = tanh(d)
        f = 1.0 + e
        g = 0.5 * f
        h = inp * g
    else:
        assert False, "only approximate gelu is supported"
    return h


def erfcc(inp):
    _a = abs(inp)
    _b = 0.5 * _a
    _c = 1.0 + _b
    _d = 1.0 / _c
    _e = _d * 0.17087277
    _f = -0.82215223 + _e
    _g = _d * _f
    _h = 1.48851587 + _g
    _i = _d * _h
    _j = -1.13520398 + _i
    _k = _d * _j
    _l = 0.27886807 + _k
    _m = _d * _l
    _n = -0.18628806 + _m
    _o = _d * _n
    _p = 0.09678418 + _o
    _q = _d * _p
    _r = 0.37409196 + _q
    _s = _d * _r
    _t = 1.00002368 + _s
    _u = _d * _t
    _v = inp * inp
    _w = -_v
    _x = _w - 1.26551223
    _y = _x + _u
    _z = exp(_y)
    _aa = _d * _z
    _ab = 1.0 - _aa
    _ac = -_ab

    _ad = (inp >= 0.0).astype(inp.dtype)
    _ae = (inp < 0.0).astype(inp.dtype)
    _af = _ad * _ab
    _ag = _ae * _ac
    ret = _af + _ag
    return ret


def gelu_grad(x, dy, approximate: bool = True):
    if approximate:
        _a = x * x
        _b = -0.5 * _a
        _c = exp(_b)
        phi = 0.3989422804014327 * _c
        _d = x / math.sqrt(2.0)
        _e = erfcc(_d)
        _f = 1.0 + _e
        normcdf_v = 0.5 * _f
        _g = x * phi
        _h = normcdf_v + _g
        ret = dy * _h
    else:
        assert False
    return ret


def relu(inp):
    mask = (inp > 0.0).astype(inp.dtype)
    return inp * mask


def relu_grad(x, dy):
    mask = (x > 0.0).astype(x.dtype)
    return dy * mask


# Elemwise.Mode is unhashable, so we convert it to str
mge_elemwise_to_xla = {
    str(mops.Elemwise.Mode.ADD): add,
    str(mops.Elemwise.Mode.MUL): mul,
    str(mops.Elemwise.Mode.SUB): sub,
    str(mops.Elemwise.Mode.EXP): exp,
    str(mops.Elemwise.Mode.LOG): log,
    str(mops.Elemwise.Mode.GELU): gelu,
    str(mops.Elemwise.Mode.GELU_GRAD): gelu_grad,
    str(mops.Elemwise.Mode.TRUE_DIV): div,
    str(mops.Elemwise.Mode.NEGATE): neg,
    str(mops.Elemwise.Mode.ABS): abs,
    str(mops.Elemwise.Mode.ABS_GRAD): abs_grad,
    str(mops.Elemwise.Mode.TANH): tanh,
    str(mops.Elemwise.Mode.TANH_GRAD): tanh_grad,
    str(mops.Elemwise.Mode.SQRT): sqrt,
    str(mops.Elemwise.Mode.POW): pow,
    str(mops.Elemwise.Mode.RELU): relu,
    str(mops.Elemwise.Mode.EQ): equal,
    str(mops.Elemwise.Mode.NEQ): not_equal,
    str(mops.Elemwise.Mode.LT): less,
    str(mops.Elemwise.Mode.LEQ): less_equal,
    str(mops.Elemwise.Mode.SWITCH_GT0): relu_grad,
}


@register_lower_rule(mops.Elemwise)
def elemwise_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert isinstance(ctx.op, mops.Elemwise), "op should be elemwise here"
    assert (
        len(ctx.vars_out) == 1
    ), f"elemwise output num should be 1, got {len(ctx.vars_out)}"
    handle = mge_elemwise_to_xla[str(ctx.op.mode)]
    oup = handle(*args)
    return oup


@register_lower_rule(mops.ElemwiseMultiType)
def elemwise_multi_type_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    opr = ctx.op
    mode = "Elemwise.Mode." + str(opr.mode).split(".")[-1]
    handle = mge_elemwise_to_xla[mode]
    oup = handle(*args).astype(opr.dtype)
    return oup
