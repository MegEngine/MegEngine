import math
from functools import partial
from typing import Sequence, Union

import numpy as np

from ...core._imperative_rt import ops as mops
from .. import ir_utils
from ..lib.mlir.dialects import chlo, hlo
from .hlotensor import HLOTensor
from .utils import register_lower_rule


def _infer_elemwise_oshape(inp_shapes):
    def _infer_binary_elemwise_oshape(lhs_shape, rhs_shape):
        if len(lhs_shape) == 0:
            return rhs_shape
        if len(rhs_shape) == 0:
            return lhs_shape

        if np.prod(lhs_shape) == 1 and len(lhs_shape) == 1 and len(rhs_shape) != 0:
            return rhs_shape
        if np.prod(rhs_shape) == 1 and len(rhs_shape) == 1 and len(rhs_shape) != 0:
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


def _elemwise_ternary(hlo_op, a, b, c):
    return _elemwise(hlo_op, [a, b, c])


neg = partial(_elemwise_unary, hlo.NegOp)
abs = partial(_elemwise_unary, hlo.AbsOp)
sin = partial(_elemwise_unary, hlo.SineOp)
cos = partial(_elemwise_unary, hlo.CosineOp)
tan = partial(_elemwise_unary, chlo.TanOp)
asin = partial(_elemwise_unary, chlo.AsinOp)
acos = partial(_elemwise_unary, chlo.AcosOp)
atan = partial(_elemwise_unary, chlo.AtanOp)
sinh = partial(_elemwise_unary, chlo.SinhOp)
cosh = partial(_elemwise_unary, chlo.CoshOp)
tanh = partial(_elemwise_unary, hlo.TanhOp)
asinh = partial(_elemwise_unary, chlo.AsinhOp)
acosh = partial(_elemwise_unary, chlo.AcoshOp)
atanh = partial(_elemwise_unary, chlo.AtanhOp)
erf = partial(_elemwise_unary, chlo.ErfOp)
exp = partial(_elemwise_unary, hlo.ExpOp)
sqrt = partial(_elemwise_unary, hlo.SqrtOp)
log = partial(_elemwise_unary, hlo.LogOp)
log1p = partial(_elemwise_unary, hlo.Log1pOp)
expm1 = partial(_elemwise_unary, hlo.Expm1Op)
floor = partial(_elemwise_unary, hlo.FloorOp)
ceil = partial(_elemwise_unary, hlo.CeilOp)
round = partial(_elemwise_unary, hlo.RoundOp)

add = partial(_elemwise_binary, hlo.AddOp)
sub = partial(_elemwise_binary, hlo.SubtractOp)
mul = partial(_elemwise_binary, hlo.MulOp)
div = partial(_elemwise_binary, hlo.DivOp)
pow = partial(_elemwise_binary, hlo.PowOp)
maximum = partial(_elemwise_binary, hlo.MaxOp)
minimum = partial(_elemwise_binary, hlo.MinOp)
atan2 = partial(_elemwise_binary, hlo.Atan2Op)
left_shift = partial(_elemwise_binary, hlo.ShiftLeftOp)
right_shift = partial(_elemwise_binary, hlo.ShiftRightArithmeticOp)

clip = partial(_elemwise_ternary, hlo.ClampOp)

equal = partial(_compare, mode="EQ")
not_equal = partial(_compare, mode="NE")
greater = partial(_compare, mode="GT")
greater_equal = partial(_compare, mode="GE")
less = partial(_compare, mode="LT")
less_equal = partial(_compare, mode="LE")

logical_and = partial(_elemwise_binary, hlo.AndOp)
logical_or = partial(_elemwise_binary, hlo.OrOp)
logical_not = partial(_elemwise_unary, hlo.NotOp)
logical_xor = partial(_elemwise_binary, hlo.XorOp)


def floor_div(x, y):
    return floor(div(x, y))


def mod(x, y):
    assert False, "xla not support"


def cond_leq_move(x, y, z):
    mask = (x <= y).astype(x.dtype)
    return mask * z


def cond_lt_move(x, y, z):
    mask = (x < y).astype(x.dtype)
    return mask * z


def log_add_exp(x, y):
    min_val = minimum(x, y)
    max_val = maximum(x, y)
    return max_val + log1p(exp(min_val - max_val))


def square(x):
    return mul(x, x)


def abs_grad(x, dy):
    return (x / abs(x)) * dy


def tan_grad(x, dy):
    return (1.0 + tan(x) ** 2.0) * dy


def tanh_grad(x, dy):
    return (1.0 - tanh(x) ** 2.0) * dy


def asinh_grad(x, dy):
    return dy / sqrt(x ** 2.0 + 1.0)


def acosh_grad(x, dy):
    return dy / sqrt(x ** 2.0 - 1.0)


def atanh_grad(x, dy):
    return dy / (1.0 - x ** 2.0)


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


def gelu_grad(x, dy, approximate: bool = True):
    if approximate:
        _a = x * x
        _b = -0.5 * _a
        _c = exp(_b)
        phi = 0.3989422804014327 * _c
        _d = x / math.sqrt(2.0)
        _e = erf(_d)
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


def sigmoid(inp):
    return 1.0 / (1.0 + exp(-inp))


def sigmoid_grad(y, dy):
    return y * (1.0 - y) * dy


def hsigmoid(x):
    from .tensor import where

    return where(x <= -3.0, 0.0, where(x >= 3.0, 1.0, (x + 3.0) / 6.0))


def hsigmoid_grad(x, dy):
    from .tensor import where

    return where(x <= -3.0, 0.0, where(x >= 3.0, 0.0, dy / 6.0))


def relu6(x):
    return clip(x, 0.0, 6.0)


def relu6_grad(x, dy):
    from .tensor import where

    return where(x <= 0.0, 0.0, where(x >= 6.0, 0.0, dy))


def hswish(x):
    return x * minimum(maximum(x + 3.0, 0.0), 6.0) * (1.0 / 6.0)


def hswish_grad(x, dy):
    from .tensor import where

    return where(x < -3.0, 0.0, where(x > 3.0, dy, (2.0 * x + 3.0) / 6.0 * dy))


def logsigmoid(x):
    from .tensor import where

    return -log1p(exp(-abs(x))) + where(x >= 0.0, 0.0, x)


def softplus(x):
    return log1p(exp(-abs(x))) + relu(x)


def softplus_grad(x, dy):
    from .tensor import where

    exp_abs = exp(-abs(x))
    logg = -dy * exp_abs / (1.0 + exp_abs)
    grad0 = where(x > 0.0, logg, -logg)
    relux = relu(x)
    grad1 = where(relux > 0.0, dy, 0.0)
    return grad0 + grad1


def prelu(inp, alpha):
    mask = (inp > 0.0).astype(inp.dtype)
    return inp * mask + alpha * (1.0 - mask) * inp


def prelu_grad(x, dy, alpha):
    mask = (x > 0.0).astype(x.dtype)
    return dy * mask + alpha * (1.0 - mask) * dy


def silu(inp):
    return inp / (1.0 + exp(-inp))


def silu_grad(x, dy):
    xsig = sigmoid(x)
    return dy * xsig * (1.0 + x * (1.0 - xsig))


# Elemwise.Mode is unhashable, so we convert it to str
mge_elemwise_to_xla = {
    str(mops.Elemwise.Mode.ADD): add,
    str(mops.Elemwise.Mode.MUL): mul,
    str(mops.Elemwise.Mode.SUB): sub,
    str(mops.Elemwise.Mode.EXP): exp,
    str(mops.Elemwise.Mode.LOG): log,
    str(mops.Elemwise.Mode.LOG1P): log1p,
    str(mops.Elemwise.Mode.LOG_SUM_EXP): log_add_exp,
    str(mops.Elemwise.Mode.MAX): maximum,
    str(mops.Elemwise.Mode.MIN): minimum,
    str(mops.Elemwise.Mode.COND_LEQ_MOV): cond_leq_move,
    str(mops.Elemwise.Mode.COND_LT_MOV): cond_lt_move,
    str(mops.Elemwise.Mode.FLOOR): floor,
    str(mops.Elemwise.Mode.CEIL): ceil,
    str(mops.Elemwise.Mode.ROUND): round,
    str(mops.Elemwise.Mode.CLIP): clip,
    str(mops.Elemwise.Mode.GELU): gelu,
    str(mops.Elemwise.Mode.GELU_GRAD): gelu_grad,
    str(mops.Elemwise.Mode.TRUE_DIV): div,
    str(mops.Elemwise.Mode.NEGATE): neg,
    str(mops.Elemwise.Mode.FLOOR_DIV): floor_div,
    str(mops.Elemwise.Mode.MOD): mod,
    str(mops.Elemwise.Mode.ABS): abs,
    str(mops.Elemwise.Mode.ABS_GRAD): abs_grad,
    str(mops.Elemwise.Mode.SIN): sin,
    str(mops.Elemwise.Mode.COS): cos,
    str(mops.Elemwise.Mode.TAN): tan,
    str(mops.Elemwise.Mode.SINH): sinh,
    str(mops.Elemwise.Mode.COSH): cosh,
    str(mops.Elemwise.Mode.TANH): tanh,
    str(mops.Elemwise.Mode.ASIN): asin,
    str(mops.Elemwise.Mode.ACOS): acos,
    str(mops.Elemwise.Mode.ASINH): asinh,
    str(mops.Elemwise.Mode.ACOSH): acosh,
    str(mops.Elemwise.Mode.ATANH): atanh,
    str(mops.Elemwise.Mode.ATAN2): atan2,
    str(mops.Elemwise.Mode.TANH_GRAD): tanh_grad,
    str(mops.Elemwise.Mode.ASINH_GRAD): asinh_grad,
    str(mops.Elemwise.Mode.ACOSH_GRAD): acosh_grad,
    str(mops.Elemwise.Mode.ATANH_GRAD): atanh_grad,
    str(mops.Elemwise.Mode.SQRT): sqrt,
    str(mops.Elemwise.Mode.SQUARE): square,
    str(mops.Elemwise.Mode.POW): pow,
    str(mops.Elemwise.Mode.EXPM1): expm1,
    str(mops.Elemwise.Mode.RELU): relu,
    str(mops.Elemwise.Mode.EQ): equal,
    str(mops.Elemwise.Mode.NEQ): not_equal,
    str(mops.Elemwise.Mode.LT): less,
    str(mops.Elemwise.Mode.LEQ): less_equal,
    str(mops.Elemwise.Mode.AND): logical_and,
    str(mops.Elemwise.Mode.OR): logical_or,
    str(mops.Elemwise.Mode.NOT): logical_not,
    str(mops.Elemwise.Mode.XOR): logical_xor,
    str(mops.Elemwise.Mode.SHL): left_shift,
    str(mops.Elemwise.Mode.SHR): right_shift,
    str(mops.Elemwise.Mode.SWITCH_GT0): relu_grad,
    str(mops.Elemwise.Mode.SIGMOID): sigmoid,
    str(mops.Elemwise.Mode.SIGMOID_GRAD): sigmoid_grad,
    str(mops.Elemwise.Mode.PRELU): prelu,
    str(mops.Elemwise.Mode.PRELU_GRAD): prelu_grad,
    str(mops.Elemwise.Mode.SILU): silu,
    str(mops.Elemwise.Mode.SILU_GRAD): silu_grad,
    str(mops.Elemwise.Mode.HSIGMOID): hsigmoid,
    str(mops.Elemwise.Mode.HSIGMOID_GRAD): hsigmoid_grad,
    str(mops.Elemwise.Mode.H_SWISH): hswish,
    str(mops.Elemwise.Mode.H_SWISH_GRAD): hswish_grad,
    str(mops.Elemwise.Mode.RELU6): relu6,
    str(mops.Elemwise.Mode.RELU6_GRAD): relu6_grad,
    str(mops.Elemwise.Mode.LOGSIGMOID): logsigmoid,
    str(mops.Elemwise.Mode.SOFTPLUS): softplus,
    str(mops.Elemwise.Mode.SOFTPLUS_GRAD): softplus_grad,
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
