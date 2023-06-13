from typing import Sequence, Union

import numpy as np

from ...core._imperative_rt import ops as mops
from .. import ir_utils
from ..lib.mlir.dialects import hlo
from .elemwise import sqrt
from .hlotensor import HLOTensor
from .utils import register_lower_rule


@register_lower_rule(mops.BatchNorm)
def batch_norm_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    if ctx.op.fwd_mode == mops.BatchNorm.FwdMode.TRAINING:
        # training mode will return the new running mean and var, so return 6 args
        assert (
            len(args) == 5 and len(ctx.vars_in) == 5 and len(ctx.vars_out) == 6
        ), f"len(args): {len(args)}, len(ctx.vars_in): {len(ctx.vars_in)}, len(ctx.vars_out): {len(ctx.vars_out)}"
    else:
        assert ctx.op.fwd_mode == mops.BatchNorm.FwdMode.INFERENCE, f"{ctx.op.fwd_mode}"
        # inference mode will not return the new running mean and var, so return 4 args
        assert (
            len(args) == 5 and len(ctx.vars_in) == 5 and len(ctx.vars_out) == 4
        ), f"len(args): {len(args)}, len(ctx.vars_in): {len(ctx.vars_in)}, len(ctx.vars_out): {len(ctx.vars_out)}"

    assert ctx.op.param_dim == "DIM_1C11", f"ctx.op.param_dim: {ctx.op.param_dim}"

    channel_dim = 1  # because param_dim is DIM_1C11
    C = args[1].shape[channel_dim]

    inp, weight, bias, running_mean, running_var = (
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
    )
    unused = HLOTensor(
        np.random.random(ctx.vars_out[-2].shape).astype(ctx.vars_out[-2].dtype)
    )

    if ctx.op.fwd_mode == mops.BatchNorm.FwdMode.TRAINING:
        rst = hlo.BatchNormTrainingOp(
            inp.tensor,
            weight.reshape((C,)).tensor,
            bias.reshape((C,)).tensor,
            ir_utils.f32_attr(ctx.op.epsilon),
            ir_utils.i64_attr(channel_dim),
        ).results
        assert len(rst) == 3, f"len(rst): {len(rst)}"
        oup, batch_mean, batch_var = (
            HLOTensor(rst[0]),
            HLOTensor(rst[1]).reshape((1, C, 1, 1)),
            HLOTensor(rst[2]).reshape((1, C, 1, 1)),
        )

        running_mean = (
            running_mean * (1 - ctx.op.avg_factor) + batch_mean * ctx.op.avg_factor
        )
        running_var = (
            running_var * (1 - ctx.op.avg_factor) + batch_var * ctx.op.avg_factor
        )
        return running_mean, running_var, batch_mean, batch_var, unused, oup
    else:
        rst = hlo.BatchNormInferenceOp(
            inp.tensor,
            weight.reshape((C,)).tensor,
            bias.reshape((C,)).tensor,
            running_mean.reshape((C,)).tensor,
            running_var.reshape((C,)).tensor,
            ir_utils.f32_attr(ctx.op.epsilon),
            ir_utils.i64_attr(channel_dim),
        ).results
        assert len(rst) == 1, f"len(rst): {len(rst)}"
        oup = HLOTensor(rst[0])
        return running_mean, running_var, unused, oup


@register_lower_rule(mops.BatchNormBackward)
def batch_norm_backward_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert (
        len(args) == 6 and len(ctx.vars_in) == 6 and len(ctx.vars_out) == 3
    ), f"len(args): {len(args)}, len(ctx.vars_in): {len(ctx.vars_in)}, len(ctx.vars_out): {len(ctx.vars_out)}"
    assert (
        ctx.op.fwd_mode == "TRAINING" and ctx.op.param_dim == "DIM_1C11"
    ), f"ctx.op.fwd_mode: {ctx.op.fwd_mode}, ctx.op.param_dim: {ctx.op.param_dim}"
    channel_dim = 1  # because param_dim is DIM_1C11
    C = args[4].shape[channel_dim]

    inp, grad, mean, var, weight = (
        args[0],
        args[1],
        args[2].reshape((C,)),
        args[3].reshape((C,)),
        args[4].reshape((C,)),
    )
    rst = hlo.BatchNormGradOp(
        inp.tensor,
        weight.tensor,
        mean.tensor,
        var.tensor,
        grad.tensor,
        ir_utils.f32_attr(ctx.op.epsilon),
        ir_utils.i64_attr(channel_dim),
    ).results
    return [
        HLOTensor(rst[1]).reshape(ctx.vars_out[0].shape),
        HLOTensor(rst[2]).reshape(ctx.vars_out[1].shape),
        HLOTensor(rst[0]),
    ]


def _normalize_lower(
    x, axes, affine, eps, w=None, b=None,
):
    x_mean = x.mean(axes, True)
    x_mean_sqr = x_mean * x_mean
    x_sqr = x * x
    x_sqr_mean = x_sqr.mean(axes, True)
    var = x_sqr_mean - x_mean_sqr
    var_plus_eps = var + eps
    std = sqrt(var_plus_eps)
    rstd = 1.0 / std
    delta = x - x_mean
    normalized = delta * rstd

    if affine:
        normalized = normalized * w + b

    return [normalized, x_mean, rstd]


@register_lower_rule(mops.LayerNorm)
def layer_norm_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert ctx.op.normalized_dim > 0
    if ctx.op.affine:
        assert len(args) == 3 and len(ctx.vars_in) == 3 and len(ctx.vars_out) == 3
        x, w, b = args[0], args[1], args[2]
    else:
        assert len(args) == 1 and len(ctx.vars_in) == 1 and len(ctx.vars_out) == 3
        x, w, b = args[0], None, None

    axes = list(range(x.ndim - ctx.op.normalized_dim, x.ndim))
    rets = _normalize_lower(x, axes, ctx.op.affine, ctx.op.eps, w, b,)
    rets[1] = rets[1].reshape(ctx.vars_out[1].shape)
    rets[2] = rets[2].reshape(ctx.vars_out[2].shape)
    return rets


@register_lower_rule("LayerNormBackward")
def layer_norm_backward_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    grad = args[0]
    if ctx.param["affine"]:
        inp = args[1]
        weight = args[2]
        mean = args[3]
        rstd = args[4]
    else:
        inp = args[1]
        mean = args[2]
        rstd = args[3]

    reduced_shape = mean.shape + (1,) * (inp.ndim - mean.ndim)
    reduce_axes = list(range(mean.ndim, inp.ndim))
    axes_divider = np.prod([inp.shape[i] for i in reduce_axes]).astype("float32")

    mean = mean.reshape(reduced_shape)
    rstd = rstd.reshape(reduced_shape)
    delta = inp - mean
    a = grad * weight if ctx.param["affine"] else grad
    reducea = a.sum(reduce_axes, True)
    b = (a * delta * rstd ** 2.0).sum(reduce_axes, True)
    x1 = a * rstd
    x2 = -1.0 * rstd / axes_divider * reducea
    x3 = -1.0 * rstd / axes_divider * (inp - mean) * b
    x4 = rstd / axes_divider * ((inp - mean) / axes_divider * b).sum(reduce_axes, True)
    dx = x1 + x2 + x3 + x4

    if ctx.param["affine"]:
        unreduce_axes = list(range(inp.ndim - weight.ndim))
        dbias = grad.sum(unreduce_axes, keepdims=False)
        dweight = (delta * rstd * grad).sum(unreduce_axes, keepdims=False)
        return [dx, dweight, dbias]
    return [dx]
