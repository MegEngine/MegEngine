from typing import Sequence, Union

import numpy as np

from ...core._imperative_rt import ops as mops
from .. import ir_utils
from ..lib.mlir.dialects import hlo
from .elemwise import sqrt
from .hlotensor import HLOTensor
from .reduction import _normalize_reduce_axes
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


def _normalize(x, axes, eps):
    """
    normalize X along axes, return normalized, mean, rstd
    for example:
        x.shape = (N, C, H, W), axes = (1, 2, 3)
        return normalized(N, C, H, W), mean(N,), rstd(N)
    """
    axes = _normalize_reduce_axes(x.shape, axes)
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

    non_reduce_shape = [x.shape[i] for i in range(x.ndim) if i not in axes]
    x_mean = x_mean.reshape(non_reduce_shape)
    rstd = rstd.reshape(non_reduce_shape)

    return normalized, x_mean, rstd


def _normalize_grad(dy, x, x_mean, rstd, axes):
    axes = _normalize_reduce_axes(x.shape, axes)

    if x_mean.ndim < x.ndim or rstd.ndim < x.ndim:
        reduced_shape = [x.shape[i] if i not in axes else 1 for i in range(x.ndim)]
        axes_divider = np.prod([x.shape[i] for i in axes]).astype("float32")

        x_mean = x_mean.reshape(reduced_shape)
        rstd = rstd.reshape(reduced_shape)

    delta = x - x_mean
    reducea = dy.sum(axes, True)
    b = (dy * delta * rstd ** 2.0).sum(axes, True)
    x1 = dy * rstd
    x2 = -1.0 * rstd / axes_divider * reducea
    x3 = -1.0 * rstd / axes_divider * delta * b
    x4 = rstd / axes_divider * (delta / axes_divider * b).sum(axes, True)
    dx = x1 + x2 + x3 + x4
    return dx, delta * rstd


def layer_norm(x, w, b, affine, normalized_dim, eps):
    reduce_axes = list(range(x.ndim - normalized_dim, x.ndim))
    y, x_mean, rstd = _normalize(x, reduce_axes, eps)
    if affine:
        y = y * w + b
    return y, x_mean, rstd


def layer_norm_grad(dy, x, w, x_mean, rstd, affine):
    reduce_axes = list(range(x_mean.ndim, x.ndim))
    scaled_dy = dy * w if w is not None else dy
    dx, dw_helper = _normalize_grad(scaled_dy, x, x_mean, rstd, reduce_axes)

    if affine:
        unreduce_axes = list(range(x.ndim - w.ndim))
        dbias = dy.sum(unreduce_axes, keepdims=False)
        dweight = (dw_helper * dy).sum(unreduce_axes, keepdims=False)
        return dx, dweight, dbias
    return dx


@register_lower_rule(mops.LayerNorm)
def layer_norm_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert ctx.op.normalized_dim > 0
    if ctx.op.affine:
        assert len(args) == 3 and len(ctx.vars_in) == 3 and len(ctx.vars_out) == 3
        x, w, b = args[0], args[1], args[2]
    else:
        assert len(args) == 1 and len(ctx.vars_in) == 1 and len(ctx.vars_out) == 3
        x, w, b = args[0], None, None

    return layer_norm(x, w, b, ctx.op.affine, ctx.op.normalized_dim, ctx.op.eps)


@register_lower_rule("LayerNormBackward")
def layer_norm_backward_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    if ctx.param["affine"]:
        dy, x, w, x_mean, rstd = args
    else:
        dy, x, x_mean, rstd = args
        w = None

    return layer_norm_grad(dy, x, w, x_mean, rstd, ctx.param["affine"])


def group_norm(x, w, b, affine, eps, group, fmt):
    assert x.ndim == 4, f"group/instance norm requires 4D input, get {x.shape}"
    N = x.shape[0]

    if fmt == "NCHW":
        grouped_shape = (N, group, -1)
        affine_shape = (1, -1, 1, 1)
        norm_axis = 2
    elif fmt == "NHWC":
        # grouped_shape = (N, -1, group)
        # affine_shape = (1, 1, 1, -1)
        # norm_axis = 1
        assert False, "NHWC format is not supported"
    else:
        assert False, f"invalid format: {fmt}"

    # reshape (N, C, H, W) to (N, Group, -1) and do normalize for each group
    grouped_x = x.reshape(grouped_shape)
    y, x_mean, rstd = _normalize(grouped_x, norm_axis, eps)
    y = y.reshape(x.shape)  # (N, Group, -1) -> (N, C, H, W)

    if affine:
        w = w.reshape(affine_shape)  # (C,) -> (1, C, 1, 1)
        b = b.reshape(affine_shape)  # (C,) -> (1, C, 1, 1)
        y = y * w + b

    return y, x_mean, rstd


def group_norm_grad(dy, x, w, x_mean, rstd, affine, group, fmt):
    assert x.ndim == 4, f"group/instance norm requires 4D input, get {x.shape}"
    if fmt == "NCHW":
        norm_axis = 2
        grouped_shape = (x.shape[0], group, -1)
        affine_axis = 1
        affine_shape = (1, -1, 1, 1)
    elif fmt == "NHWC":
        norm_axis = 1
        grouped_shape = (x.shape[0], -1, group)
        affine_axis = 1
        affine_shape = (1, 1, 1, -1)
        assert False, "not checked NHWC format"
    else:
        assert False, f"invalid format: {fmt}"

    scaled_dy = dy if w is None else dy * w.reshape(affine_shape)
    grouped_x = x.reshape(grouped_shape)  # (N, C, H, W) -> (N, Group, -1)
    grouped_dy = scaled_dy.reshape(grouped_shape)

    dx, dw_helper = _normalize_grad(grouped_dy, grouped_x, x_mean, rstd, norm_axis)
    dx = dx.reshape(x.shape)
    if affine:
        no_affine_axis = [i for i in range(dx.ndim) if i != affine_axis]
        db = dy.sum(no_affine_axis, False)
        dw = (dy * dw_helper.reshape(dy.shape)).sum(no_affine_axis, False)
        return dx, dw, db
    return dx


@register_lower_rule(mops.GroupNorm)
def group_norm_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    if ctx.op.affine:
        assert len(args) == 3 and len(ctx.vars_in) == 3 and len(ctx.vars_out) == 3
        x, w, b = args[0], args[1], args[2]
    else:
        assert len(args) == 1 and len(ctx.vars_in) == 1 and len(ctx.vars_out) == 3
        x, w, b = args[0], None, None

    assert x.ndim == 4, f"group norm requires 4D input, get {x.shape}"

    if ctx.op.format == mops.GroupNorm.Format.NCHW:
        return group_norm(x, w, b, ctx.op.affine, ctx.op.eps, ctx.op.group, "NCHW")
    elif ctx.op.format == mops.GroupNorm.Format.NHWC:
        return group_norm(x, w, b, ctx.op.affine, ctx.op.eps, ctx.op.group, "NHWC")
    else:
        assert False, f"invalid format: {ctx.op.format}"


@register_lower_rule("GroupNormBackward")
def group_norm_backward_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    if ctx.param["affine"] == True:
        assert (
            len(args) == 5 and len(ctx.vars_in) == 5 and len(ctx.vars_out) == 3
        ), f"{len(args)}, {len(ctx.vars_in)}, {len(ctx.vars_out)}"
        dy, x, w, x_mean, rstd = args
    else:
        assert (
            len(args) == 4 and len(ctx.vars_in) == 4 and len(ctx.vars_out) == 1
        ), f"{len(args)}, {len(ctx.vars_in)}, {len(ctx.vars_out)}"
        dy, x, x_mean, rstd = args
        w = None

    return group_norm_grad(
        dy,
        x,
        w,
        x_mean,
        rstd,
        ctx.param["affine"],
        ctx.param["group"],
        ctx.param["format"],
    )


@register_lower_rule(mops.InstanceNorm)
def instance_norm_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    if ctx.op.affine:
        assert len(args) == 3 and len(ctx.vars_in) == 3 and len(ctx.vars_out) == 3
        x, w, b = args[0], args[1], args[2]
    else:
        assert len(args) == 1 and len(ctx.vars_in) == 1 and len(ctx.vars_out) == 3
        x, w, b = args[0], None, None

    # instance norm is special group norm
    if ctx.op.format == mops.AdaptivePooling.Format.NCHW:
        return group_norm(x, w, b, ctx.op.affine, ctx.op.eps, x.shape[1], "NCHW")
    elif ctx.op.format == mops.AdaptivePooling.Format.NHWC:
        return group_norm(x, w, b, ctx.op.affine, ctx.op.eps, x.shape[-1], "NHWC")
    else:
        assert False, f"invalid format: {ctx.op.format}"


@register_lower_rule("InstanceNormBackward")
def instance_norm_backward_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    if ctx.param["affine"] == True:
        assert (
            len(args) == 5 and len(ctx.vars_in) == 5 and len(ctx.vars_out) == 3
        ), f"{len(args)}, {len(ctx.vars_in)}, {len(ctx.vars_out)}"
        dy, x, w, x_mean, rstd = args
    else:
        assert (
            len(args) == 4 and len(ctx.vars_in) == 4 and len(ctx.vars_out) == 1
        ), f"{len(args)}, {len(ctx.vars_in)}, {len(ctx.vars_out)}"
        dy, x, x_mean, rstd = args
        w = None

    # instance norm is special group norm
    group = x.shape[1] if ctx.param["format"] == "NCHW" else x.shape[-1]
    return group_norm_grad(
        dy, x, w, x_mean, rstd, ctx.param["affine"], group, ctx.param["format"]
    )
