from typing import Sequence, Union

import numpy as np

from ...core._imperative_rt import ops as mops
from ..lib.mlir import ir
from .hlotensor import HLOTensor
from .tensor import fill
from .utils import _check_shape, register_lower_rule


@register_lower_rule(mops.GetVarShape)
def get_var_shape_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    if len(args) > 1:
        assert len(args) == 2, f"{len(args)}"
        _check_shape(args[0].shape, args[1].shape)

    shp = args[0].shape
    if ctx.op.axis != 7:
        shp = (shp[ctx.op.axis],)

    shp = np.array(shp, np.int64)
    ctx.module_context.set_value(ctx.vars_out[0], shp)
    return HLOTensor(shp)


@register_lower_rule("create_tensor")
def create_tensor_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert len(args) == len(ctx.vars_in) == len(ctx.vars_out) == 1
    var_in, var_out = ctx.vars_in[0], ctx.vars_out[0]
    if var_in.bound_data is not None:
        ctx.module_context.set_value(var_in, var_in.bound_data)
        ctx.module_context.set_value(var_out, var_in.bound_data)
    assert var_in.shape == var_out.shape
    if var_out.bound_data is not None:
        data = np.asarray(var_out.bound_data, var_out.dtype)
    elif var_in.bound_data is not None:
        data = np.asarray(var_in.bound_data, var_out.dtype)
    else:
        assert False, "only support create tensor from const now"

    return HLOTensor(data)


@register_lower_rule("io_mark_var")
def io_mark_var_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert len(args) == 1
    return args


@register_lower_rule("rename")
def rename_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert len(args) == 1
    return args


@register_lower_rule("fake_op_rule_for_debug")
def fake_op_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    return [fill(0.0, out.shape, out.dtype) for out in ctx.vars_out]
