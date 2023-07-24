from typing import Sequence, Union

import numpy as np

from ...core._imperative_rt import ops as mops
from .. import ir_utils
from ..lib.mlir.dialects import hlo
from .hlotensor import HLOTensor
from .utils import (
    _can_broadcast_to,
    _check_shape,
    _parse_var_as_value,
    _shape_equal,
    register_lower_rule,
)


def broadcast_to(inp, oshape, broadcast_dims=None):
    """
    [x, y, z] or [x, 1, z] only can broadcast to [a, b, c, ..., x, y, z], rather than [x, y, z, ..., a, b, c]
    but you can realize the latter specify broadcast_dims, for example:
    broadcast_to([x, y, z], [x, y, z, ..., a, b, c], broadcast_dims=[0, 1, 2])

    broadcast_dims specify which dimension in the target shape each dimension of the
    operand shape corresponds to, for example:
    (1, 64, 1, 1) -> (32, 64, 32, 32), broadcast_dims is [0, 1, 2, 3]
    (16, 64, 32) -> (16, 64, 1, 32), broadcast_dims is [0, 1, 3]
    """
    inp = HLOTensor(inp) if not isinstance(inp, HLOTensor) else inp
    ishape, idtype = inp.shape, inp.dtype
    if _shape_equal(ishape, oshape):
        return inp

    assert _can_broadcast_to(
        ishape, oshape, broadcast_dims
    ), f"cannot broadcast {ishape} to {oshape} with broadcast_dims {broadcast_dims}"

    if broadcast_dims is None:
        broadcast_dims = list(range(len(oshape) - len(ishape), len(oshape)))

    result = hlo.BroadcastInDimOp(
        ir_utils.make_ir_type_according_meta(oshape, idtype),
        inp.tensor,
        ir_utils.dense_int_elements(broadcast_dims),
    ).result
    return HLOTensor(result, oshape, idtype)


def reshape(inp, oshape):
    if -1 in oshape:
        assert oshape.count(-1) == 1, f"invalid shape {oshape}"
        oshape = list(oshape)
        oshape[oshape.index(-1)] = int(np.abs(np.prod(inp.shape) / np.prod(oshape)))

    if _shape_equal(inp.shape, oshape):
        return inp

    assert np.prod(inp.shape) == np.prod(
        oshape
    ), f"cannot reshape {inp.shape} to {oshape}"

    return HLOTensor(
        hlo.ReshapeOp(
            ir_utils.make_ir_type_according_meta(oshape, inp.dtype), inp.tensor
        ).result,
        oshape,
        inp.dtype,
    )


def transpose(inp, permutation):
    assert len(inp.shape) == len(
        permutation
    ), f"incompatible shape and permutation: {inp.shape} vs {permutation}"
    return HLOTensor(
        hlo.TransposeOp(inp.tensor, ir_utils.dense_int_elements(permutation)).result
    )


def expand_dims(inp, axis):
    assert isinstance(axis, int), f"only int axis supported, get {axis}"
    assert (
        axis >= -inp.ndim - 1 and axis <= inp.ndim
    ), f"invalid axis {axis} for {inp.shape}"

    dst_shape = list(inp.shape)
    insert_pos = axis if axis >= 0 else (axis + inp.ndim + 1)
    dst_shape.insert(insert_pos, 1)

    return inp.reshape(tuple(dst_shape))


@register_lower_rule(mops.Dimshuffle)
def dim_shuffle_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert len(args) == 1 and len(ctx.vars_in) == 1 and len(ctx.vars_out) == 1
    # mge dimshuffle can do transpose and broadcast simutaneously
    # for example:
    #   case1: (16, 32, 64) with pattern [0, 2, 1] -> (16, 64, 32)
    #   case2: (16, 32, 64) with pattern [0, -1, 2, -1, 1] -> (16, 1, 64, 1, 32)
    #   case3: (16, 1, 64, 1, 32) with pattern [0, 4, 2] -> (16, 32, 64)

    pattern = ctx.op.pattern
    inp = args[0]
    if len(pattern) == inp.ndim:
        permutation = pattern
        return transpose(inp, permutation)
    elif len(pattern) > inp.ndim:
        permutation = [item for item in pattern if item != -1]
        return transpose(inp, permutation).reshape(ctx.vars_out[0].shape)
    else:
        permutation = [i for i in range(inp.ndim) if i not in pattern] + list(pattern)
        return transpose(inp, permutation).reshape(ctx.vars_out[0].shape)


def concat(inps, axis):
    assert len(inps) > 0, f"concat inputs should not be empty"
    if axis < 0:
        axis = axis + inps[0].ndim

    hlo_inps = [inp.tensor for inp in inps]

    return HLOTensor(hlo.ConcatenateOp(hlo_inps, ir_utils.i64_attr(axis)).results)


@register_lower_rule(mops.Concat, "Concat")
def concat_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert len(args) > 1 and isinstance(ctx.param["axis"], int)
    if ctx.param["axis"] < 0:
        axis = ctx.param["axis"] + len(ctx.vars_in[0].shape)
    else:
        axis = ctx.param["axis"]
    return concat(args, axis)


# if nsplit_or_sections is int, means divide inputs into nsplit_or_sections parts
# if nsplit_or_sections is Sequence[int], means divide inputs into
# len(nsplit_or_sections) parts, and the i-th part has nsplit_or_sections[i] elements
def split(inp, nsplit_or_sections, axis):
    from .indexing import index_with_slices

    ishape = inp.shape
    if axis < 0:
        axis = axis + len(ishape)

    if isinstance(nsplit_or_sections, int):
        dimlen = ishape[axis]
        assert dimlen % nsplit_or_sections == 0, "not an equal division"
        sections = [dimlen // nsplit_or_sections] * nsplit_or_sections
    else:
        sections = nsplit_or_sections

    assert np.sum(sections) == ishape[axis], "error split param"

    slices = []
    start = 0
    for section in sections:
        slices.append(
            [
                None if idx != axis else slice(start, start + section, 1)
                for idx in range(len(ishape))
            ]
        )
        start = start + section

    return [index_with_slices(inp, slices[i]) for i in range(len(sections))]


@register_lower_rule(mops.Split, "Split")
def split_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    nr_inp, nr_oup = len(ctx.vars_in), len(ctx.vars_out)
    assert len(args) == nr_inp and nr_inp == nr_oup + 1 and len(args) > 1

    assert isinstance(ctx.param["axis"], int)
    axis = ctx.param["axis"]

    sections = []
    for i in range(nr_oup):
        section = ctx.vars_out[i].shape[axis]
        if ctx.vars_in[i + 1].bound_data is not None:
            assert section == _parse_var_as_value(ctx.vars_in[i + 1])
        sections.append(section)

    return split(args[0], sections, axis)


def fill(value, shape, dtype):
    assert isinstance(value, (int, float, bool))
    value = np.asarray(value, dtype=dtype)
    return broadcast_to(HLOTensor(value, dtype=dtype), shape)


def iota(dtype, shape, dimension):
    """
    do some thing like arange.
    for example:
        shape = (2, 3), dimension=1, output is [[0, 1, 2], [0, 1, 2]]
        shape = (2, 3), dimension=-1, output is [[0, 0, 0], [1, 1, 1]]
    """
    dimension = dimension + len(shape) if dimension < 0 else dimension
    ret = hlo.IotaOp(
        ir_utils.make_ir_type_according_meta(shape, dtype), ir_utils.i64_attr(dimension)
    ).results
    assert len(ret) == 1, f"{len(ret)}"
    return HLOTensor(ret[0])


@register_lower_rule(mops.Fill)
def fill_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert len(args) == 1 and len(ctx.vars_in) == 1 and len(ctx.vars_out) == 1
    assert ctx.vars_out[0].dtype == ctx.op.dtype
    _check_shape(ctx.vars_out[0].shape, ctx.vars_in[0].bound_data)
    value = ctx.op.value
    dtype = ctx.vars_out[0].dtype
    shape = ctx.vars_out[0].shape
    return fill(value, shape, dtype)


@register_lower_rule(mops.FillLike)
def fill_like_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert len(args) == 1 and len(ctx.vars_in) == 1 and len(ctx.vars_out) == 1
    var_in, var_out, opr = ctx.vars_in[0], ctx.vars_out[0], ctx.op
    value = opr.value

    assert _shape_equal(var_in.shape, var_out.shape) and _shape_equal(
        var_out.shape, args[0].shape
    )
    assert var_in.dtype == var_out.dtype and args[0].dtype == var_out.dtype
    shape, dtype = var_out.shape, var_out.dtype

    return fill(value, shape, dtype)


def pad(inp, pad_value, padding):
    # interior is used as dilated padding if it is not zero
    assert isinstance(
        pad_value, (int, float, bool, np.ndarray)
    ), f"pad_value error {type(pad_value)}"
    pad_value = HLOTensor(pad_value, dtype=inp.dtype)

    low, high, interior = [], [], []
    for p in padding:
        assert len(p) == 3
        low.append(p[0])
        high.append(p[1])
        interior.append(p[2])

    return HLOTensor(
        hlo.PadOp(
            inp.tensor,
            pad_value.tensor,
            ir_utils.dense_int_elements(low),
            ir_utils.dense_int_elements(high),
            ir_utils.dense_int_elements(interior),
        ).result
    )


def where(mask, x, y):
    mask = mask.astype("float32")
    return mask * x + (1.0 - mask) * y


@register_lower_rule(mops.Reshape)
def reshape_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert len(args) == 2
    return args[0].reshape(ctx.vars_out[0].shape)


@register_lower_rule(mops.RemoveAxis)
def remove_axis_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert len(args) == 1
    return args[0].reshape(ctx.vars_out[0].shape)


@register_lower_rule(mops.AddAxis)
def add_axis_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert len(args) == 1
    return args[0].reshape(ctx.vars_out[0].shape)


@register_lower_rule(mops.TypeCvt)
def typecvt_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    return args[0].astype(ctx.vars_out[0].dtype)


@register_lower_rule("TypeCvtV2")
def typecvt_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    return args[0].astype(ctx.vars_out[0].dtype)


@register_lower_rule(mops.Broadcast)
def broadcast_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    return args[0].broadcast_to(ctx.vars_out[0].shape)


@register_lower_rule(mops.Copy, mops.Identity)
def copy_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    return args
