from collections import namedtuple
from enum import IntEnum
from typing import Any, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from ...core._imperative_rt import ops as mops
from .. import ir_utils
from ..lib.mlir import ir
from ..lib.mlir.dialects import hlo
from .hlotensor import HLOTensor
from .tensor import xla_gather, xla_scatter
from .utils import _parse_var_as_value, register_lower_rule


def _is_canonicalized_axis(sl: slice, axis_len: int):
    return (
        (0 <= sl.start and sl.start < axis_len)
        and (0 <= sl.stop and sl.stop <= axis_len)
        and (0 < sl.step)
    )


def _canonicalize_slice_with_axis_len(sl: slice, axis_len: int):
    """
    make slice canonicalized: 0 <= sl.start < axis_len and 0 <= sl.stop <= axis_len
    """

    def impl(idx, axis_len):
        if idx < 0:
            idx = idx + axis_len
        assert idx >= 0 and idx <= axis_len, f"{idx}, {axis_len}"
        if idx < 0:
            idx = 0
        if idx > axis_len:
            idx = axis_len
        return idx

    assert isinstance(sl, slice)
    start = impl(sl.start, axis_len)
    stop = impl(sl.stop, axis_len)

    new_sl = slice(start, stop, sl.step)

    assert new_sl.step > 0, "step <= 0 is not supported now"
    assert _is_canonicalized_axis(
        new_sl, axis_len
    ), f"slice {new_sl} is illegal for axis whose length is {axis_len}"
    return new_sl


def _hslice_with_step_is_one(inp, slices):
    """
    if inp_shape is N-dim, slices should contain N slice, slice can not None.
    for shape [12, 15], slices can be [slice(0, 3, 1), slice(12, 15, 1)].
    the step of slice should must be 1
    """
    assert all([sl.step == 1 for sl in slices])
    starts = [int(sl.start) for sl in slices]
    slice_sizes = [int(max(0, sl.stop - sl.start)) for sl in slices]

    starts = [ir_utils.ir_constant(si) for si in starts]
    slice_sizes = ir_utils.dense_int_elements(slice_sizes)

    return hlo.DynamicSliceOp(inp, starts, slice_sizes).results


def _hslice_with_any_step(inp, slices):
    """
    if inp_shape is N-dim, slices should contain N slice, slice can not None.
    for shape [12, 15], slices can be [slice(0, 3, 1), slice(12, 15, 1)]
    """
    starts = [int(sl.start) for sl in slices]
    stops = [int(sl.stop) for sl in slices]
    steps = [int(sl.step) for sl in slices]

    return hlo.SliceOp(
        inp,
        ir_utils.dense_int_elements(starts),
        ir_utils.dense_int_elements(stops),
        ir_utils.dense_int_elements(steps),
    ).results


def index_with_slices(inp, slices):
    """
    if inp_shape is N-dim, slices should contain N slice, slice can be None.
    for shape [12, 15], slices can be [slice(0, 3, 1), slice(12, 15, 1)] or [None, None]
    """
    assert isinstance(slices, Sequence), f"{slices}"
    assert len(inp.shape) >= len(slices), f"{inp.shape}, {slices}"
    slices = list(slices) + [None,] * (len(inp.shape) - len(slices))

    slices = [
        sl if sl is not None else slice(0, axis_len, 1)
        for (sl, axis_len) in zip(slices, inp.shape)
    ]
    slices = [
        _canonicalize_slice_with_axis_len(sl, axis_len)
        for (sl, axis_len) in zip(slices, inp.shape)
    ]

    all_step_is_one = all(sl.step == 1 for sl in slices)
    if all_step_is_one:
        return HLOTensor(_hslice_with_step_is_one(inp.tensor, slices))
    else:
        return HLOTensor(_hslice_with_any_step(inp.tensor, slices))


def _parse_subtensor_items(items, dst_shape, idx_vars, idx_hlotensors):
    def parse_idx_var(var, hlotensor) -> Union[int, HLOTensor]:
        if isinstance(var.bound_data, int):
            return var.bound_data
        if isinstance(var.bound_data, np.ndarray):
            assert var.bound_data.shape == (1,)
            return int(var.bound_data[0])
        if var.bound_data is None:
            return hlotensor
        else:
            assert False, var.bound_data

    SliceMeta = namedtuple("SliceMeta", "start stop step isidx")

    raw_slices = [SliceMeta(0, dst_shape[i], 1, False) for i in range(len(dst_shape))]
    inp_offset = 0
    for (axis, has_start, has_stop, has_step, is_idx) in items:
        axis_len = dst_shape[axis]
        is_slice = has_start or has_stop or has_step
        # slice like x[1:3], idx like x[1]
        assert is_slice ^ is_idx, f"cannot specify index idx and slice simultaneously"

        start, stop, step = 0, axis_len, 1
        if is_slice:
            if has_start:
                start = parse_idx_var(idx_vars[inp_offset], idx_hlotensors[inp_offset])
                inp_offset += 1
            if has_stop:
                stop = parse_idx_var(idx_vars[inp_offset], idx_hlotensors[inp_offset])
                inp_offset += 1
            if has_step:
                step = parse_idx_var(idx_vars[inp_offset], idx_hlotensors[inp_offset])
                inp_offset += 1

        elif is_idx:
            idx = parse_idx_var(idx_vars[inp_offset], idx_hlotensors[inp_offset])
            inp_offset += 1
            start, stop = idx, idx + 1

        else:
            assert False

        raw_slices[axis] = SliceMeta(start, stop, step, is_idx)

    # canonicalize slice
    for i, (start, stop, step, is_idx) in enumerate(raw_slices):
        assert all(
            [isinstance(x, (int, HLOTensor)) for x in [start, stop, step]]
        ), f"must be int/tensor, get {start}, {stop}, {step}"

        if isinstance(step, int):
            assert step >= 0, f"xla not support slice with step < 0, get {step}"

        from .tensor import where

        start = (
            slice(start, None, None).indices(dst_shape[i])[0]
            if isinstance(start, int)
            else where(start < 0, start + dst_shape[i], start)
        )
        stop = (
            slice(None, stop, None).indices(dst_shape[i])[1]
            if isinstance(stop, int)
            else where(stop < 0, stop + dst_shape[i], stop)
        )
        step = (
            slice(None, None, step).indices(dst_shape[i])[2]
            if isinstance(step, int)
            else step
        )

        raw_slices[i] = SliceMeta(start, stop, step, is_idx)

    return raw_slices


@register_lower_rule(mops.Subtensor)
def subtensor_lower(
    ctx, *args: Union[ir.Value, Sequence[ir.Value]], explicit_type=False
):
    assert len(ctx.op.slice_items) == 0 and len(ctx.vars_out) == 1
    opr, inp, inp_shape = ctx.op, args[0], ctx.vars_in[0].shape
    raw_slices = _parse_subtensor_items(opr.items, inp_shape, ctx.vars_in[1:], args[1:])
    slices = [slice(start, stop, step) for start, stop, step, _ in raw_slices]
    any_axis_is_index = any([isidx for _, _, _, isidx in raw_slices])
    assert len(slices) == inp.ndim, f"{len(slices)}, {inp.ndim}"

    oup = index_with_slices(inp, slices)

    if any_axis_is_index:
        return oup.reshape(ctx.vars_out[0].shape)
    else:
        return oup


# to understand the following code, you should read the docstring of xla_scatter/xla_gather
def _get_scatter_configs_and_indices(
    dst_shape, src_shape, raw_slices, index_dtype=np.int32
):
    # the src.ndim cannot match dst.ndim, some axis of src maybe collapsed
    # the dst_axis/src_axis is the current analyzed axis of dst/src before collpased
    # collapsed_src_axis is the current axis of src after collapsed
    dst_axis, src_axis, collapsed_src_axis = 0, 0, 0

    # you should read the docstring of xla_scatter/xla_gather to understand these
    # meaning of these variables
    offset_dims: Sequence[int] = []
    collapsed_slice_dims: Sequence[int] = []
    start_index_map: Sequence[int] = []

    # Pairs of (array, start_dim) values. These will be broadcast into
    # gather_indices_shape, with the array dimensions aligned to start_dim, and
    # then concatenated.
    gather_indices: List[Tuple[Sequence, int]] = []
    gather_indices_shape: List[int] = []
    # the basic unit of xla scatter is a slice, slice_shape describe the shape of slice
    slice_shape: Sequence[int] = []
    gather_slice_shape: Sequence[int] = []

    for start, stop, step, isidx in raw_slices:
        # process index
        if isidx:
            start_arr = (
                np.array(start, index_dtype) if isinstance(start, int) else start
            )
            gather_indices.append((start_arr, len(gather_indices_shape)))
            collapsed_slice_dims.append(dst_axis)
            gather_slice_shape.append(1)
            start_index_map.append(dst_axis)
            dst_axis += 1
            continue

        # slice like slice(None, None, None)
        if start is None and stop is None and step is None:
            slice_shape.append(dst_shape[dst_axis])
            gather_slice_shape.append(dst_shape[dst_axis])
            offset_dims.append(collapsed_src_axis)
            collapsed_src_axis += 1
            src_axis += 1
            dst_axis += 1
            continue

        # slice like slice(any, any, 1)
        if isinstance(step, int) and step == 1:
            start_arr = (
                np.array(start, index_dtype) if isinstance(start, int) else start
            )
            gather_indices.append((start_arr, len(gather_indices_shape)))
            if isinstance(stop, int) and isinstance(start, int):
                slice_shape.append(stop - start)
                gather_slice_shape.append(stop - start)
            else:
                slice_shape.append(src_shape[src_axis])
                gather_slice_shape.append(src_shape[src_axis])
            offset_dims.append(collapsed_src_axis)
            start_index_map.append(dst_axis)
            collapsed_src_axis += 1
            src_axis += 1
            dst_axis += 1
            continue

        # general case
        if all([isinstance(x, int) for x in [start, stop, step]]):
            indices = np.arange(start, stop, step, dtype=index_dtype)
        else:
            from .tensor import arange

            indices = arange(
                start, stop, step, num=src_shape[src_axis], dtype=index_dtype,
            )

        size = indices.shape[0]
        slice_shape.append(size)
        gather_slice_shape.append(1)
        gather_indices.append((indices, len(gather_indices_shape)))
        gather_indices_shape.append(size)

        start_index_map.append(dst_axis)
        collapsed_slice_dims.append(dst_axis)
        collapsed_src_axis += 1
        src_axis += 1
        dst_axis += 1

    if len(gather_indices) == 0:
        gather_indices_array = np.zeros((0,), dtype=index_dtype)
    elif len(gather_indices) == 1:
        g, _ = gather_indices[0]
        if isinstance(g, np.ndarray):
            gather_indices_array = np.expand_dims(g, (g.ndim,))
        else:
            from .tensor import expand_dims

            gather_indices_array = expand_dims(g, g.ndim)
    else:
        last_dim = len(gather_indices_shape)
        gather_indices_shape.append(1)

        def _broadcast_to(src, tgt_shape, axises):
            if isinstance(src, np.ndarray):
                src_shape = src.shape
                expanded_src_shape = [1,] * len(tgt_shape)
                for i, ax in enumerate(axises):
                    expanded_src_shape[ax] = src_shape[i]
                src = np.reshape(src, expanded_src_shape)
                return np.broadcast_to(src, tgt_shape)
            else:
                from .tensor import broadcast_to

                return broadcast_to(src, tgt_shape, axises)

        if all([isinstance(g, np.ndarray) for g, _ in gather_indices]):
            gather_indices_array = HLOTensor(
                np.concatenate(
                    [
                        _broadcast_to(
                            g, gather_indices_shape, tuple(range(i, i + g.ndim))
                        )
                        for g, i in gather_indices
                    ],
                    last_dim,
                )
            )
        else:
            from .tensor import concat

            gather_indices = [
                (g, i) if isinstance(g, HLOTensor) else (HLOTensor(g), i)
                for (g, i) in gather_indices
            ]
            gather_indices_array = concat(
                [
                    _broadcast_to(g, gather_indices_shape, tuple(range(i, i + g.ndim)))
                    for g, i in gather_indices
                ],
                last_dim,
            )
    return (
        offset_dims,
        collapsed_slice_dims,
        start_index_map,
        gather_indices_array,
        slice_shape,
    )


@register_lower_rule(mops.SetSubtensor)
def setsubtensor_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert len(ctx.vars_out) == 1
    opr, dst, src = ctx.op, args[0], args[1]  # dst[indices] = src

    raw_slices = _parse_subtensor_items(opr.items, dst.shape, ctx.vars_in[2:], args[2:])
    (
        update_window_dims,
        inserted_window_dims,
        scattered_dims_to_operand_dims,
        indices,
        slice_shape,
    ) = _get_scatter_configs_and_indices(dst.shape, src.shape, raw_slices)

    if len(slice_shape) == 0 or np.prod(slice_shape) == 0:
        return [dst]

    src = src.broadcast_to(slice_shape)

    out = xla_scatter(
        dst,
        indices,
        src,
        update_window_dims=update_window_dims,
        inserted_window_dims=inserted_window_dims,
        scattered_dims_to_operand_dims=scattered_dims_to_operand_dims,
        reduce_mode=None,
    )

    return out


def _check_tensor_indexing_arg(src, index, axis):
    assert src.ndim - 1 == index.ndim, f"{src.shape} {index.shape}"
    assert axis < src.ndim and 0 <= axis, f"invalid axis {axis} for shape {src.shape}"

    src_shape = list(src.shape)
    del src_shape[axis]
    assert src_shape == list(index.shape), f"{src.shape} {index.shape} {axis}"

    assert str(index.dtype) in [
        "int16",
        "int32",
        "int64",
        "uint16",
        "uint32",
        "uint64",
    ], f"{index.dtype}"


def indexing_with_tensor_index(src, index, axis=-1, keepdims=False):
    """
    indexing select items from src according to index in one dimension.
    src.ndim should equal to index.ndim + 1.
    if the src.shape remove the axis-th element, it should equal to index.shape.
    for example:
        src.shape=(2, 4, 6), index.shape=(2, 4), axis=2, out.shape=(2, 4, 1), out[i, j, 1] = src[i, j, index[i, j]]
        src.shape=(2, 4, 6), index.shape=(2, 6), axis=1, out.shape=(2, 1, 6), out[i, 1, j] = src[i, index[i, j], j]
        src.shape=(3, 9), index.shape=(3,), axis=1, out.shape=(3, 1), out[i, 1] = src[i, index[i]]
        src.shape=(3, 9), index.shape=(9,), axis=0, out.shape=(1, 9), out[1, i] = src[index[i], i]
    """
    axis = (axis + src.ndim) if axis < 0 else axis
    _check_tensor_indexing_arg(src, index, axis)

    arange_array = np.arange(src.shape[axis], dtype=index.dtype)
    arange_array = HLOTensor(arange_array).broadcast_to(
        src.shape, broadcast_dims=[axis]
    )
    broadcast_dims = [i for i in range(src.ndim) if i != axis]
    index_array = index.broadcast_to(src.shape, broadcast_dims=broadcast_dims)

    mask = (arange_array == index_array).astype(src.dtype)
    return (src * mask).sum(axis, keepdims=keepdims)


@register_lower_rule(mops.IndexingOneHot)
def indexing_one_hot_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert (
        len(ctx.vars_out) == 1 and len(ctx.vars_in) == 2 and len(args) == 2
    ), f"{len(ctx.vars_out)}, {len(ctx.vars_in)}, {len(args)}"
    assert ctx.op.ndim == args[0].ndim, f"{ctx.op.ndim}, {args[0].shape}"
    return indexing_with_tensor_index(args[0], args[1], ctx.op.axis, keepdims=True)


def indexing_set_with_tensor_index(src, value, index, axis):
    """
    indexing set value to src according to index in one dimension. 
    value shape should can be broadcast or reshape to index shape.
    if value shape not equal to index shape, it should be broadcast to index shape firstly
    examples:
        src.shape=(2, 4, 6), value.shape=(2, 4), index.shape=(2, 4), axis=2, out.shape=(2, 4, 6)
        out[i, j, k] = value[i, j] if k == index[i, j] else src[i, j, k]

        src.shape=(2, 4, 6), value.shape=(2, 6), index.shape=(2, 6), axis=1, out.shape=(2, 4, 6)
        out[i, j, k] = value[i, k] if j == index[i, k] else src[i, j, k]
    """
    axis = (axis + src.ndim) if axis < 0 else axis
    _check_tensor_indexing_arg(src, index, axis)

    value = value if isinstance(value, HLOTensor) else HLOTensor(value)
    assert src.dtype == value.dtype, f"{src.dtype}, {value.dtype}"

    arange_array = np.arange(src.shape[axis]).astype(index.dtype)
    arange_array = HLOTensor(arange_array).broadcast_to(src.shape, [axis])
    broadcast_dims = [i for i in range(src.ndim) if i != axis]
    index_array = index.broadcast_to(src.shape, broadcast_dims=broadcast_dims)

    mask1 = (arange_array == index_array).astype(src.dtype)
    mask2 = (arange_array != index_array).astype(value.dtype)

    return mask1 * value + mask2 * src


@register_lower_rule(mops.IndexingSetOneHot)
def indexing_set_one_hot_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert (
        len(ctx.vars_out) == 1 and len(ctx.vars_in) == 3 and len(args) == 3
    ), f"{len(ctx.vars_out)}, {len(ctx.vars_in)}, {len(args)}"
    return indexing_set_with_tensor_index(args[0], args[2], args[1], ctx.op.axis)


def convert_negative_index(indices: HLOTensor, max_indices: int):
    max_i = HLOTensor(np.array([max_indices], dtype="int32"))
    zero = HLOTensor(np.array([0], dtype="int32"))
    zeros = zero.broadcast_to(indices.shape)
    max_i = max_i.broadcast_to(indices.shape)
    positive_indices = indices + max_i
    mask = indices < zeros
    return HLOTensor(
        hlo.SelectOp(mask.tensor, positive_indices.tensor, indices.tensor).results
    )


@register_lower_rule(mops.IndexingMultiAxisVec)
def vec_indexing_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert len(ctx.param["items"]) == 1
    axis, _, _, _, is_index = ctx.param["items"][0]
    assert is_index
    inp, indices = args[0], args[1]
    indices = convert_negative_index(indices, inp.shape[axis])
    indices = indices.reshape(indices.shape + (1,))
    slices_size = tuple(
        (inp.shape[i] if i != axis else 1 for i in range(len(inp.shape)))
    )
    return xla_gather(
        inp,
        indices,
        slices_size,
        offset_dims=tuple(i for i in range(len(inp.shape)) if i != axis),
        collapsed_slice_dims=(axis,),
        start_index_map=(axis,),
    )


@register_lower_rule(mops.IndexingIncrMultiAxisVec)
def vec_indexing_incr_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert len(ctx.param["items"]) == 1
    axis, _, _, _, is_index = ctx.param["items"][0]
    assert is_index
    inp, y, indices = args[0], args[1], args[2]
    indices = convert_negative_index(indices, inp.shape[axis])
    indices = indices.reshape(indices.shape + (1,))

    out = xla_scatter(
        inp,
        indices,
        y,
        update_window_dims=tuple(i for i in range(len(inp.shape)) if i != axis),
        inserted_window_dims=(axis,),
        scattered_dims_to_operand_dims=(axis,),
        reduce_mode="sum",
    )
    return out
