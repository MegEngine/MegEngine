from typing import Sequence, Union

import numpy as np

from ...core._imperative_rt import ops as mops
from .. import ir_utils
from ..ir_utils import bool_attr, i64_attr
from ..lib.mlir import ir
from ..lib.mlir.dialects import chlo, hlo
from ..utils import flatten_list
from .hlotensor import HLOTensor
from .indexing import ScatterDimensionNumbers, scatter
from .tensor import concat, expand_dims, fill, iota
from .utils import _can_broadcast_to, _shape_equal, register_lower_rule


@register_lower_rule(mops.Dot)
def dot_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert (
        len(ctx.vars_in) == 2 and len(ctx.vars_out) == 1 and len(args) == 2
    ), f"{len(ctx.vars_in)}, {len(ctx.vars_out)}, {len(args)}"
    assert args[0].ndim == 1 and args[1].ndim == 1, f"{args[0].shape}, {args[1].shape}"
    assert args[0].shape[0] == args[1].shape[0], f"{args[0].shape}, {args[1].shape}"

    dot_dnums = hlo.DotDimensionNumbers.get(
        lhs_batching_dimensions=tuple(),
        rhs_batching_dimensions=tuple(),
        lhs_contracting_dimensions=(0,),
        rhs_contracting_dimensions=(0,),
    )

    return [
        HLOTensor(
            hlo.DotGeneralOp(
                ir_utils.make_ir_type_according_meta((), ctx.vars_out[0].dtype),
                args[0].tensor,
                args[1].tensor,
                dot_dnums,
                precision_config=ir_utils.precision_attr(args[0].dtype, args[1].dtype),
            ).result
        ).reshape(ctx.vars_out[0].shape)
    ]


@register_lower_rule(mops.MatrixMul)
def matmul_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert len(ctx.vars_in) == 2 and len(ctx.vars_out) == 1 and len(args) == 2
    assert (
        ctx.op.compute_mode == mops.BatchedMatrixMul.ComputeMode.DEFAULT
    ), f"{ctx.op.compute_mode}"
    assert ctx.op.format == mops.BatchedMatrixMul.Format.DEFAULT, f"{ctx.op.format}"
    assert ctx.op.dimA == len(args[0].shape) and ctx.op.dimB == len(
        args[1].shape
    ), f"{ctx.op.dimA}, {ctx.op.dimB}, {args[0].shape}, {args[1].shape}"
    assert args[0].ndim >= 2 and args[1].ndim >= 2, f"{args[0].shape}, {args[1].shape}"
    lhs, rhs = args[0], args[1]

    # in mge batchmatmul, [a, b, c, d] * [a, b, c, f] -> [a, b, f, d]
    # but in mge matmul, dims [:-1] is interpreted as one edge of matrix
    # that means [a, b, c, d] * [a, b, c, f] -> [a*b*c, d] * [a*b*c, f] -> [f, d]
    if lhs.ndim > 2 and rhs.ndim > 2:
        lhs = lhs.reshape(shape=(-1, lhs.shape[-1]))
        rhs = rhs.reshape(shape=(-1, rhs.shape[-1]))

    lhs_reduce_axis = lhs.ndim - 2 if ctx.op.transposeA else lhs.ndim - 1
    rhs_reduce_axis = rhs.ndim - 1 if ctx.op.transposeB else rhs.ndim - 2
    assert (
        lhs.shape[lhs_reduce_axis] == rhs.shape[rhs_reduce_axis]
    ), f"reduce axis length mismatch: {lhs.shape}, {rhs.shape}, {lhs_reduce_axis}, {rhs_reduce_axis}"

    dot_dnums = hlo.DotDimensionNumbers.get(
        lhs_batching_dimensions=tuple(),
        rhs_batching_dimensions=tuple(),
        lhs_contracting_dimensions=(lhs_reduce_axis,),
        rhs_contracting_dimensions=(rhs_reduce_axis,),
    )

    return [
        HLOTensor(
            hlo.DotGeneralOp(
                ir_utils.mge_varinfo_to_ir_type(ctx.vars_out[0]),
                lhs.tensor,
                rhs.tensor,
                dot_dnums,
                precision_config=ir_utils.precision_attr(lhs.dtype, rhs.dtype),
            ).result
        )
    ]


def _bmm_shape_helper(lhs_shape, rhs_shape, lhs_transpose, rhs_transpose):
    lhs_reduce_axis = len(lhs_shape) - 2 if lhs_transpose else len(lhs_shape) - 1
    rhs_reduce_axis = len(rhs_shape) - 1 if rhs_transpose else len(rhs_shape) - 2

    # get the shape of inputs after transpose
    lhs_shape, rhs_shape = list(lhs_shape), list(rhs_shape)
    if lhs_transpose:
        lhs_shape[-2], lhs_shape[-1] = lhs_shape[-1], lhs_shape[-2]
    if rhs_transpose:
        rhs_shape[-2], rhs_shape[-1] = rhs_shape[-1], rhs_shape[-2]
    # get the batch info of inputs
    lhs_prefix, rhs_prefix = lhs_shape[:-2], rhs_shape[:-2]

    # only the batch of input_a can broadcast to input_b supported
    assert _can_broadcast_to(lhs_prefix, rhs_prefix) or _can_broadcast_to(
        rhs_prefix, lhs_prefix
    ), f"{lhs_shape}, {rhs_shape}"

    # get the batch axis of input_a and input_b, for example:
    # (3, 4, 5) * (3, 5, 6), the batch axis is (0,) and (0,)
    # (3, 4, 5) * (2, 3, 5, 6), the batch axis is (0,) and (1,)
    # (2, 3, 4, 5) * (2, 3, 5, 6), the batch axis is (0, 1) and (0, 1)
    lhs_batch_axis, rhs_batch_axis = [], []
    min_len = min(len(lhs_shape), len(rhs_shape))
    for i in range(-3, -min_len - 1, -1):
        if lhs_shape[i] == rhs_shape[i]:
            lhs_batch_axis.append(i)
            rhs_batch_axis.append(i)

        elif lhs_shape[i] == 1 or rhs_shape[i] == 1:
            lhs_batch_axis.append(i)
            rhs_batch_axis.append(i)

        else:
            break

    lhs_batch_axis = [val + len(lhs_shape) for val in lhs_batch_axis]
    rhs_batch_axis = [val + len(rhs_shape) for val in rhs_batch_axis]
    lhs_batch_axis.sort()
    rhs_batch_axis.sort()

    assert len(lhs_batch_axis) == len(lhs_prefix) or len(rhs_batch_axis) == len(
        rhs_prefix
    ), f"{lhs_batch_axis}, {rhs_batch_axis}, {lhs_prefix}, {rhs_prefix}, {lhs_shape}, {rhs_shape}"

    # for case [m, ... , n, a, b] * [i, ..., j, m, ..., n, b, c]
    if _can_broadcast_to(lhs_prefix, rhs_prefix):
        # [m, ..., n]
        batched_part = [rhs_prefix[ax] for ax in rhs_batch_axis]
        # [i, ..., j]
        nonbatched_part = rhs_prefix[0 : len(rhs_prefix) - len(rhs_batch_axis)]

        # in xla, [m, ... , n, a, b] * [i, ..., j, m, ..., n, b, c] -> [m, ..., n, a, i, ..., j, c]
        # in mge, [m, ... , n, a, b] * [i, ..., j, m, ..., n, b, c] -> [i, ..., j, m, ..., n, a, c]
        # so we need permute
        xla_oshape = [*batched_part, lhs_shape[-2], *nonbatched_part, rhs_shape[-1]]
        nonbatched_perm = [
            idx + 1 + len(batched_part) for idx in range(len(nonbatched_part))
        ]
        batched_perm = [idx for idx in range(len(batched_part))]
        permutation = [
            *nonbatched_perm,
            *batched_perm,
            len(batched_part),
            len(xla_oshape) - 1,
        ]
    # for case [i, ..., j, m, ..., n, a, b] * [m, ..., n, b, c]
    else:
        # [m, ..., n]
        batched_part = [lhs_prefix[ax] for ax in lhs_batch_axis]
        # [i, ..., j]
        nonbatched_part = lhs_prefix[0 : len(lhs_prefix) - len(lhs_batch_axis)]

        # in xla, [i, ..., j, m, ... , n, a, b] * [m, ..., n, b, c] -> [m, ..., n, i, ..., j, a, c]
        # in mge, [i, ..., j, m, ... , n, a, b] * [m, ..., n, b, c] -> [i, ..., j, m, ..., n, a, c]
        # so we need permute
        xla_oshape = [*batched_part, *nonbatched_part, lhs_shape[-2], rhs_shape[-1]]
        nonbatched_perm = [
            idx + len(batched_part) for idx in range(len(nonbatched_part))
        ]
        batched_perm = [idx for idx in range(len(batched_part))]
        permutation = [
            *nonbatched_perm,
            *batched_perm,
            len(xla_oshape) - 2,
            len(xla_oshape) - 1,
        ]

    return (
        lhs_reduce_axis,
        rhs_reduce_axis,
        lhs_batch_axis,
        rhs_batch_axis,
        xla_oshape,
        permutation,
    )


@register_lower_rule(mops.BatchedMatrixMul)
def batched_matmul_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert len(ctx.vars_in) == 2 and len(ctx.vars_out) == 1 and len(args) == 2
    assert (
        ctx.op.compute_mode == mops.BatchedMatrixMul.ComputeMode.DEFAULT
    ), f"{ctx.op.compute_mode}"
    assert ctx.op.format == mops.BatchedMatrixMul.Format.DEFAULT, f"{ctx.op.format}"
    assert ctx.op.dimA == len(args[0].shape) and ctx.op.dimB == len(
        args[1].shape
    ), f"{ctx.op.dimA}, {ctx.op.dimB}, {args[0].shape}, {args[1].shape}"
    assert args[0].ndim >= 2 and args[1].ndim >= 2, f"{args[0].shape}, {args[1].shape}"
    lhs, rhs = args[0], args[1]

    (
        lhs_reduce_axis,
        rhs_reduce_axis,
        lhs_batch_axis,
        rhs_batch_axis,
        xla_oshape,
        permutation,
    ) = _bmm_shape_helper(lhs.shape, rhs.shape, ctx.op.transposeA, ctx.op.transposeB)

    # in xla, [3, 4, 5, 6] * [3, 1, 6, 7] is illegal, so we broadcast [3, 1, 6, 7] -> [3, 4, 6, 7]
    if _can_broadcast_to(lhs.shape[:-2], rhs.shape[:-2]):
        lshape = [
            rhs.shape[r] if lhs.shape[l] == 1 else lhs.shape[l]
            for l, r in zip(lhs_batch_axis, rhs_batch_axis)
        ]
        lshape = [*lshape, *lhs.shape[-2:]]
        if not _shape_equal(lshape, lhs.shape):
            lhs = lhs.broadcast_to(lshape)
    else:
        assert _can_broadcast_to(rhs.shape[:-2], lhs.shape[:-2])
        rshape = [
            lhs.shape[l] if rhs.shape[r] == 1 else rhs.shape[r]
            for l, r in zip(lhs_batch_axis, rhs_batch_axis)
        ]
        rshape = [*rshape, *rhs.shape[-2:]]
        if not _shape_equal(rshape, rhs.shape):
            rhs = rhs.broadcast_to(rshape)

    dot_dnums = hlo.DotDimensionNumbers.get(
        lhs_batching_dimensions=list(lhs_batch_axis),
        rhs_batching_dimensions=list(rhs_batch_axis),
        lhs_contracting_dimensions=(lhs_reduce_axis,),  # the reduce axis in lhs
        rhs_contracting_dimensions=(rhs_reduce_axis,),  # the reduce axis in rhs
    )

    return HLOTensor(
        hlo.DotGeneralOp(
            ir_utils.make_ir_type_according_meta(xla_oshape, ctx.vars_out[0].dtype),
            lhs.tensor,
            rhs.tensor,
            dot_dnums,
            precision_config=ir_utils.precision_attr(lhs.dtype, rhs.dtype),
        ).result
    ).transpose(permutation)


def _sort_according_to_key(key, *vals, axis=-1, descending=True, is_stable=True):
    """
    sort key and vals in the specified axis, return the sorted key and vals.
    key and vals should have the same shape, then we reorder both key and vals according
    to the value of the key.

    example 1: (implement argsort)
    inp: 1.7783 -> 0, -1.8184 -> 1, 1.0701 -> 2
        [[ 1.7783 -1.8184  1.0701]
         [-0.0712 -1.4623  1.3243]]
        [[0 1 2]
         [0 1 2]]
    axis: -1
    descend: True
    return: after reorder, 1.7783 -> 0, -1.8184 -> 1, 1.0701 -> 2
        [[ 1.7783  1.0701 -1.8184]
         [ 1.3243 -0.0712 -1.4623]]
        [[0 2 1]
         [2 0 1]]
    
    example 2:
    inp:
        [[0 2 1]
         [2 0 1]]
        [[ 1.7783  1.0701 -1.8184]
         [ 1.3243 -0.0712 -1.4623]]
    axis: -1
    descend: False
    return:
        [[0 1 2]
         [0 1 2]]
        [[ 1.7783 -1.8184  1.0701]
         [-0.0712 -1.4623  1.3243]]
    """

    for val in vals:
        assert _shape_equal(
            key.shape, val.shape
        ), f"sort key and vals shape mismatch: {key.shape}, {val.shape}"

    axis = axis + key.ndim if axis < 0 else axis
    sorted_key = ir_utils.make_ir_type_according_meta(key.shape, key.dtype)
    sorted_vals = [
        ir_utils.make_ir_type_according_meta(val.shape, val.dtype) for val in vals
    ]

    sort_op = hlo.SortOp(
        [sorted_key, *sorted_vals],
        [key.tensor, *[val.tensor for val in vals]],
        dimension=i64_attr(axis),
        is_stable=bool_attr(is_stable),
    )

    key_type = ir_utils.make_ir_type_according_meta(tuple(), key.dtype)
    val_types = [
        ir_utils.make_ir_type_according_meta(tuple(), val.dtype) for val in vals
    ]
    arg_types = [key_type] + val_types

    comparator = sort_op.comparator.blocks.append(
        *flatten_list(zip(arg_types, arg_types))
    )
    with ir.InsertionPoint(comparator):
        lhs = HLOTensor(comparator.arguments[0])
        rhs = HLOTensor(comparator.arguments[1])

        if descending:
            hlo.ReturnOp([(lhs > rhs).tensor])
        else:
            hlo.ReturnOp([(lhs < rhs).tensor])

    assert len(sort_op.results) == len(vals) + 1, f"{len(vals)}, {len(sort_op.results)}"
    return (HLOTensor(ret) for ret in sort_op.results)


def argsort(inp, axis=-1, descending=True, is_stable=True):
    """
    sort inp in the specfic axis, and return the sorted value and index
    for example:
    inp:
        [[ 1.7783 -1.8184  1.0701]
         [-0.0712 -1.4623  1.3243]]
    axis: -1
    descend: True
    return:
        [[ 1.7783  1.0701 -1.8184]
         [ 1.3243 -0.0712 -1.4623]]
        [[0 2 1]
         [2 0 1]]
    """
    axis = axis + inp.ndim if axis < 0 else axis
    idx = iota(np.int32, inp.shape, axis)
    return _sort_according_to_key(
        inp, idx, axis=axis, descending=descending, is_stable=is_stable
    )


@register_lower_rule(mops.Argsort)
def argsort_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert (
        len(args) == 1 and len(ctx.vars_in) == 1 and len(ctx.vars_out) == 2
    ), f"{len(args)}, {len(ctx.vars_in)}, {len(ctx.vars_out)}"
    assert ctx.op.order in [
        mops.Argsort.Order.DESCENDING,
        mops.Argsort.Order.ASCENDING,
    ], f"{ctx.op.order}"
    descending = ctx.op.order == mops.Argsort.Order.DESCENDING
    axis = args[0].ndim - 1  # megengine only support sort in the last dimension
    return argsort(args[0], axis, descending, is_stable=True)


@register_lower_rule("ArgsortBackward")
def argsort_backward_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert (
        len(args) == 3 and len(ctx.vars_in) == 3 and len(ctx.vars_out) == 1
    ), f"{len(args)}, {len(ctx.vars_in)}, {len(ctx.vars_out)}"
    dy, idx, x = args[0], args[1], args[2]
    if _shape_equal(x.shape, dy.shape):
        # for argsort backward
        _, dx = _sort_according_to_key(
            idx, dy, axis=-1, descending=False, is_stable=True
        )
    else:
        # for topk backward, only support axis=-1 and the dx is 2d tensor
        dx = fill(0, ctx.vars_out[0].shape, ctx.vars_out[0].dtype)
        expander = iota(np.int32, idx.shape, dimension=0)
        idx = expand_dims(idx, -1)
        expander = expand_dims(expander, -1)
        idx = concat([expander, idx], -1)

        dnums = ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0, 1),
            scatter_dims_to_operand_dims=(0, 1),
        )
        dx = scatter(dx, idx, dy, dnums, unique_indices=True)
    return dx


def topk(inp, k, descending=True, kth_only=False, no_sort=False):
    """
    do topk in the last dimension of inp, for example:
        inp.shape = (2, 3, 4), k = 2, out_shape = (2, 3, 2)
    """
    assert k > 0, f"k of topk must bigger than 0, get {k}"
    assert no_sort == False, f"no_sort must be False now"
    assert kth_only == False, f"kth_only is not support now"

    if descending == True:
        out, idx = [
            HLOTensor(rst) for rst in chlo.TopKOp(inp.tensor, i64_attr(k)).results
        ]
    else:
        inp = -inp
        out, idx = [
            HLOTensor(rst) for rst in chlo.TopKOp(inp.tensor, i64_attr(k)).results
        ]
        out = -out

    return out, idx


@register_lower_rule(mops.TopK)
def topk_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert (
        len(args) == 2 and len(ctx.vars_in) == 2
    ), f"{len(args)}, {len(ctx.vars_in)}, {len(ctx.vars_out)}"
    assert isinstance(
        ctx.vars_in[1].bound_data, np.ndarray
    ), f"{ctx.vars_in[1].bound_data}"
    k = int(ctx.vars_in[1].bound_data)

    descending = True if k < 0 else False
    k = -k if k < 0 else k

    if ctx.op.mode == mops.TopK.Mode.VALUE_IDX_SORTED:
        assert len(ctx.vars_out) == 2, f"{len(ctx.vars_out)}"
        kth_only, no_sort = False, False
    elif ctx.op.mode == mops.TopK.Mode.VALUE_IDX_NOSORT:
        assert len(ctx.vars_out) == 2, f"{len(ctx.vars_out)}"
        kth_only, no_sort = False, True
    else:
        assert (
            ctx.op.mode == mops.TopK.Mode.KTH_ONLY
        ), f"invalid mode for topk, {ctx.op.mode}"
        kth_only, no_sort = True, False
        assert len(ctx.vars_out) == 1, f"{len(ctx.vars_out)}"

    return topk(args[0], k, descending, kth_only, no_sort)
