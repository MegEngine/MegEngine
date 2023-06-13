from typing import Sequence, Union

from ...core._imperative_rt import ops as mops
from .. import ir_utils
from ..lib.mlir.dialects import hlo
from .hlotensor import HLOTensor
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
