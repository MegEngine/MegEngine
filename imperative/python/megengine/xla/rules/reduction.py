from functools import partial
from typing import Sequence, Union

import numpy as np

from ...core._imperative_rt import ops as mops
from .. import ir_utils
from ..lib.mlir import ir
from ..lib.mlir.dialects import hlo
from .elemwise import div
from .hlotensor import HLOTensor
from .tensor import reshape
from .utils import _check_dtype, _check_shape, _shape_equal, register_lower_rule


def _get_sum_identity(dtype) -> np.ndarray:
    return np.array(0, dtype)


def _get_prod_identity(dtype) -> np.ndarray:
    return np.array(1, dtype)


def _get_max_identity(dtype) -> np.ndarray:
    if dtype == np.float32 or dtype == np.float64 or dtype == np.float16:
        return np.array(-np.inf, dtype)
    elif (
        dtype == np.int32 or dtype == np.int64 or dtype == np.int16 or dtype == np.int8
    ):
        return np.array(np.iinfo(dtype).min, dtype)
    else:
        assert False, f"unsupported dtype for max: {dtype}"


def _get_min_identity(dtype) -> np.ndarray:
    if dtype == np.float32 or dtype == np.float64 or dtype == np.float16:
        return np.array(np.inf, dtype)
    elif (
        dtype == np.int32 or dtype == np.int64 or dtype == np.int16 or dtype == np.int8
    ):
        return np.array(np.iinfo(dtype).max, dtype)
    else:
        assert False, f"unsupported dtype for max: {dtype}"


def _get_bitwise_and_identity(dtype) -> np.ndarray:
    return np.array(-1).astype(dtype)


def _get_bitwise_or_identity(dtype) -> np.ndarray:
    return np.array(0, dtype)


def _normalize_reduce_axes(ishape, axes):
    axes = list(range(len(ishape))) if axes is None else axes
    axes = [axes] if isinstance(axes, int) else axes
    axes = [axis if axis >= 0 else axis + len(ishape) for axis in axes]
    return axes


def _infer_reduce_shape(ishape, axes, keepdims=False):
    axes = _normalize_reduce_axes(ishape, axes)
    reduced_shape = []

    for axis, length in enumerate(ishape):
        if axis not in axes:
            reduced_shape.append(length)
        else:
            if keepdims:
                reduced_shape.append(1)
    return tuple(reduced_shape)


def _reduce(
    reducer, fidentity, inp, axes=None, keepdims=False, oshape=None, odtype=None
):
    def _reduce_nokeepdim(reducer, fidentity, inp, axes=None, oshape=None, odtype=None):
        reduced_shape = _infer_reduce_shape(inp.shape, axes)

        _check_shape(reduced_shape, oshape)
        _check_dtype(inp.dtype, odtype)

        reduce_out = ir_utils.make_ir_type_according_meta(reduced_shape, inp.dtype)
        init_val = ir_utils.ir_constant_tuple(fidentity(inp.dtype))
        reduce_op = hlo.ReduceOp(
            [reduce_out], [inp.tensor], init_val, ir_utils.dense_int_elements(axes)
        )
        scalar_type = ir_utils.make_ir_type_according_meta(tuple(), inp.dtype)
        reducer_region = reduce_op.regions[0].blocks.append(scalar_type, scalar_type)
        with ir.InsertionPoint(reducer_region):
            reducer_ret = reducer(*reducer_region.arguments)
            hlo.ReturnOp(reducer_ret.results)

        return HLOTensor(reduce_op.result)

    axes = _normalize_reduce_axes(inp.shape, axes)
    maykeepdim_shape = _infer_reduce_shape(inp.shape, axes, keepdims)
    _check_shape(maykeepdim_shape, oshape)

    oup = _reduce_nokeepdim(reducer, fidentity, inp, axes, oshape, odtype)
    if _shape_equal(oup.shape, maykeepdim_shape):
        return oup
    else:
        return reshape(oup, maykeepdim_shape)


sum = partial(_reduce, hlo.AddOp, _get_sum_identity)
prod = partial(_reduce, hlo.MulOp, _get_prod_identity)
max = partial(_reduce, hlo.MaxOp, _get_max_identity)
min = partial(_reduce, hlo.MinOp, _get_min_identity)
all = partial(_reduce, hlo.AndOp, _get_bitwise_and_identity)
any = partial(_reduce, hlo.OrOp, _get_bitwise_or_identity)


def mean(inp, axes=None, keepdims=False):
    axes = _normalize_reduce_axes(inp.shape, axes)
    inp_sum = sum(inp, axes, keepdims)
    inp_shape = inp.shape

    divider = 1.0
    for ax in axes:
        divider *= inp_shape[ax]

    return div(inp_sum, divider)


@register_lower_rule(mops.Reduce)
def reduce_lower(ctx, *args: Union[ir.Value, Sequence[ir.Value]]):
    assert ctx.op.data_type == mops.Reduce.DataType.DEFAULT

    opr = ctx.op
    keepdims = opr.keepdim
    if len(args) == 1:
        assert isinstance(opr.axis, int)
        if opr.axis < 0:
            axes = opr.axis + args[0].ndim
        else:
            axes = (opr.axis,)
        if opr.axis > 7:
            axes = tuple(np.arange(args[0].ndim))
            keepdims = False
    else:
        assert len(args) == 2
        src_shape = args[0].shape
        if src_shape == ctx.vars_out[0].shape:
            return args[0]
        tgt_shape = list(ctx.vars_out[0].shape)
        tgt_shape = [1,] * (len(src_shape) - len(tgt_shape)) + tgt_shape
        src_idx, tgt_idx, axes = 0, 0, []
        while src_idx < len(src_shape) and tgt_idx < len(tgt_shape):
            if src_shape[src_idx] != 1 and tgt_shape[tgt_idx] == 1:
                axes.append(src_idx)
                src_idx = src_idx + 1
                tgt_idx = tgt_idx + 1
            elif src_shape[src_idx] != tgt_shape[tgt_idx]:
                axes.append(src_idx)
                src_idx = src_idx + 1
            else:
                src_idx = src_idx + 1
                tgt_idx = tgt_idx + 1
        assert tgt_idx == len(
            tgt_shape
        ), f"src_shape: {src_shape}, tgt_shape: {tgt_shape}"
        axes = axes + list(range(src_idx, len(src_shape)))

    if opr.mode == mops.Reduce.Mode.SUM:
        ret = sum(args[0], axes, keepdims)
    elif opr.mode == mops.Reduce.Mode.MEAN:
        ret = mean(args[0], axes, keepdims)
    elif opr.mode == mops.Reduce.Mode.PRODUCT:
        ret = prod(args[0], axes, keepdims)
    elif opr.mode == mops.Reduce.Mode.MAX:
        ret = max(args[0], axes, keepdims)
    elif opr.mode == mops.Reduce.Mode.MIN:
        ret = min(args[0], axes, keepdims)
    else:
        assert False, f"no support reduce mode {opr.mode}"

    if not _shape_equal(ret.shape, ctx.vars_out[0].shape):
        ret = ret.reshape(ctx.vars_out[0].shape)
    return ret
