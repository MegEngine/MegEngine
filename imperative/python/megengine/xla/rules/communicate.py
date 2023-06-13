import itertools
from functools import partial
from typing import Sequence, Union

import numpy as np

from ...core._imperative_rt import ops as mops
from .. import ir_utils
from ..lib.mlir import ir
from ..lib.mlir.dialects import hlo
from .hlotensor import HLOTensor
from .tensor import concat, split
from .utils import register_lower_rule


@register_lower_rule(mops.ParamPackConcat)
def parampack_concat_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    flattened = []
    for arg, var_in in zip(args[:-1], ctx.vars_in[:-1]):
        ishape_1d = (int(np.prod(var_in.shape)),)
        flattened.append(arg.reshape(ishape_1d))
    concated = concat(flattened, 0)
    return concated


@register_lower_rule(mops.ParamPackSplit)
def parampack_split_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    offsets, shapes, var_outs = ctx.op.offsets, ctx.op.shapes, ctx.vars_out
    assert (len(offsets) // 2) == len(shapes) == len(var_outs), "error params"
    for var_out, shape in zip(var_outs, shapes):
        assert tuple(var_out.shape) == tuple(shape), f"{var_out.shape} .vs {shape}"

    sections = [np.prod(shape) for shape in shapes]
    for i, section in enumerate(sections):
        assert section == offsets[2 * i + 1] - offsets[2 * i], "error offsets"

    pieces = split(args[0], sections, axis=0)
    outputs = [piece.reshape(var_out.shape) for piece, var_out in zip(pieces, var_outs)]
    return outputs


def _all_reduce(reducer, inp, world_size):
    def _replica_groups_hlo(replica_groups: Sequence[Sequence[int]]):
        groups = np.array(
            list(itertools.zip_longest(*replica_groups, fillvalue=-1)), dtype=np.int64
        ).T
        return ir.DenseIntElementsAttr.get(np.ascontiguousarray(groups))

    replica_groups = _replica_groups_hlo([[i for i in range(world_size)]])
    hlo_cfgs = {}

    all_reduce_op = hlo.AllReduceOp(
        inp.tensor.type, inp.tensor, replica_groups=replica_groups, **hlo_cfgs
    )
    scalar_type = ir_utils.make_ir_type_according_meta(tuple(), inp.dtype)
    reducer_region = all_reduce_op.regions[0].blocks.append(scalar_type, scalar_type)
    with ir.InsertionPoint(reducer_region):
        reducer_ret = reducer(*reducer_region.arguments)
        hlo.ReturnOp(reducer_ret.results)
    return HLOTensor(all_reduce_op.results)


all_reduce_sum = partial(_all_reduce, hlo.AddOp)
all_reduce_prod = partial(_all_reduce, hlo.MulOp)
all_reduce_min = partial(_all_reduce, hlo.MinOp)
all_reduce_max = partial(_all_reduce, hlo.MaxOp)


@register_lower_rule(mops.CollectiveComm)
def collective_comm_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert len(args) == 1, "collective comm only support one input"
    if ctx.op.mode == mops.CollectiveComm.Mode.ALL_REDUCE_SUM:
        ret = all_reduce_sum(args[0], ctx.op.nr_devices)
    elif ctx.op.mode == mops.CollectiveComm.Mode.ALL_REDUCE_PROD:
        ret = all_reduce_prod(args[0], ctx.op.nr_devices)
    elif ctx.op.mode == mops.CollectiveComm.Mode.ALL_REDUCE_MIN:
        ret = all_reduce_min(args[0], ctx.op.nr_devices)
    elif ctx.op.mode == mops.CollectiveComm.Mode.ALL_REDUCE_MAX:
        ret = all_reduce_max(args[0], ctx.op.nr_devices)
    else:
        assert False, f"not support mode{ctx.op.mode}"
    return ret
