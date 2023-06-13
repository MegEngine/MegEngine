from typing import Sequence, Union

import numpy as np

from ...core._imperative_rt import ops as mops
from .. import ir_utils
from ..lib import xla_client as xc
from ..lib.mlir.dialects import hlo
from .hlotensor import HLOTensor
from .utils import _shape_equal, register_lower_rule

RandomAlgorithm = xc.ops.RandomAlgorithm
RandomAlgorithm.__str__ = lambda algorithm: algorithm.name


def _rng_algorithm(algorithm: RandomAlgorithm):
    assert algorithm == RandomAlgorithm.RNG_THREE_FRY
    if algorithm == RandomAlgorithm.RNG_THREE_FRY:
        return hlo.RngAlgorithmAttr.get("THREE_FRY")
    elif algorithm == RandomAlgorithm.RNG_PHILOX:
        return hlo.RngAlgorithmAttr.get("PHILOX")
    elif algorithm == RandomAlgorithm.RNG_DEFAULT:
        return hlo.RngAlgorithmAttr.get("DEFAULT")
    else:
        assert False


def rng_uint_generator(
    key, oshape, odtype="uint32", algorithm=RandomAlgorithm.RNG_THREE_FRY
):

    assert np.dtype(odtype) in {
        np.dtype("uint8"),
        np.dtype("uint16"),
        np.dtype("uint32"),
        np.dtype("uint64"),
    }, f"only unsigned int supported, got {odtype}({type(odtype)})"
    assert algorithm == RandomAlgorithm.RNG_THREE_FRY, "only ThreeFry supported now"
    assert _shape_equal(key.shape, (2, 2)), f"key shape error, {key.shape}"
    assert key.dtype == "int32", f"key dtype error, {key.dtype}"

    # bitcast (2x2,i32) -> (2,u64)
    org_key_shape, org_key_dtype = key.shape, key.dtype
    key = key.bitcast((2,), "uint64")

    if odtype == "uint32" or odtype == "uint64":
        rng_odtype = odtype
    else:
        rng_odtype = "uint32"

    algorithm_attr = _rng_algorithm(algorithm)
    new_key, out_vals = hlo.RngBitGeneratorOp(
        ir_utils.make_ir_type_according_meta(key.shape, key.dtype),
        ir_utils.make_ir_type_according_meta(oshape, rng_odtype),
        algorithm_attr,
        key.tensor,
    ).results
    new_key, out_vals = HLOTensor(new_key), HLOTensor(out_vals)
    new_key = new_key.bitcast(org_key_shape, org_key_dtype)

    if rng_odtype != odtype:
        out_vals = out_vals.astype(odtype)
    return out_vals, new_key


@register_lower_rule(mops.Dropout)
def dropout_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert len(ctx.vars_in) == 2 and len(args) == 2 and len(ctx.vars_out) == 3
    inp, key = args
    random_val, new_key = rng_uint_generator(key, inp.shape, "uint32")
    mask = random_val > np.array(ctx.op.drop_prob * np.iinfo(np.uint32).max, np.uint32)
    multiplier = mask.astype(inp.dtype)
    multiplier = multiplier / (1.0 - ctx.op.drop_prob)
    out = inp * multiplier
    mask = mask.reshape((-1,)).astype("uint8")
    return out, mask, new_key


@register_lower_rule("DropoutBackward")
def droupout_backward_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert len(args) == 2 and len(ctx.vars_in) == 2 and len(ctx.vars_out) == 1
    dy, mask = args[0], args[1]
    scale = 1.0 - ctx.param["drop_prob"]
    multiplier = mask.reshape(dy.shape).astype(dy.dtype) / scale
    return dy * multiplier
