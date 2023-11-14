from typing import Sequence, Union

import numpy as np

from ...core._imperative_rt import ops as mops
from .elemwise import floor, minimum
from .hlotensor import HLOTensor
from .tensor import iota, ones, stack, where, xla_gather, xla_scatter, zeros
from .utils import _shape_equal, register_lower_rule


def resize_nearest_helper(ilen: int, olen: int):
    assert olen != 0 and ilen != 0
    scale = olen / ilen
    base = iota(dtype=np.float32, shape=(olen,), dimension=-1)
    idx = (base / scale).astype(np.int32)
    return minimum(idx, ilen - 1)


def resize_linear_helper(ilen: int, olen: int):
    assert olen != 0 and ilen != 0
    scale = olen / ilen
    if ilen == 1:
        alpha0 = ones((olen,), dtype=np.float32)
        idx0 = zeros((olen,), dtype=np.int32)
        alpha1 = zeros((olen,), dtype=np.float32)
        idx1 = zeros((olen,), dtype=np.int32)
    else:
        base = iota(dtype=np.float32, shape=(olen,), dimension=-1)
        alpha = (base + 0.5) / scale - 0.5
        origin_idx = floor(alpha)
        alpha = alpha - origin_idx
        origin_idx_l0 = origin_idx < 0.0
        origin_idx_ll = origin_idx + 1.0 >= (ilen * 1.0)

        origin_idx = where(origin_idx_l0, 0.0, origin_idx)
        origin_idx = where(origin_idx_ll, ilen - 2.0, origin_idx)
        alpha = where(origin_idx_l0, 0.0, alpha)
        alpha = where(origin_idx_ll, 1.0, alpha)

        alpha0 = 1.0 - alpha
        idx0 = origin_idx.astype(np.int32)
        alpha1 = alpha
        idx1 = (origin_idx + 1.0).astype(np.int32)

    return alpha0, idx0, alpha1, idx1


def resize_nearest(inp: HLOTensor, size: Sequence[int]):
    N, C, IH, IW = inp.shape
    OH, OW = size
    assert IH != 0 and IW != 0 and OH != 0 and OW != 0, f"{inp.shape}, {size}"

    ihidx = resize_nearest_helper(IH, OH)
    iwidx = resize_nearest_helper(IW, OW)
    ihidx = ihidx.reshape((OH, 1)).broadcast_to((OH, OW))
    iwidx = iwidx.reshape((1, OW)).broadcast_to((OH, OW))
    iidx = ihidx * IW + iwidx
    iidx = iidx.broadcast_to((N, C, OH, OW)).reshape((N, C, OH * OW))

    dim0 = (
        iota(dtype=np.int32, shape=(N,), dimension=-1)
        .reshape((N, 1, 1))
        .broadcast_to(iidx.shape)
    )
    dim1 = (
        iota(dtype=np.int32, shape=(C,), dimension=-1)
        .reshape((1, C, 1))
        .broadcast_to(iidx.shape)
    )
    iidx = stack([dim0, dim1, iidx], axis=-1)

    inp = inp.reshape((N, C, -1))
    out = xla_gather(
        inp,
        iidx,
        slice_sizes=(1, 1, 1),
        offset_dims=tuple(),
        collapsed_slice_dims=(0, 1, 2),
        start_index_map=(0, 1, 2),
    )
    return out.reshape((N, C, OH, OW))


def resize_linear(inp: HLOTensor, size: Sequence[int]):
    N, C, IH, IW = inp.shape
    OH, OW = size
    assert IH != 0 and IW != 0 and OH != 0 and OW != 0, f"{inp.shape}, {size}"

    ah0s, ih0s, ah1s, ih1s = resize_linear_helper(IH, OH)
    aw0s, iw0s, aw1s, iw1s = resize_linear_helper(IW, OW)

    ih0s = ih0s.reshape((OH, 1)).broadcast_to((OH, OW))
    ih1s = ih1s.reshape((OH, 1)).broadcast_to((OH, OW))
    iw0s = iw0s.reshape((1, OW)).broadcast_to((OH, OW))
    iw1s = iw1s.reshape((1, OW)).broadcast_to((OH, OW))
    iidx00 = ih0s * IW + iw0s
    iidx01 = ih0s * IW + iw1s
    iidx10 = ih1s * IW + iw0s
    iidx11 = ih1s * IW + iw1s
    iidx00 = iidx00.broadcast_to((N, C, OH, OW)).reshape((N, C, -1))
    iidx01 = iidx01.broadcast_to((N, C, OH, OW)).reshape((N, C, -1))
    iidx10 = iidx10.broadcast_to((N, C, OH, OW)).reshape((N, C, -1))
    iidx11 = iidx11.broadcast_to((N, C, OH, OW)).reshape((N, C, -1))

    dim0 = (
        iota(dtype=np.int32, shape=(N,), dimension=-1)
        .reshape((N, 1, 1))
        .broadcast_to(iidx00.shape)
    )
    dim1 = (
        iota(dtype=np.int32, shape=(C,), dimension=-1)
        .reshape((1, C, 1))
        .broadcast_to(iidx00.shape)
    )
    iidx00 = stack([dim0, dim1, iidx00], axis=-1)
    iidx01 = stack([dim0, dim1, iidx01], axis=-1)
    iidx10 = stack([dim0, dim1, iidx10], axis=-1)
    iidx11 = stack([dim0, dim1, iidx11], axis=-1)

    inp = inp.reshape((N, C, -1))
    out00 = xla_gather(
        inp,
        iidx00,
        slice_sizes=(1, 1, 1),
        offset_dims=tuple(),
        collapsed_slice_dims=(0, 1, 2),
        start_index_map=(0, 1, 2),
    )
    out01 = xla_gather(
        inp,
        iidx01,
        slice_sizes=(1, 1, 1),
        offset_dims=tuple(),
        collapsed_slice_dims=(0, 1, 2),
        start_index_map=(0, 1, 2),
    )
    out10 = xla_gather(
        inp,
        iidx10,
        slice_sizes=(1, 1, 1),
        offset_dims=tuple(),
        collapsed_slice_dims=(0, 1, 2),
        start_index_map=(0, 1, 2),
    )
    out11 = xla_gather(
        inp,
        iidx11,
        slice_sizes=(1, 1, 1),
        offset_dims=tuple(),
        collapsed_slice_dims=(0, 1, 2),
        start_index_map=(0, 1, 2),
    )

    out00 = out00.reshape((N, C, OH, OW))
    out01 = out01.reshape((N, C, OH, OW))
    out10 = out10.reshape((N, C, OH, OW))
    out11 = out11.reshape((N, C, OH, OW))

    ah0s = ah0s.reshape((1, 1, OH, 1))
    ah1s = ah1s.reshape((1, 1, OH, 1))
    aw0s = aw0s.reshape((1, 1, 1, OW))
    aw1s = aw1s.reshape((1, 1, 1, OW))
    out00 = out00 * ah0s * aw0s
    out01 = out01 * ah0s * aw1s
    out10 = out10 * ah1s * aw0s
    out11 = out11 * ah1s * aw1s

    return out00 + out01 + out10 + out11


def resize_nearest_backward(dout: HLOTensor, inp: HLOTensor):
    N, C, OH, OW = dout.shape
    IH, IW = inp.shape[-2:]
    assert IH != 0 and IW != 0 and OH != 0 and OW != 0, f"{inp.shape}, {dout.shape}"

    ihidx = resize_nearest_helper(IH, OH)
    iwidx = resize_nearest_helper(IW, OW)

    ihidx = ihidx.reshape((OH, 1)).broadcast_to((OH, OW))
    iwidx = iwidx.reshape((1, OW)).broadcast_to((OH, OW))
    iidx = ihidx * IW + iwidx
    iidx = iidx.broadcast_to((N, C, OH, OW)).reshape((N, C, -1))

    dim0 = (
        iota(dtype=np.int32, shape=(N,), dimension=-1)
        .reshape((N, 1, 1))
        .broadcast_to(iidx.shape)
    )
    dim1 = (
        iota(dtype=np.int32, shape=(C,), dimension=-1)
        .reshape((1, C, 1))
        .broadcast_to(iidx.shape)
    )
    iidx = stack([dim0, dim1, iidx], axis=-1)

    din = zeros((N, C, IH * IW), dtype=inp.dtype)
    dout = dout.reshape((N, C, OH * OW, 1, 1, 1))
    din = xla_scatter(
        din,
        iidx,
        dout,
        update_window_dims=(3, 4, 5),
        inserted_window_dims=tuple(),
        scattered_dims_to_operand_dims=(0, 1, 2),
        reduce_mode="sum",
    )
    din = din.reshape(inp.shape)
    return din


def resize_linear_backward(dout: HLOTensor, inp: HLOTensor):
    N, C, OH, OW = dout.shape
    IH, IW = inp.shape[-2:]
    assert IH != 0 and IW != 0 and OH != 0 and OW != 0, f"{inp.shape}, {dout.shape}"

    ah0s, ih0s, ah1s, ih1s = resize_linear_helper(IH, OH)
    aw0s, iw0s, aw1s, iw1s = resize_linear_helper(IW, OW)

    ih0s = ih0s.reshape((OH, 1)).broadcast_to((OH, OW))
    ih1s = ih1s.reshape((OH, 1)).broadcast_to((OH, OW))
    iw0s = iw0s.reshape((1, OW)).broadcast_to((OH, OW))
    iw1s = iw1s.reshape((1, OW)).broadcast_to((OH, OW))
    iidx00 = ih0s * IW + iw0s
    iidx01 = ih0s * IW + iw1s
    iidx10 = ih1s * IW + iw0s
    iidx11 = ih1s * IW + iw1s
    iidx00 = iidx00.broadcast_to((N, C, OH, OW)).reshape((N, C, -1))
    iidx01 = iidx01.broadcast_to((N, C, OH, OW)).reshape((N, C, -1))
    iidx10 = iidx10.broadcast_to((N, C, OH, OW)).reshape((N, C, -1))
    iidx11 = iidx11.broadcast_to((N, C, OH, OW)).reshape((N, C, -1))

    dim0 = (
        iota(dtype=np.int32, shape=(N,), dimension=-1)
        .reshape((N, 1, 1))
        .broadcast_to(iidx00.shape)
    )
    dim1 = (
        iota(dtype=np.int32, shape=(C,), dimension=-1)
        .reshape((1, C, 1))
        .broadcast_to(iidx00.shape)
    )
    iidx00 = stack([dim0, dim1, iidx00], axis=-1)
    iidx01 = stack([dim0, dim1, iidx01], axis=-1)
    iidx10 = stack([dim0, dim1, iidx10], axis=-1)
    iidx11 = stack([dim0, dim1, iidx11], axis=-1)

    ah0s = ah0s.reshape((1, 1, OH, 1))
    ah1s = ah1s.reshape((1, 1, OH, 1))
    aw0s = aw0s.reshape((1, 1, 1, OW))
    aw1s = aw1s.reshape((1, 1, 1, OW))
    dout00 = dout * ah0s * aw0s
    dout01 = dout * ah0s * aw1s
    dout10 = dout * ah1s * aw0s
    dout11 = dout * ah1s * aw1s
    dout00 = dout00.reshape((N, C, -1, 1, 1, 1))
    dout01 = dout01.reshape((N, C, -1, 1, 1, 1))
    dout10 = dout10.reshape((N, C, -1, 1, 1, 1))
    dout11 = dout11.reshape((N, C, -1, 1, 1, 1))

    din = zeros((N, C, IH * IW), dtype=inp.dtype)
    din = xla_scatter(
        din,
        iidx00,
        dout00,
        update_window_dims=(3, 4, 5),
        inserted_window_dims=tuple(),
        scattered_dims_to_operand_dims=(0, 1, 2),
        reduce_mode="sum",
    )
    din = xla_scatter(
        din,
        iidx01,
        dout01,
        update_window_dims=(3, 4, 5),
        inserted_window_dims=tuple(),
        scattered_dims_to_operand_dims=(0, 1, 2),
        reduce_mode="sum",
    )
    din = xla_scatter(
        din,
        iidx10,
        dout10,
        update_window_dims=(3, 4, 5),
        inserted_window_dims=tuple(),
        scattered_dims_to_operand_dims=(0, 1, 2),
        reduce_mode="sum",
    )
    din = xla_scatter(
        din,
        iidx11,
        dout11,
        update_window_dims=(3, 4, 5),
        inserted_window_dims=tuple(),
        scattered_dims_to_operand_dims=(0, 1, 2),
        reduce_mode="sum",
    )
    din = din.reshape(inp.shape)
    return din


@register_lower_rule(mops.Resize)
def resize_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert (
        len(args) == 2 and len(ctx.vars_in) == 2
    ), f"Resize should have 2 inputs, get {len(args)}"
    assert (
        len(ctx.vars_out) == 1
    ), f"Resize should have 1 output, get {len(ctx.vars_out)}"

    inp, size = args[0], ctx.vars_in[1].bound_data
    fmt, imode = ctx.op.format, ctx.op.imode
    assert _shape_equal(size.shape, (2,)), f"illegal size {size}"

    if fmt == mops.AdaptivePooling.Format.NCHW:
        size = ctx.vars_out[0].shape[2:]
        if imode == mops.Remap.InterpolationMode.NEAREST:
            return resize_nearest(inp, size)
        elif imode == mops.Remap.InterpolationMode.LINEAR:
            return resize_linear(inp, size)
        else:
            assert False, f"imode {imode} is not supported"
    else:
        assert False, f"format {fmt} is not supported"


# we rewrite the cuda kernel of megdnn to implement resize
@register_lower_rule("ResizeBackward")
def resize_backward_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert (
        len(args) == 2 and len(ctx.vars_in) == 2
    ), f"Resizebackward should have 2 inputs, get {len(args)}"
    assert (
        len(ctx.vars_out) == 1
    ), f"Resizebackward should have 1 output, get {len(ctx.vars_out)}"

    dout, inp = args
    fmt, imode = ctx.param["format"], ctx.param["imode"]

    if fmt == "NCHW":
        if imode == "NEAREST":
            return resize_nearest_backward(dout, inp)
        elif imode == "LINEAR":
            return resize_linear_backward(dout, inp)
        else:
            assert False, f"imode {imode} is not supported"
    else:
        assert False, f"format {fmt} is not supported"
