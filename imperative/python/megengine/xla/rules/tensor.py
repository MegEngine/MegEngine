from typing import Sequence, Union

import numpy as np

from ...core._imperative_rt import ops as mops
from .. import ir_utils
from ..lib.mlir import ir
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


def stack(inps, axis):
    assert len(inps) > 0, f"stack inputs should not be empty"
    inps = [expand_dims(inp, axis) for inp in inps]
    return concat(inps, axis)


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


def fill(value, shape, dtype=np.float32):
    assert isinstance(value, (int, float, bool))
    value = np.asarray(value, dtype=dtype)
    return broadcast_to(HLOTensor(value, dtype=dtype), shape)


def ones(shape, dtype=np.float32):
    return fill(1, shape, dtype)


def zeros(shape, dtype=np.float32):
    return fill(0, shape, dtype)


def fill_like(value, inp: HLOTensor):
    return fill(value, inp.shape, inp.dtype)


def zeros_like(inp: HLOTensor):
    return zeros(inp.shape, inp.dtype)


def ones_like(inp: HLOTensor):
    return ones(inp.shape, inp.dtype)


def where(mask, x, y):
    assert isinstance(mask, HLOTensor), f"mask must be HLOTensor, get {type(mask)}"
    x = x if isinstance(x, HLOTensor) else HLOTensor(x)
    y = y if isinstance(y, HLOTensor) else HLOTensor(y)

    return mask * x + (np.array(1.0).astype(x.dtype) - mask) * y


def where_grad(dout, mask):
    return dout * mask, dout * (np.array(1.0).astype(dout.dtype) - mask)


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


def linspace(
    start: Union[int, float, HLOTensor],
    stop: Union[int, float, HLOTensor],
    num: Union[int, HLOTensor],
    dtype=np.float32,
    oshape=None,
):
    if isinstance(num, HLOTensor):
        assert (
            oshape is not None
        ), "if num is a HLOTensor, please specify the output shape with oshape"

    if isinstance(num, int) and oshape is not None:
        assert _shape_equal(
            oshape, (num,)
        ), f"shape error: shape {oshape} .vs num {num}"

    oshape = (num,) if oshape is None else oshape

    if any([isinstance(x, HLOTensor) for x in [start, stop, num]]):
        start, stop, num = [
            x if isinstance(x, HLOTensor) else HLOTensor(float(x))
            for x in [start, stop, num]
        ]
        start, stop, num = [
            x if x.dtype == np.float32 else x.astype(np.float32)
            for x in [start, stop, num]
        ]
        # we want to implement: scale = where(num==1, 0.0, (stop-start)/(num-1.0))
        # to avoid divided by zero, but xla where is implemented by mul and add, the
        # result is still NAN, so we add a huge number to num to simulate.
        # we use 1e32 rather than np.inf because np.inf * zero will also return NAN
        num_refactor = where(num == 1.0, 1e32, num)
        scale = (stop - start) / (num_refactor - 1.0)
        scale = where(num == 1.0, 0.0, scale)
        offset = start
    else:
        assert all([isinstance(x, (int, float)) for x in [start, stop, num]])
        if num == 1:
            scale = 0
        else:
            scale = (stop - start) / (num - 1.0)
        offset = float(start)

    fsrc = iota(np.float32, oshape, -1)
    fout = fsrc * scale + offset

    return fout if dtype == np.float32 else fout.astype(dtype)


def arange(
    start: Union[int, float, HLOTensor] = 0,
    stop: Union[int, float, HLOTensor] = None,
    step: Union[int, float, HLOTensor] = 1,
    num: int = None,
    dtype="float32",
):
    if stop is None:
        start, stop = 0, start

    if isinstance(step, int):
        assert step != 0, "step should not be zero"

    if all([isinstance(x, (int, float)) for x in [start, stop, step]]):
        assert num is None, "cannot specify num when all args are int or float"
        num = int(np.ceil((stop - start) / step))
        stop = start + step * (num - 1)
    else:
        assert num is not None, "must specify num when hlotensor exists in args"
        stop = start + step * (num - 1)

    return linspace(start, stop, num, dtype=dtype)


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


def xla_gather(
    src: HLOTensor,
    index: HLOTensor,
    slice_sizes: Sequence[int],
    offset_dims: Sequence[int],
    collapsed_slice_dims: Sequence[int],
    start_index_map: Sequence[int],
    indices_are_sorted: bool = False,
):
    """
    This is xla Gather op wrapper. The biggest difference between xla Gather and mge
    Gather is that xla Gather gather blocks each time from source. 
    
    We will use some examples to explain how it works. For more information, you can
    read the hlo doc: https://github.com/openxla/stablehlo/blob/main/docs/spec.md.
    
    ## Case1: What does `src`, `index`, `slice_sizes` mean?
    We have `src` Tensor with shape (4, 5) as below:
        src = [[ 0  1  2  3  4]
               [ 5  6  7  8  9]
               [10 11 12 13 14]
               [15 16 17 18 19]]
    We have `index` Tensor with shape (2,) as below:
        index = [1, 2]
    We set the `slice_sizes` in arglist as (2, 3):
    That means we want to gather a block with shape (2, 3) at `src[1, 2]`, it returns:
        out = [[ 7  8  9]
               [12 13 14]]
    The `out.shape` is (2, 3). The last two dimensions of `out` are same as
    `slice_sizes`.
    
    The last dimension of the `index` will be interpreted as coordinate to indicate
    where we start to gather block. The length of the last dimension of `index` is 2
    because the `src` is 2-d tensor. In this case, we start gather at
    `src[index[0], index[1]]`. 

    Here the `slice_sizes` determine the size of the block to gather. Because the `src`
    is 2-d tensor, the length of `slice_sizes` is also 2. Each element of `slice_sizes`
    corresponds to a dimension of `src`. In this case, we gather block of size (2, 3).

    Factually, the gather operation above is equal to:
        out = src[index[0]:index[0]+slice_sizes[0], index[1]:index[1]+slice_sizes[1]].

    ## Case2: How to do batch gather?
    The source Tensor is the same as above with shape (4, 5).
    The index Tensor with shape (3, 2) as below:
        index = [[0 1]
                 [2 1]
                 [1 2]]
    The `slice_sizes` is also set as (2, 3).
    That means we want to gather 3 block with shape (2, 3) from `src[0, 1]`, 
    `src[2, 1]`, `src[1, 2]`it returns:
        out = [[[ 1  2  3]
                [ 6  7  8]]
               [[11 12 13]
                [16 17 18]]
               [[ 7  8  9]
                [12 13 14]]].
    The `out.shape` is (3, 2, 3). The last two dimensions of `out` are same as
    `slice_sizes`. The other dimensions of `out` are same as `index.shape[:-1]`.
    
    The `index.shape[:-1]` determine how times gather we will perform. If the shape of
    `index` is (2, 3, 4, 5, 2), then the `out` shape will be (2, 3, 4, 5, 2, 3).

    Factually, the gather operation above is equal to:
        out[i, :] = src[
            index[i][0]:index[i][0]+slice_sizes[0], 
            index[i][1]:index[i][1]+slice_sizes[1]
        ].

    ## Case3: What does `offset_dims` do?
    The `offset_dims` can map the dimension of the gather block to the dimension of the
    output.

    For example, if we have `src` with shape (8, 12), `index` with shape (5, 7, 2),
    `slice_sizes` with value (3, 6).
    If we set `offset_dims` to (2, 3), then the output shape is (5, 7, 3, 6). The `3` in
    `slice_sizes` will be at the second dimension of output. The `6` in the `slice_sizes`
    will be at the third dimension of output.
    If we set `offset_dims` to (0, 1), then the output shape is (3, 6, 5, 7).
    If we set `offset_dims` to (0, 2), then the output shape is (3, 5, 6, 7).

    These outputs with different shapes can tranpose into each other.
    
    Here are three constraint of `offset_dims`:
        * The `len(offset_dims)` must equal to `len(slice_sizes)`.
        * The `offset_dims` cannot have replicated elements.
        * Each element within `offset_dims` should smaller than `out.ndim`.
        * The elements in `offset_dims` should be sorted ascending.

    ## Case4: What does `start_index_map` do?
    The `start_index_map` determine how to resolve the index when gathering. 
    
    For example, supposing we have `index` with value (x, y).
    If the `start_index_map` is (0, 1), then we gather block from src[x, y].
    If the `start_index_map` is (1, 0), then we gather block from src[y, x].

    More complex examples: supposing we have `index` with value (a, b, c, d).
    If the `start_index_map` is (0, 1, 2, 3), then we gather block from src[a, b, c, d].
    If the `start_index_map` is (0, 3, 2, 1), then we gather block from src[a, d, c, b].

    ## Case5: What does `collapsed_slice_dims` do?
    We have `src` Tensor with shape (4, 5) as below:
        src = [[ 0  1  2  3  4]
               [ 5  6  7  8  9]
               [10 11 12 13 14]
               [15 16 17 18 19]]
    The index Tensor with shape (3, 2) as below:
        index = [[0 1]
                 [2 1]
                 [1 2]]
    We but we set the `slice_sizes` in arglist as (1, 2):
    In general, it will return
        out = [[[ 1  2]]
               [[11 12]]
               [[ 7  8]]]
    The `out.shape` is (3, 1, 2), the gather block size is (1, 2).

    If we set `collapsed_slice_dims` as (0,), then the `out.shape` is (3, 2). That
    means the dim0 of the `slice_sizes` will be collpased implicitly and the
    `slice_sizes` can be considered as (2,). The gather block size is (2,).

    If you have set `collapsed_slice_dims`, the `offset_dims` should be changed
    correpondingly because the dimension of gather block is collapsed. For example, the
    `offset_dims` should be changed from (1, 2) to (1,).

    ## Case6: Must index.shape[-1] and src.ndim be equal?
    In above cases, if `src.ndim` is 2, then the `index.shape` will be set as
    [x, ..., y, 2] and the last dimension of `index` will be interpreted as coordinate.
    But it is not necessary.

    We have `src` Tensor with shape (4, 5) as below:
        src = [[ 0  1  2  3  4]
               [ 5  6  7  8  9]
               [10 11 12 13 14]
               [15 16 17 18 19]]
    We have `index` Tensor with shape (2, 1) as below:
        index = [[2]
                 [1]]
    The `slice_sizes` is also set as (2, 3).
    
    If `start_index_map` is set to (0,):
        out = [[[10 11 12]
                [15 16 17]]
               [[ 5  6  7]
                [10 11 12]]]
    That means we gather twice at src[2, 0] and src[1, 0]. The `index` is extend from 
    (2, 1) to (2, 2) with zero. The `start_index_map` determine how to extend.
    If `start_index_map` is set to (1,):
        out = [[[2 3 4]
                [7 8 9]]
               [[1 2 3]
                [6 7 8]]]
    That means we gather twice at src[0, 2] and src[0, 1].
    """
    gather_dnums = hlo.GatherDimensionNumbers.get(
        offset_dims=list(offset_dims),
        collapsed_slice_dims=list(collapsed_slice_dims),
        start_index_map=list(start_index_map),
        index_vector_dim=index.ndim - 1,
    )

    op = hlo.GatherOp(
        src.tensor,
        index.tensor,
        gather_dnums,
        indices_are_sorted=ir.BoolAttr.get(indices_are_sorted),
        slice_sizes=ir_utils.dense_int_elements(slice_sizes),
    )

    return HLOTensor(op.results)


def xla_scatter(
    dst: HLOTensor,
    index: HLOTensor,
    src: HLOTensor,
    update_window_dims: Sequence[int],
    inserted_window_dims: Sequence[int],
    scattered_dims_to_operand_dims: Sequence[int],
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    reduce_mode: str = None,
):
    """
    This is xla Scatter op wrapper, which is inverse process of Gather. Please read the
    docstring of Gather op before reading the below contents.

    We will use some examples to explain how it works. For more information, you can
    read the hlo doc: https://github.com/openxla/stablehlo/blob/main/docs/spec.md.

    ## Case1: What does `src`, `index`, `dst`, `reduce_mode`, `update_window_dims` do?
    We have `src` Tensor with shape (2, 3) as below:
        src = [[0 1 2]
               [3 4 5]]
    We have `index` Tensor with shape (2,):
        index = [1, 2]
    We have `dst` Tensor is zeros(size=(4, 5)).

    The `update_window_dims` is (0, 1) here. `update_window_dims` determine which dim
    of `src` are combined into the scatter block. In this case, it is the dim0 and dim1.
    Because the `src` is only 2-d tensor, it means the whole src will be seen as 2-d
    block to scatter.

    The last dimension of the `index` is also interpreted as coordinate to indicate
    where we start to place the scattered block. The length of the last dimension of
    `index` is 2 because the `dst` is 2-d tensor in this case. We place the scattered
    block at `dst[index[0], index[1]]`. it returns:
        out = [[0 0 0 0 0]
               [0 0 0 1 2]
               [0 0 3 4 5]
               [0 0 0 0 0]]
    The `reduce_mode` control how to combine `dst` and the scattered block.
    If `reduce_mode` is None, dst[index] = scattered[...].
    If `reduce_mode` is "sum", dst[index] += scattered[...].
    If `reduce_mode` is "min", dst[index] = min(dst[index], scattered[...]).
    If `reduce_mode` is "max", dst[index] = max(dst[index], scattered[...]).

    Here we always set `reduce_mode` as `sum`.

    Factually, the scatter operation above is equal to:
        scatter_block_shape = src.shape[update_window_dims] # (2, 3)
        dst[index[0]:index[0]+scatter_block_shape[0], index[1]:index[1]+scatter_block_shape[1]] += src.
    
    ## Case2: How to do batch scatter with `update_window_dims`?
    We have `src` Tensor with shape (2, 2, 3) as below:
        src = [[[ 0  1  2]
                [ 3  4  5]]
               [[ 6  7  8]
                [ 9 10 11]]]
    We have `index` Tensor with shape (2, 2):
        index = [[0 1]
                 [2 0]]
    We have `dst` Tensor is zeros(size=(4, 5)).

    The `update_window_dims` is (1, 2) here, it means the dim1 and dim2 of `src` will be
    combined into the scatter block. So the block size is (2, 3). The other dimensions
    of `src` determine how many times scatter we should perform.

    So the above case means there are two (2, 3) scatter block in `src`, two sets of
    coordinates in `index`, we scatter the `src[0]` into `dst[index[0]]`, and scatter
    the `src[1]` into `dst[index[1]]`. It returns:
        out = [[ 0  0  1  2  0]
               [ 0  3  4  5  0]
               [ 6  7  8  0  0]
               [ 9 10 11  0  0]]

    The dims of `src` which is not in `update_window_dims` should equal to
    `index.shape[:-1]`. In this case, `src.shape` is (2, 2, 3). After removing dim1 and
    dim2 in `update_window_dims`, it is (2,), which is equal to `index.shape[:-1]`.

    Factually, the scatter operation above is equal to:
        scatter_block_shape = src.shape[update_window_dims] # (2, 3)
        out[
            index[i][0]:index[i][0]+scatter_block_shape[0],
            index[i][1]:index[i][1]+scatter_block_shape[1]
        ] += src.

    ## Case3: What does `scattered_dims_to_operand_dims` do?
    The `scattered_dims_to_operand_dims` determine how to resolve the index when
    scattering. 
    
    For example, supposing we have `index` with value (x, y).
    If the `scattered_dims_to_operand_dims` is (0, 1), then we scattered block will be
    place at dst[x, y].
    If the `scattered_dims_to_operand_dims` is (1, 0), then we scattered block will be
    place at dst[y, x].

    More complex examples: supposing we have `index` with value (a, b, c).
    If the `scattered_dims_to_operand_dims` is (0, 1, 3), then we scattered block will be
    place at dst[a, b, :, c].
    If the `scattered_dims_to_operand_dims` is (0, 3, 2), then we scattered block will be
    place at dst[a, :, c, b].

    ## Case4: What does `inserted_window_dims` do?
    In above cases, the dimension of scattered block is the same as `dst.ndim`. For
    example, we have scattered block of shape (2, 3) and `dst` of shape (4, 5). The
    placement of scattered block is trivial. Scattered block dim0 match `dst` dim0.
    Scattered block dim1 match `dst` dim1. But if `dst.ndim` is not equal to block
    dimension, how we can match thest dimension? 
    
    `inserted_window_dims` indicates which dim of `dst` not participating in the process
    of dimension matching. For example, we have block of (3,) and `dst` of shape (4, 5).
    
    If `inserted_window_dims` is (0,), block dim0 match `dst` dim1, block will be placed
    along `dst` dim1.
    If `inserted_window_dims` is (1,), block dim0 match `dst` dim0, block will be placed
    along `dst` dim0.

    We have `src` Tensor with shape (2, 3) as below:
        src = [[ 0  1  2]
               [ 3  4  5]]
    We have `index` Tensor with shape (2, 2):
        index = [[0 1]
                 [1 2]]
    We have `dst` Tensor is zeros(size=(4, 5)).

    If `inserted_window_dims` is (0,), block placed along `dst` dim1, it returns:
        out = [[0 0 1 2 0]
               [0 0 3 4 5]
               [0 0 0 0 0]
               [0 0 0 0 0]]
    If `inserted_window_dims` is (1,), block placed along `dst` dim0, it returns:
        out = [[0 0 0 0 0]
               [0 1 3 0 0]
               [0 2 4 0 0]
               [0 0 5 0 0]]
    
    ## Case5: Must index.shape[-1] and dst.ndim be equal?
    In above cases, if `dst.ndim` is 2, then the `index.shape` will be set as
    [x, ..., y, 2] and the last dimension of `index` will be interpreted as coordinate.
    But it is not necessary.

    We have `src` Tensor with shape (2, 2, 3) as below:
        src = [[[ 0  1  2]
                [ 3  4  5]]
               [[ 6  7  8]
                [ 9 10 11]]]
    We have `index` Tensor with shape (2, 1) as below:
        index = [[2]
                 [1]]
    The `update_window_dims` is also set as (1, 2).
    
    If `scatter_dims_to_operand_dims` is set to (1,):
        out = [[ 0  6  7  9  2]
               [ 0  9 13 15  5]
               [ 0  0  0  0  0]
               [ 0  0  0  0  0]]
    That means we scatter `src[0]` to `dst[0, 2]` and `src[1]` to `dst[0, 1]`. The
    `index` is extend from (2, 1) to (2, 2) with zero. The `scatter_dims_to_operand_dims`
    determine how to extend.
    If `scatter_dims_to_operand_dims` is set to (0,):
        out = [[ 0  0  0  0  0]
               [ 6  7  8  0  0]
               [ 9 11 13  0  0]
               [ 3  4  5  0  0]]
    That means we scatter twice at `dst[2, 0]` and `dst[1, 0]`.
    """
    scatter_dnums = hlo.ScatterDimensionNumbers.get(
        update_window_dims=list(update_window_dims),
        inserted_window_dims=list(inserted_window_dims),
        scattered_dims_to_operand_dims=list(scattered_dims_to_operand_dims),
        index_vector_dim=index.ndim - 1,
    )

    oshape, odtype = dst.shape, dst.dtype
    op = hlo.ScatterOp(
        ir_utils.make_ir_type_according_meta_tuple(oshape, odtype),
        [dst.tensor],
        index.tensor,
        [src.tensor],
        scatter_dnums,
        indices_are_sorted=ir.BoolAttr.get(indices_are_sorted),
        unique_indices=ir.BoolAttr.get(unique_indices),
    )

    scalar_type = ir_utils.make_ir_type_according_meta(tuple(), odtype)
    update = op.update_computation.blocks.append(scalar_type, scalar_type)
    with ir.InsertionPoint(update):
        if reduce_mode is not None:
            reduce_mode_to_op = {"sum": hlo.AddOp, "min": hlo.MinOp, "max": hlo.MaxOp}
            reduce_op = reduce_mode_to_op[reduce_mode]
            add = reduce_op(*update.arguments)
            hlo.ReturnOp(add.results)
        else:
            hlo.ReturnOp((update.arguments[1],))

    return HLOTensor(op.results)


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


@register_lower_rule("AxisAddRemove")
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


@register_lower_rule(mops.Where)
def where_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert len(args) == 3 and len(ctx.vars_in) == 3 and len(ctx.vars_out) == 1
    assert _shape_equal(ctx.vars_in[0].shape, ctx.vars_in[1].shape) and _shape_equal(
        ctx.vars_in[0].shape, ctx.vars_in[2].shape
    )
    return where(args[0], args[1], args[2])


@register_lower_rule("WhereBackward", mops.WhereBackward)
def where_backward_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert len(args) == 2 and len(ctx.vars_in) == 2 and len(ctx.vars_out) == 2
    assert _shape_equal(ctx.vars_in[0].shape, ctx.vars_in[1].shape)
    return where_grad(args[0], args[1])


@register_lower_rule(mops.Stack)
def stack_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    return stack(args, ctx.op.axis)


@register_lower_rule(mops.Linspace)
def linspace_lower(ctx, *args: Union[HLOTensor, Sequence[HLOTensor]]):
    assert len(args) == 3 and len(ctx.vars_in) == 3 and len(ctx.vars_out) == 1
    assert len(ctx.vars_out[0].shape) == 1, "linspace/arange only support 1D"

    odtype, oshape = ctx.vars_out[0].dtype, ctx.vars_out[0].shape
    return linspace(args[0], args[1], args[2], odtype, oshape)
