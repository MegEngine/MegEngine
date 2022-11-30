import os
from functools import lru_cache, partial, reduce
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from ..core.tensor.utils import subgraph_fn
from ..tensor import Tensor
from .math import matmul, sum
from .tensor import arange, broadcast_to, reshape, transpose

EinsumDimension = str
EinsumShape = Tuple[EinsumDimension, ...]


# Different einsum implementations, 0 means using subgraph, 1 means using interpret
# The former is faster but may have some unknown bugs.
# When the run fails, we recommend using the latter(by setting this variable to 1) to run your code again.
einsum_impl = int(os.environ.get("MGE_EINSUM_IMPL", "0"))


class EinsumContext:
    dims2val: Callable[[str], Any]
    reshape: Callable[[Any, tuple], Any]
    broadcast: Callable[[Any, tuple], Any]
    transpose: Callable[[Any, List[int]], Any]
    reduce: Callable[[Any, List[int]], Any]
    matmul: Callable[[Any, Any], Any]
    diag_plane: Callable[[Any, int, List[int]], Any]


class EinsumOperand:
    def __init__(self, shape: Optional[EinsumShape], tracer, ctx: EinsumContext):
        self._shape = shape
        self._tracer = tracer
        self._ctx = ctx

    @property
    def shape(self) -> EinsumShape:
        assert self._shape is not None
        return self._shape

    def reshape(self, shape: EinsumShape) -> "EinsumOperand":
        if shape == self._shape:
            return self
        shape_val = tuple(map(lambda x: self._ctx.dims2val(x), shape))
        return EinsumOperand(
            shape, self._ctx.reshape(self._tracer, shape_val), self._ctx
        )

    def broadcast(self, shape: EinsumShape) -> "EinsumOperand":
        if shape == self._shape:
            return self
        for dim in shape:
            assert len(dim) == 1
        shape_val = tuple(map(lambda x: self._ctx.dims2val(x), shape))
        return EinsumOperand(
            shape, self._ctx.broadcast(self._tracer, shape_val), self._ctx
        )

    def transpose(self, shape: EinsumShape) -> "EinsumOperand":
        if shape == self._shape:
            return self
        for dim in shape:
            assert len(dim) == 1
        assert len(shape) == len(self.shape)
        axis = [*map(lambda x: self.shape.index(x), shape)]
        return EinsumOperand(shape, self._ctx.transpose(self._tracer, axis), self._ctx)

    def reduce(self, shape: EinsumShape) -> "EinsumOperand":
        if shape == self._shape:
            return self
        for dim in shape:
            assert len(dim) == 1
            assert dim in self.shape
        axis = [*filter(lambda x: self.shape[x] not in shape, range(len(self.shape)))]
        assert tuple([*filter(lambda x: x in shape, self.shape)]) == shape
        return EinsumOperand(shape, self._ctx.reduce(self._tracer, axis), self._ctx)

    # TODO: Some scenarios should be re-implemented using dnn kernel
    def diag_plane(self, shape: EinsumShape) -> "EinsumOperand":
        # iiijkpppqqqrpppqqq -> ikjprq (may reorder)
        if shape == self._shape:
            return self
        for dim in shape:
            assert len(dim) == 1
            assert dim in self.shape
            assert shape.count(dim) == 1
        acc = self
        while True:
            dup_found = False
            for dim in acc.shape:
                if acc.shape.count(dim) == 1:
                    continue
                dup_found = True
                axes = [i for i, e in enumerate(acc.shape) if e == dim]
                acc = EinsumOperand(
                    (dim,) + tuple([e for e in acc.shape if e != dim]),
                    acc._ctx.diag_plane(acc._tracer, len(acc.shape), axes),
                    acc._ctx,
                )
            if not dup_found:
                return acc

    def dedup_dim(self) -> "EinsumOperand":
        return self.diag_plane(tuple(set(self.shape)))

    def __matmul__(self, rhs: "EinsumOperand") -> "EinsumOperand":
        assert len(self.shape) == 3 and len(rhs.shape) == 3
        assert self.shape[0] == rhs.shape[0]
        assert self.shape[2] == rhs.shape[1]
        output_shape = (self.shape[0], self.shape[1], rhs.shape[2])
        # TODO: maybe not matmul
        return EinsumOperand(
            output_shape, self._ctx.matmul(self._tracer, rhs._tracer), self._ctx
        )


def einsum_matmul(
    lhs: EinsumOperand, rhs: EinsumOperand, batch_dims, sum_dims, left_dims, right_dims
):
    lshape = lhs.shape
    rshape = rhs.shape
    broadcast_lhs = 0
    broadcast_rhs = 0
    for dim in batch_dims:
        if dim not in lshape:
            broadcast_lhs += 1
            lshape = lshape + (dim,)
        if dim not in rshape:
            broadcast_rhs += 1
            rshape = rshape + (dim,)
    # always use matmul
    for dim in sum_dims:
        if dim not in lshape:
            broadcast_lhs += 1
            lshape = lshape + (dim,)
        if dim not in rshape:
            broadcast_rhs += 1
            rshape = rshape + (dim,)

    if broadcast_lhs:
        lhs = lhs.broadcast(lshape)

    if broadcast_rhs:
        rhs = rhs.broadcast(rshape)

    lhs = lhs.transpose(batch_dims + left_dims + sum_dims)
    rhs = rhs.transpose(batch_dims + sum_dims + right_dims)

    lhs = lhs.reshape(("".join(batch_dims), "".join(left_dims), "".join(sum_dims)))
    rhs = rhs.reshape(("".join(batch_dims), "".join(sum_dims), "".join(right_dims)))
    # matmul TODO: sometimes unnecessary
    oup = (lhs @ rhs).reshape(batch_dims + left_dims + right_dims)
    return oup


def einsum_infer_output_shape(shapes):
    ellipsis_found = "." in (dim for shape in shapes for dim in shape)
    output_shape = (".",) if ellipsis_found else ()
    dims = {"."}
    dup_dims = {"."}
    for shape in shapes:
        for dim in shape:
            if dim in dims:
                dup_dims.add(dim)
            else:
                dims.add(dim)
    unique_shapes = set()
    for shape in shapes:
        for dim in shape:
            if dim not in dup_dims:
                unique_shapes.add(dim)
    unique_shapes = tuple(sorted(unique_shapes))
    return output_shape + unique_shapes


def einsum_remove_ellipsis(
    shapes: List[EinsumShape], output_shape: EinsumShape, input_ndims: Tuple[int, ...]
):
    dims = set(dim for shape in shapes for dim in shape)

    def get_free_dim():
        import string

        for ch in string.ascii_letters:
            if ch not in dims:
                dims.add(ch)
                return ch
        assert False, "no free dim left"

    ellipsis_dims = None

    def replace_ellipsis(shape):
        assert shape.count(".") == 1, ""
        idx = shape.index(".")
        shape = shape[:idx] + ellipsis_dims + shape[idx + 1 :]
        return shape

    for i in range(len(shapes)):
        shape, ndim = shapes[i], input_ndims[i]
        if "." in shape:
            if not ellipsis_dims:
                len_ellipsis = ndim - (len(shape) - 1)
                ellipsis_dims = tuple(
                    map(lambda _: get_free_dim(), range(len_ellipsis))
                )
            shapes[i] = replace_ellipsis(shape)

    if ellipsis_dims:
        output_shape = replace_ellipsis(output_shape)

    return shapes, output_shape


def einsum_parse_equation_only(
    equation: str,
) -> Tuple[List[EinsumShape], Optional[EinsumShape]]:
    pos = 0
    shapes = []
    cur_shape = []
    output_found = False

    def append_shape(dim):
        cur_shape.append(dim)

    while pos < len(equation):
        ch = equation[pos]
        if ch == ",":
            assert not output_found
            shapes.append(tuple(cur_shape))
            cur_shape = []
            pos += 1
        elif ch == ".":
            assert pos + 2 < len(equation)
            assert equation[pos : pos + 3] == "..."
            assert "." not in cur_shape, "duplicated ellipsis"
            append_shape(".")
            pos += 3
        elif ch == "-":
            assert not output_found
            assert pos + 1 < len(equation)
            assert equation[pos : pos + 2] == "->"
            shapes.append(tuple(cur_shape))
            cur_shape = []
            output_found = True
            pos += 2
        elif ch == " ":
            pos += 1
        else:
            assert str.islower(ch) or str.isupper(ch)
            append_shape(ch)
            pos += 1

    if output_found:
        output_shape = tuple(cur_shape)
    else:
        shapes.append(tuple(cur_shape))
        output_shape = None
    return shapes, output_shape


def einsum_parse_equation(
    equation: str, input_ndims: Tuple[int, ...]
) -> Tuple[Tuple[EinsumShape, ...], EinsumShape]:
    shapes, output_shape = einsum_parse_equation_only(equation)

    if output_shape is None:
        output_shape = einsum_infer_output_shape(shapes)

    shapes, output_shape = einsum_remove_ellipsis(shapes, output_shape, input_ndims)

    return tuple(shapes), output_shape


def einsum_impl(shapes, output_shape, inputs, ctx: EinsumContext):
    assert len(shapes) == len(inputs), "input size mismatch"
    dim2firstop = dict()
    dim2lastop = dict()
    dims = set()
    for i, shape in enumerate(shapes):
        for dim in shape:
            if dim not in output_shape:
                dim2lastop[dim] = i
                if dim not in dim2firstop:
                    dim2firstop[dim] = i
            dims.add(dim)
    result = EinsumOperand(shapes[0], inputs[0], ctx).dedup_dim()
    for i in range(1, len(inputs)):
        rhs = EinsumOperand(shapes[i], inputs[i], ctx).dedup_dim()
        lshape = result.shape
        batch_dims: Tuple[EinsumDimension, ...] = ()
        sum_dims: Tuple[EinsumDimension, ...] = ()
        left_dims: Tuple[EinsumDimension, ...] = ()
        right_dims: Tuple[EinsumDimension, ...] = ()
        reduce_dims: Tuple[EinsumDimension, ...] = ()
        lshape = result.shape
        rshape = rhs.shape
        for dim in lshape:
            lastop = dim2lastop.get(dim, -1)
            if lastop == i - 1:
                reduce_dims = reduce_dims + (dim,)
            elif lastop == i:
                sum_dims = sum_dims + (dim,)
            else:
                if dim in rshape:
                    batch_dims = batch_dims + (dim,)
                else:
                    left_dims = left_dims + (dim,)
        for dim in rshape:
            if dim not in lshape:
                lastop = dim2lastop.get(dim, -1)
                if lastop == i:
                    reduce_dims = reduce_dims + (dim,)
                else:
                    right_dims = right_dims + (dim,)
        result = result.reduce(
            tuple(filter(lambda x: x not in reduce_dims, result.shape))
        )
        rhs = rhs.reduce(tuple(filter(lambda x: x not in reduce_dims, rhs.shape)))
        result = einsum_matmul(result, rhs, batch_dims, sum_dims, left_dims, right_dims)
    result = result.reduce(
        tuple(filter(lambda x: x in output_shape, result.shape))
    ).transpose(output_shape)
    return result._tracer


def diag_plane_interpret(inp: Tensor, inp_ndim: int, axes: List[int]) -> Tensor:
    r"""Get the diagonal plane of input tensor along given axes.

    This function is primarily used internally by einsum.

    Args:
        inp: input tensor.
        axes: axes forming the square which will derive the diagonal vector;
            they are assumed to have same length.

    Returns:
        a tensor, whose first dimension corresponds to the diagonal vector,
        and remaining dimensions have the same order as dimensions in ``inp``
        which are not in ``axes``.
    """
    all_axes = set(range(inp_ndim))
    remaining_axes = sorted(all_axes.difference(axes))
    transposed = transpose(inp, list(axes) + remaining_axes)
    diag_len = inp.shape[axes[0]]
    return transposed[tuple(arange(diag_len, dtype=np.int32) for _ in range(len(axes)))]


def diag_plane_subgraph(
    inp: Tensor, inp_ndim: int, axes: List[int], f: Any, c: Any
) -> Any:
    from megengine.core.ops import builtin

    all_axes = set(range(inp_ndim))
    remaining_axes = sorted(all_axes.difference(axes))
    transposed = f(builtin.Dimshuffle(list(axes) + remaining_axes), inp)
    diag_len = f(builtin.GetVarShape(axis=axes[0]), inp)
    mav_arg = list(map(lambda x: (x, False, False, False, True), range(len(axes))))
    mav_index = lambda: f(
        builtin.TypeCvt("int32"),
        f(
            builtin.Linspace(),
            c(0),
            f(builtin.Elemwise("SUB"), diag_len, c(1)),
            diag_len,
        ),
    )
    mav_indices = [mav_index() for _ in range(len(axes))]
    return f(builtin.IndexingMultiAxisVec(mav_arg), transposed, *mav_indices)


def einsum_interpret(equation, inputs):
    input_ndims = tuple(map(lambda x: x.ndim, inputs))
    shapes, output_shape = einsum_parse_equation(equation, input_ndims)
    ctx = EinsumContext()
    dim2val_cache = {}

    def dim2val(dim: str):
        if dim in dim2val_cache:
            return dim2val_cache[dim]
        for i, shape in enumerate(shapes):
            if dim in shape:
                dim2val_cache[dim] = inputs[i].shape[shape.index(dim)]
                break
        return dim2val_cache[dim]

    def dims2val(dims: str):
        if len(dims) == 0:
            return 1
        return reduce(lambda x, y: x * y, map(dim2val, dims))

    ctx.dims2val = dims2val
    ctx.reduce = lambda x, axis: sum(x, axis=axis)
    ctx.reshape = reshape
    ctx.matmul = matmul
    ctx.broadcast = broadcast_to
    ctx.transpose = transpose
    ctx.diag_plane = diag_plane_interpret
    return einsum_impl(shapes, output_shape, inputs, ctx)


@lru_cache(maxsize=None)
def _get_einsum_op(
    equation: str, dtype, device, input_ndims
) -> Callable[[Tuple[Tensor, ...]], Tensor]:
    @subgraph_fn(
        "Einsum", dtype=dtype, device=device, nr_inputs=len(input_ndims), gopt_level=2
    )
    def einsum(inputs, f, c):
        from megengine.core.ops import builtin

        shapes, output_shape = einsum_parse_equation(equation, input_ndims)
        ctx = EinsumContext()
        dim2val_cache = {}

        def dim2val(dim: str):
            if dim in dim2val_cache:
                return dim2val_cache[dim]
            for i, shape in enumerate(shapes):
                if dim in shape:
                    axis = shape.index(dim)
                    dim2val_cache[dim] = f(builtin.GetVarShape(axis=axis), inputs[i])
                    break
            return dim2val_cache[dim]

        def dims2val(dims: str):
            if len(dims) == 0:
                return c(1, dtype=np.int32, device=device)
            return reduce(lambda x, y: f("*", x, y), map(dim2val, dims))

        def concat_dims(dims):
            if len(dims) == 0:
                return c([1], dtype=np.int32, device=device)
            else:
                shape = dims[0]
                for dim in dims[1:]:
                    shape = f(builtin.Concat(axis=0, comp_node=device), shape, dim)
                return shape

        def reduce_sum(x, axis):
            if len(axis) == 0:
                return x
            for i in reversed(axis):
                x = f(builtin.Reduce(mode="sum", axis=i), x)
            x = f(builtin.RemoveAxis(axis=[*reversed(axis)]), x)
            return x

        ctx.dims2val = dims2val
        ctx.reduce = reduce_sum
        ctx.reshape = lambda x, dims: f(builtin.Reshape(), x, concat_dims(dims))
        ctx.matmul = lambda x, y: f(builtin.BatchedMatrixMul(), x, y)
        ctx.broadcast = lambda x, dims: f(builtin.Broadcast(), x, concat_dims(dims))
        ctx.transpose = lambda x, axis: f(builtin.Dimshuffle(axis), x)
        ctx.diag_plane = partial(diag_plane_subgraph, f=f, c=c)
        return (einsum_impl(shapes, output_shape, inputs, ctx),), (True,)

    return einsum


def einsum_subgraph(equation, inputs: Tuple[Tensor, ...]) -> Tensor:
    dtype = inputs[0].dtype
    device = inputs[0].device
    # assume all inputs has same dtype and device
    for input in inputs[1:]:
        assert input.dtype == dtype
        assert input.device == device
    einsum = _get_einsum_op(
        equation, dtype, device, tuple(map(lambda x: x.ndim, inputs))
    )
    return einsum(*inputs)[0]


def einsum(equation: str, *args: Tensor) -> Tensor:
    if einsum_impl == 0:
        return einsum_subgraph(equation, args)
    return einsum_interpret(equation, args)
