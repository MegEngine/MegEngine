# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Iterable

import numpy as np

from .._trace_option import use_tensor_shape
from ..ops import builtin
from ..ops.special import Const
from .core import TensorBase, TensorWrapperBase, apply
from .utils import astensor1d, make_shape_tuple


def remove_ellipsis(tensor, tuple_val):
    ndim_sum = tensor.ndim
    cur_sum = 0
    pos = -1
    for i_idx, i in enumerate(tuple_val):
        if i is Ellipsis:
            for j in tuple_val[:i_idx:-1]:
                if j is Ellipsis:
                    raise IndexError("only one ellipsis is allowed")
            pos = i_idx
        else:
            cur_sum += i.ndim if hasattr(i, "ndim") else 1
    if pos == -1:
        return tuple_val
    else:
        return (
            tuple_val[:pos]
            + (slice(None, None, None),) * (ndim_sum - cur_sum)
            + tuple_val[pos + 1 :]
        )


# XXX: assume same results during trace
def check_bool_index(tensor, tuple_val):
    cur_shape = make_shape_tuple(tensor.shape)
    new_tuple_val = []
    offset = 0
    tdim = 0
    for idx, i in enumerate(tuple_val):
        if hasattr(i, "dtype") and i.dtype == np.bool_:
            if i.ndim > 1:
                tot = i.ndim
                ishape = make_shape_tuple(i.shape)
                for j in range(i.ndim):
                    if cur_shape[tdim + j - offset] != ishape[j]:
                        raise IndexError(
                            "boolean index did not match tensor along dimension {}; dimension is {} but corresponding boolean dimension is {}".format(
                                tdim + j, cur_shape[tdim + j - offset], ishape[j]
                            )
                        )
                i = i.reshape(-1)
                if not use_tensor_shape():
                    cur_shape = (
                        cur_shape[:idx]
                        + (i.shape[0],)
                        + cur_shape[tdim + tot - offset :]
                    )
                else:
                    # XXX: use only for trace
                    new_shape = []
                    for ii in range(idx):
                        new_shape.append(tensor.shape[ii])
                    new_shape.append(i.shape[0])
                    for ii in range(tdim + tot - offset, len(cur_shape)):
                        new_shape.append(cur_shape[ii])
                    cur_shape = astensor1d(new_shape)
                offset += 1
                tensor = tensor.reshape(cur_shape)
                tdim += tot
                if use_tensor_shape():
                    cur_shape = make_shape_tuple(cur_shape)
            new_tuple_val.append(i)
        else:
            new_tuple_val.append(i)
            tdim += 1
    return tensor, new_tuple_val


def unpack_getitem(inp, tuple_val, *, allow_newaxis=True):
    if not isinstance(tuple_val, tuple):
        tuple_val = (tuple_val,)
    ndim_indexed = 0
    for i in tuple_val:
        if not i is Ellipsis:
            ndim_indexed += 1 if not hasattr(i, "ndim") else i.ndim
    if ndim_indexed > inp.ndim:
        raise IndexError(
            "too many indices for tensor: tensor is {}-dimensional, but {} were indexed".format(
                inp.ndim, ndim_indexed
            )
        )

    tuple_val = remove_ellipsis(inp, tuple_val)
    use_subtensor = True
    inp, tuple_val = check_bool_index(inp, tuple_val)

    def is_scalar(d):
        if isinstance(i, int):
            return True
        if type(d).__module__ == np.__name__:
            return np.isscalar(d)
        # if isinstance(d, (TensorBase, TensorWrapperBase)):
        #     return d.shape == (1,)
        return False

    new_axes = []
    tensors = []
    items = []
    cur_axis = -1
    for i_idx, i in enumerate(tuple_val):
        cur_axis += 1
        if i is np.newaxis:
            if cur_axis >= 0:
                new_axes.append(cur_axis)
            continue

        if i is Ellipsis:
            cur_axis = -1
            for j in tuple_val[:i_idx:-1]:
                if j is Ellipsis:
                    raise IndexError("only one ellipsis is allowed")
                if j is np.newaxis:
                    new_axes.append(cur_axis)
                cur_axis -= 1
            continue

        if (
            not is_scalar(i)
            and not i is np.newaxis
            and not i is Ellipsis
            and not isinstance(i, slice)
        ):
            use_subtensor = False

        item = [
            cur_axis,
        ]

        def is_bool_list(x):
            if not isinstance(x, list):
                return False
            for i in x:
                if not isinstance(i, bool):
                    return False
            return True

        def get_index(i):
            if not isinstance(i, (TensorBase, TensorWrapperBase)):
                if is_bool_list(i) or isinstance(i, np.ndarray) and i.dtype == np.bool_:
                    (i,) = Const(i, dtype=np.bool_, device=inp.device)(inp)
                else:
                    (i,) = Const(i, dtype=np.int32, device=inp.device)(inp)
                    return i
            assert isinstance(i, (TensorBase, TensorWrapperBase))
            if i.dtype != np.bool_:
                return i
            _, ind = apply(builtin.CondTake(), i, i)
            return ind

        def push(v, item, tensors):
            if v is None:
                item.append(False)
            else:
                item.append(True)
                v = get_index(v)
                assert np.issubdtype(v.dtype, np.integer) or np.issubdtype(
                    v.dtype, np.bool_
                ), "var type in the subscript must be int or bool"
                tensors.append(v)

        if isinstance(i, slice):
            if i.start is None and i.stop is None and i.step is None:
                continue
            push(i.start, item, tensors)
            push(i.stop, item, tensors)
            push(i.step, item, tensors)
            item.append(False)  # idx
        else:
            item += [False,] * 3  # begin, end, stop
            push(i, item, tensors)
        assert len(item) == 5
        items.append(item)
    if new_axes:
        raise IndexError("newaxis is not allowed here")
    return inp, tensors, items, use_subtensor


def try_condtake(tensor, index):
    if not hasattr(index, "dtype") or not hasattr(index, "shape"):
        return []
    if index.dtype != np.bool_ or make_shape_tuple(index.shape) != make_shape_tuple(
        tensor.shape
    ):
        return []
    if isinstance(index, np.ndarray):
        (index,) = Const(index, dtype=np.bool_, device=tensor.device)(tensor)
    assert isinstance(index, (TensorBase, TensorWrapperBase))
    if not isinstance(tensor, (TensorWrapperBase, TensorBase)):
        raise TypeError("input must be a tensor")
    if tensor.device != index.device:
        raise ValueError(
            "ambiguous device: {} vs {}".format(tensor.device, index.device)
        )
    return apply(builtin.CondTake(), tensor, index)


def getitem(tensor, index):
    try_result = try_condtake(tensor, index)
    if len(try_result) == 2:
        return try_result[0]
    tensor, tensors, items, use_subtensor = unpack_getitem(tensor, index)
    for v in tensors:
        if isinstance(v.shape, v.__class__):
            break
        if v.shape[0] == 0:
            (empty_tensor,) = Const([], dtype=tensor.dtype, device=tensor.device)(
                tensor
            )
            return empty_tensor
    if use_subtensor:
        op = builtin.Subtensor(items=items)
    else:
        op = builtin.IndexingMultiAxisVec(items=items)
    (result,) = apply(op, tensor, *tensors)
    return result


def setitem(tensor, index, value):
    org_shape = tensor.shape
    try_result = try_condtake(tensor, index)
    if len(try_result) == 2:
        index = try_result[1]
        if index.shape[0] == 0:
            return tensor
        tensor = tensor.reshape(-1)
    if not isinstance(value, (TensorBase, TensorWrapperBase)):
        op = Const(value, dtype=tensor.dtype, device=tensor.device)
        (value,) = op(tensor)
    tensor, tensors, items, use_subtensor = unpack_getitem(tensor, index)
    for v in tensors:
        if v.shape[0] == 0:
            return tensor
    if use_subtensor:
        op = builtin.Subtensor(items=items)
    else:
        op = builtin.IndexingMultiAxisVec(items=items)
    (tmp_result,) = apply(op, tensor, *tensors)

    # XXX: broadcast can always be applied even if shapes are equal
    if make_shape_tuple(value.shape) != make_shape_tuple(tmp_result.shape):
        for i in range(min(len(value.shape), len(tmp_result.shape))):
            if (
                value.shape[-i - 1] != 1
                and value.shape[-i - 1] != tmp_result.shape[-i - 1]
            ):
                raise ValueError(
                    "cannot copy tensor with shape {} to subtensor with shape {}".format(
                        value.shape, tmp_result.shape
                    )
                )
        value = value._broadcast(tmp_result.shape)
    if use_subtensor:
        op = builtin.SetSubtensor(items=items)
    else:
        op = builtin.IndexingSetMultiAxisVec(items=items)
    (result,) = apply(op, tensor, value, *tensors)
    result = result.reshape(org_shape)
    return result
